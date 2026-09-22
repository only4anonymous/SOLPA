import argparse
import json
import math
import re
import os
import pickle
import sys
import fcntl
import time
import random
import traceback
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import OrderedDict, defaultdict

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)

try:
    # transformers>=4.57 provides Qwen3VLForConditionalGeneration
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None

REASONING_TEMPLATES: List[str] = [
    "Thinking about the latest frames...",
    "Analyzing the current situation...",
    "Considering the frame sequence...",
    "Evaluating the scene...",
    "Processing the visual information...",
]


def select_reasoning_template(seed: int, sample_index: int) -> str:
    rng = random.Random(seed + sample_index)
    return rng.choice(REASONING_TEMPLATES)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import EvalPrediction
from transformers import TrainerCallback

# NOTE: translated from Chinese (import)
from train.l2.unified_metrics import (
    compute_all_metrics,
    merge_global_metrics,
    restore_onestep_metrics_columns,
)
from eval_sync_utils import eval_shard_path, onestep_metrics_relpath
from two_stage_prompting import (
    build_decision_user_prompt,
    build_state_user_prompt,
    build_two_stage_system_prompt,
    format_decision_output_two_stage,
    format_state_output_two_stage,
)
import torch.nn.functional as F


def _safe_dist_barrier(tag: str = "") -> None:
    """
    调用 torch.distributed.barrier，并在 NCCL 场景下显式传入 device_ids，
    以消除 PyTorch 2.4+ 的 “No device id is provided...” 警告。
    """
    try:
        if not (dist.is_available() and dist.is_initialized()):
            return
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    except Exception as e:  # NOTE: translated from Chinese
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"[WARN] barrier{f'({tag})' if tag else ''} 失败: {e}")


def _get_env_silence_and_rank() -> Tuple[bool, bool, int, int]:
    """
    获取静默开关与分布式 rank 信息。
    返回: (silence, is_main, world_size, local_rank)
    """
    import os as _os
    silence = _os.environ.get("L2_SILENCE", "0") == "1"
    try:
        world_size = int(_os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        world_size = 1
    try:
        local_rank = int(_os.environ.get("LOCAL_RANK", _os.environ.get("RANK", "0")))
    except Exception:
        local_rank = 0
    is_main = (local_rank == 0)
    return silence, is_main, world_size, local_rank

# NOTE: Use benchmark one-step planner implementation for GT/eval parity.
# 强制使用 test/onestep_planning/task_planner.py（人机混合二元熵），找不到直接报错
_onestep_planner_dir = str(Path(__file__).resolve().parents[2] / "test" / "onestep_planning")
_onestep_planner_file = Path(_onestep_planner_dir) / "task_planner.py"
if not _onestep_planner_file.exists():
    raise FileNotFoundError(
        f"[train_cot_sft_two_stage] 找不到指定的 task_planner.py: {_onestep_planner_file}\n"
        f"请确保 test/onestep_planning/task_planner.py 存在（人机混合二元熵版本）。不允许 fallback 到其他版本。"
    )
if _onestep_planner_dir not in sys.path:
    sys.path.insert(0, _onestep_planner_dir)
from task_planner import TaskGraphManager, EntropyPlanner


class MarginLoss:
    """
    边际损失类，用于提高正负样本的分离度
    目标：提高正例的准确率，允许召回率适当降低
    
    这个实现专门针对trigger detection任务设计：
    - 对于正样本（需要触发），鼓励模型给出更高的置信度
    - 对于负样本（不需要触发），鼓励模型给出更低的置信度
    - 通过margin来增强正负样本之间的分离度
    """
    def __init__(self, margin: float = 0.5, margin_weight: float = 0.1, 
                 pos_weight: float = 2.0, neg_weight: float = 1.0):
        """
        Args:
            margin: 边际值，控制正负样本之间的最小距离
            margin_weight: 边际损失的权重，控制边际损失在总损失中的比重
            pos_weight: 正样本的权重（默认更高，因为我们要提高正例准确率）
            neg_weight: 负样本的权重
        """
        self.margin = margin
        self.margin_weight = margin_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def compute_margin_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算边际损失
        Args:
            logits: 模型输出的logits [batch_size, seq_len, vocab_size]
            labels: 真实标签 [batch_size, seq_len]，-100表示忽略的位置
        Returns:
            margin_loss: 边际损失值
        """
        # NOTE: translated from Chinese
        valid_mask = labels != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # NOTE: translated from Chinese
        valid_logits = logits[valid_mask]  # [valid_tokens, vocab_size]
        valid_labels = labels[valid_mask]  # [valid_tokens]
        
        # NOTE: translated from Chinese (predict)
        probs = F.softmax(valid_logits, dim=-1)  # [valid_tokens, vocab_size]
        
        # NOTE: translated from Chinese (predict)
        max_probs, predicted_tokens = torch.max(probs, dim=-1)  # [valid_tokens]
        
        # NOTE: translated from Chinese (loss)
        margin_losses = []
        
        # NOTE: translated from Chinese
        for i, (prob, pred_token, true_token) in enumerate(zip(max_probs, predicted_tokens, valid_labels)):
            # NOTE: translated from Chinese (predict)
            if pred_token == true_token:
                # NOTE: translated from Chinese
                if true_token == 1:  # NOTE: translated from Chinese
                    # NOTE: translated from Chinese (loss)
                    loss_val = self.pos_weight * (1.0 - prob) ** 2
                else:  # NOTE: translated from Chinese
                    # NOTE: translated from Chinese
                    loss_val = self.neg_weight * (1.0 - prob) ** 2
            else:
                # NOTE: translated from Chinese (predict)
                if true_token == 1:  # NOTE: translated from Chinese
                    # NOTE: translated from Chinese
                    loss_val = self.pos_weight * (prob + self.margin) ** 2
                else:  # NOTE: translated from Chinese
                    # NOTE: translated from Chinese (weight)
                    loss_val = self.neg_weight * (prob + self.margin) ** 2
            
            margin_losses.append(loss_val)
        
        # NOTE: translated from Chinese (loss)
        if margin_losses:
            return torch.stack(margin_losses).mean()
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)


class TaskStepMapper:
    """
    Task-Step 映射管理器，从 annotation 文件中加载合法的 task-step 组合
    """
    def __init__(self, annotation_path: Optional[str] = None):
        self.task_to_steps: Dict[str, Set[str]] = defaultdict(set)
        self.step_to_task: Dict[str, str] = {}
        
        if annotation_path and os.path.exists(annotation_path):
            self._load_from_annotation(annotation_path)
    
    def _load_from_annotation(self, annotation_path: str):
        """从 annotation 文件中加载 task-step 映射"""
        try:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            with open(annotation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # NOTE: translated from Chinese
            taxonomy = data.get("taxonomy", {})
            vocabulary = data.get("vocabulary", {})
            
            if not taxonomy or not vocabulary:
                if not silence and is_main:
                    print(f"警告：annotation 文件中缺少 taxonomy 或 vocabulary 字段")
                return
            
            # NOTE: translated from Chinese (task, iterate)
            for task_name, step_ids_dict in taxonomy.items():
                if not isinstance(step_ids_dict, dict):
                    continue
                
                task_norm = self._normalize(task_name)
                
                # NOTE: translated from Chinese (task, step, iterate)
                for step_id_str in step_ids_dict.keys():
                    # NOTE: translated from Chinese (step)
                    step_name = vocabulary.get(step_id_str, "")
                    if step_name:
                        step_norm = self._normalize(step_name)
                        self.task_to_steps[task_norm].add(step_norm)
                        self.step_to_task[step_norm] = task_norm
            
            if not silence and is_main:
                print(f"Task-Step 映射加载完成：")
                print(f"  - 任务数: {len(self.task_to_steps)}")
            total_steps = len(set().union(*self.task_to_steps.values()) if self.task_to_steps else set())
            if not silence and is_main:
                print(f"  - 步骤数（去重）: {total_steps}")
            if self.task_to_steps:
                avg_steps = sum(len(steps) for steps in self.task_to_steps.values()) / len(self.task_to_steps)
                if not silence and is_main:
                    print(f"  - 平均每个任务的步骤数: {avg_steps:.1f}")
        except Exception as e:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"警告：加载 Task-Step 映射失败: {e}")
                import traceback
                traceback.print_exc()
    
    def _normalize(self, text: str) -> str:
        """归一化文本（小写、去除空格）"""
        return "".join(text.lower().split())
    
    def is_valid_task_step_pair(self, task: str, step: str) -> bool:
        """检查 task-step 组合是否合法"""
        if not task or not step:
            return True  # NOTE: translated from Chinese (valid)
        
        task_norm = self._normalize(task)
        step_norm = self._normalize(step)
        
        # NOTE: translated from Chinese (check)
        if task_norm in self.task_to_steps:
            return step_norm in self.task_to_steps[task_norm]
        
        # NOTE: translated from Chinese (check)
        if step_norm in self.step_to_task:
            return self.step_to_task[step_norm] == task_norm
        
        # NOTE: translated from Chinese (valid)
        return True


class BeliefStatePostProcessor:
    """
    Lightweight belief-state postprocessor for online eval.

    Goals:
    - keep predictions task-graph consistent
    - reduce invalid task-step combinations
    - maintain a small per-video episodic memory of confirmed step transitions
    """
    def __init__(
        self,
        annotation_path: Optional[str],
        known_tasks: Set[str],
        known_steps: Set[str],
        task_window: int = 6,
        step_window: int = 4,
        future_topk: int = 5,
    ):
        self._known_tasks = {str(x).strip() for x in (known_tasks or set()) if str(x).strip()}
        self._known_steps = {str(x).strip() for x in (known_steps or set()) if str(x).strip()}
        self._task_window = max(1, int(task_window or 1))
        self._step_window = max(1, int(step_window or 1))
        self._future_topk = max(1, int(future_topk or 1))
        self._task_step_mapper = TaskStepMapper(annotation_path) if annotation_path else None
        self._normalized_task_to_name = {self._normalize(x): x for x in self._known_tasks}
        self._video_task_history: Dict[str, List[str]] = defaultdict(list)
        self._video_step_history: Dict[str, List[str]] = defaultdict(list)
        self._video_completed_steps: Dict[str, List[str]] = defaultdict(list)
        self._video_last_task: Dict[str, str] = {}
        self._video_last_step: Dict[str, str] = {}

    def _normalize(self, text: str) -> str:
        return ''.join(str(text or '').lower().split())

    def _append_recent(self, bucket: List[str], value: str, limit: int) -> None:
        if not value:
            return
        bucket.append(value)
        if len(bucket) > limit:
            del bucket[:-limit]

    def _recent_majority(self, values: List[str]) -> str:
        counts: Dict[str, int] = {}
        best_value = ''
        best_count = 0
        for value in values:
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
            if counts[value] >= best_count:
                best_value = value
                best_count = counts[value]
        return best_value

    def _fuzzy_match(self, pred_text: str, candidate_set: Set[str], threshold: float = 0.9) -> Optional[str]:
        if not pred_text or not candidate_set:
            return None
        import difflib
        pred_norm = self._normalize(pred_text)
        best_match = None
        best_ratio = 0.0
        for candidate in candidate_set:
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, pred_norm, self._normalize(candidate)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        if best_ratio >= threshold:
            return best_match
        return None

    def _allowed_steps_for_task(self, task_name: str) -> Set[str]:
        if not task_name or self._task_step_mapper is None:
            return set()
        task_norm = self._normalize(task_name)
        raw_steps = self._task_step_mapper.task_to_steps.get(task_norm, set())
        if not raw_steps:
            return set()
        allowed = set()
        for step_name in self._known_steps:
            if self._normalize(step_name) in raw_steps:
                allowed.add(step_name)
        return allowed

    def _infer_task_from_step(self, step_name: str) -> str:
        if not step_name or self._task_step_mapper is None:
            return ''
        task_norm = self._task_step_mapper.step_to_task.get(self._normalize(step_name), '')
        if not task_norm:
            return ''
        return self._normalized_task_to_name.get(task_norm, '')

    def get_completed_steps(self, video_id: str) -> List[str]:
        return list(self._video_completed_steps.get(video_id or '', []))

    def refine(
        self,
        *,
        video_id: str,
        pred_is: bool,
        pred_task: str,
        pred_step: str,
        pred_future_steps: List[str],
    ) -> Dict[str, Any]:
        video_id = str(video_id or '').strip()
        if not pred_is:
            return {'task': '', 'step': '', 'future_steps': []}

        task_name = self._fuzzy_match(pred_task, self._known_tasks, threshold=0.88) or ''
        if not task_name:
            task_name = self._video_last_task.get(video_id, '')

        allowed_steps = self._allowed_steps_for_task(task_name)
        step_candidates = allowed_steps or self._known_steps
        step_name = self._fuzzy_match(pred_step, step_candidates, threshold=0.82 if allowed_steps else 0.88) or ''

        filtered_future_steps: List[str] = []
        for raw_step in list(pred_future_steps or []):
            matched = self._fuzzy_match(raw_step, step_candidates, threshold=0.82 if allowed_steps else 0.88) or ''
            if not matched:
                continue
            if task_name and self._task_step_mapper is not None and not self._task_step_mapper.is_valid_task_step_pair(task_name, matched):
                continue
            if matched not in filtered_future_steps:
                filtered_future_steps.append(matched)

        if not task_name and step_name:
            task_name = self._infer_task_from_step(step_name) or task_name
            allowed_steps = self._allowed_steps_for_task(task_name)
        if not task_name and filtered_future_steps:
            task_name = self._infer_task_from_step(filtered_future_steps[0]) or task_name
            allowed_steps = self._allowed_steps_for_task(task_name)

        if task_name and step_name and self._task_step_mapper is not None and not self._task_step_mapper.is_valid_task_step_pair(task_name, step_name):
            step_name = ''

        if not step_name and filtered_future_steps:
            step_name = filtered_future_steps[0]

        last_task = self._video_last_task.get(video_id, '')
        last_step = self._video_last_step.get(video_id, '')

        if not task_name:
            task_name = self._recent_majority(self._video_task_history.get(video_id, [])) or last_task
        if not step_name and task_name == last_task:
            step_name = self._recent_majority(self._video_step_history.get(video_id, [])) or last_step

        if step_name:
            filtered_future_steps = [x for x in filtered_future_steps if x != step_name]

        if task_name:
            self._append_recent(self._video_task_history[video_id], task_name, self._task_window)
            self._video_last_task[video_id] = task_name
        if step_name:
            if last_task == task_name and last_step and last_step != step_name:
                completed = self._video_completed_steps[video_id]
                if last_step not in completed:
                    completed.append(last_step)
            self._append_recent(self._video_step_history[video_id], step_name, self._step_window)
            self._video_last_step[video_id] = step_name

        return {
            'task': task_name,
            'step': step_name,
            'future_steps': filtered_future_steps[: self._future_topk],
        }


class MarginSFTTrainer(SFTTrainer):
    """
    自定义的SFTTrainer，使用边际损失来提高正负样本的分离度
    """
    def __init__(
        self,
        margin_loss: MarginLoss = None,
        processor=None,
        trigger_loss_weight: float = 1.0,
        bind_trigger_task: bool = False,
        bind_task_step: bool = False,
        bind_loss_weight: float = 0.1,
        bind_tt_weight: float = None,
        bind_ts_weight: float = None,
        bind_trigger_disc_weight: float = 0.1,
        enable_task_step_constraint: bool = False,
        task_step_constraint_weight: float = 0.5,
        annotation_path: Optional[str] = None,
        class_weighted: bool = False,
        train_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        # NOTE: translated from Chinese
        self._debug_processor = processor
        self._classification_weight = float(trigger_loss_weight)
        # NOTE: translated from Chinese (added, binding, config)
        self._bind_trigger_task = bool(bind_trigger_task)
        self._bind_task_step = bool(bind_task_step)
        self._bind_loss_weight = float(bind_loss_weight)
        # NOTE: translated from Chinese (weight)
        self._bind_tt_weight = float(bind_tt_weight) if bind_tt_weight is not None else float(bind_loss_weight)
        self._bind_ts_weight = float(bind_ts_weight) if bind_ts_weight is not None else float(bind_loss_weight)
        # NOTE: translated from Chinese (added, weight, loss)
        self._bind_trigger_disc_weight = float(bind_trigger_disc_weight)
        # NOTE: translated from Chinese
        self._bind_temperature = 0.07
        # NOTE: translated from Chinese (added, config)
        self._enable_task_step_constraint = bool(enable_task_step_constraint)
        self._task_step_constraint_weight = float(task_step_constraint_weight)
        self._task_step_mapper = None
        if self._enable_task_step_constraint and annotation_path:
            self._task_step_mapper = TaskStepMapper(annotation_path)
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"✅ Task-Step 约束已启用，权重={self._task_step_constraint_weight}")

        tokenizer = None
        if processor is not None:
            tokenizer = getattr(processor, "tokenizer", None)
        # NOTE: translated from Chinese (cache)
        self._tokenizer = tokenizer
        
        # NOTE: translated from Chinese (added, config)
        self._class_weighted = bool(class_weighted)
        self._task_weights = None  # NOTE: translated from Chinese (weight)
        self._step_weights = None  # NOTE: translated from Chinese (weight)
        
        if self._class_weighted and train_dataset is not None and tokenizer is not None:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print("🔍 开始统计训练集中的 task 和 step 分布...")
            self._compute_class_weights(train_dataset, tokenizer)
            if not silence and is_main:
                print(f"✅ 类别加权已启用")
                if self._task_weights:
                    print(f"   - Task类别数: {len(self._task_weights)}")
                if self._step_weights:
                    print(f"   - Step类别数: {len(self._step_weights)}")
        # NOTE: translated from Chinese
        kwargs.pop('processor', None)
        # NOTE: translated from Chinese
        if "train_dataset" not in kwargs and train_dataset is not None:
            kwargs["train_dataset"] = train_dataset
        super().__init__(**kwargs)
        self.margin_loss = margin_loss
        self._debug_counter = 0  # NOTE: translated from Chinese (debug)
        self._debug_print_limit = 10  # NOTE: translated from Chinese
        self._cls_true_id = None
        self._cls_false_id = None
        if tokenizer is not None:
            true_ids = tokenizer.encode("true", add_special_tokens=False)
            false_ids = tokenizer.encode("false", add_special_tokens=False)
            if true_ids and false_ids:
                self._cls_true_id = int(true_ids[0])
                self._cls_false_id = int(false_ids[0])
        # NOTE: print at most a few sample snippets for failure context.
        self._error_trace_max_samples = max(1, int(os.environ.get("COT_ERR_TRACE_MAX_SAMPLES", "2")))

    def _print_failure_context(self, stage: str, exc: Exception, inputs: Dict[str, Any], outputs: Any = None) -> None:
        """Print detailed failure context with traceback and sample snippets."""
        silence, is_main, world_size, local_rank = _get_env_silence_and_rank()
        if silence and not is_main:
            return
        try:
            print("\n" + "=" * 96, flush=True)
            print(f"[FATAL][compute_loss][{stage}] rank={local_rank}/{world_size} error={repr(exc)}", flush=True)
            print("[Traceback]", flush=True)
            print(traceback.format_exc(), flush=True)
            if isinstance(inputs, dict):
                in_ids = inputs.get("input_ids")
                lbs = inputs.get("labels")
                tpos = inputs.get("trigger_position")
                tlab = inputs.get("trigger_label")
                if isinstance(in_ids, torch.Tensor):
                    print(f"  input_ids.shape={tuple(in_ids.shape)} dtype={in_ids.dtype} device={in_ids.device}", flush=True)
                if isinstance(lbs, torch.Tensor):
                    print(f"  labels.shape={tuple(lbs.shape)} dtype={lbs.dtype} device={lbs.device}", flush=True)
                if isinstance(tpos, torch.Tensor):
                    print(f"  trigger_position.shape={tuple(tpos.shape)} min={int(tpos.min().item())} max={int(tpos.max().item())}", flush=True)
                if isinstance(tlab, torch.Tensor):
                    uniq = tlab.detach().cpu().unique().tolist()
                    print(f"  trigger_label.shape={tuple(tlab.shape)} unique={uniq[:8]}", flush=True)
                dbg_prompt = inputs.get("debug_prompt_text")
                dbg_target = inputs.get("debug_target_text")
                if isinstance(dbg_prompt, list) and len(dbg_prompt) > 0:
                    n = min(self._error_trace_max_samples, len(dbg_prompt))
                    for i in range(n):
                        ptxt = str(dbg_prompt[i])[:300].replace("\n", "\\n")
                        ttxt = ""
                        if isinstance(dbg_target, list) and i < len(dbg_target):
                            ttxt = str(dbg_target[i])[:300].replace("\n", "\\n")
                        print(f"  sample[{i}] prompt={ptxt}", flush=True)
                        if ttxt:
                            print(f"  sample[{i}] target={ttxt}", flush=True)
            if outputs is not None and hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
                lg = outputs.logits
                print(f"  outputs.logits.shape={tuple(lg.shape)} dtype={lg.dtype} device={lg.device}", flush=True)
            print("=" * 96 + "\n", flush=True)
        except Exception:
            print("[FATAL] Failed to print failure context.", flush=True)
            print(traceback.format_exc(), flush=True)

    def _compute_class_weights(self, train_dataset, tokenizer):
        """
        统计训练集中 task 和 step 的分布，并计算类别权重
        使用 inverse frequency 方法：weight = total_samples / (num_classes * class_count)
        """
        from collections import Counter
        import os
        import json
        
        # NOTE: translated from Chinese
        silence, is_main, world_size, local_rank = _get_env_silence_and_rank()

        # NOTE: translated from Chinese (weight, cache)
        cache_dir = "<REPO_ROOT>/data/l1/class_weight"
        os.makedirs(cache_dir, exist_ok=True)

        dataset_tag = "default"
        jsonl_path = getattr(train_dataset, "_jsonl_path", None)
        if isinstance(jsonl_path, str) and jsonl_path:
            base_name = os.path.basename(jsonl_path)
            dataset_tag = os.path.splitext(base_name)[0] or "default"
        tag_lower = dataset_tag.lower()
        if "tiny" in tag_lower and not dataset_tag.endswith("_tiny"):
            dataset_tag = f"{dataset_tag}_tiny"

        cache_path = os.path.join(cache_dir, f"class_weights_{dataset_tag}.json")

        if os.path.exists(cache_path):
            try:
                if not silence and is_main:
                    print(f"检测到已存在的类别权重缓存: {cache_path}，正在加载...")
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                task_w = data.get("task_weights") or {}
                step_w = data.get("step_weights") or {}
                # NOTE: translated from Chinese
                self._task_weights = {int(k): float(v) for k, v in task_w.items()}
                self._step_weights = {int(k): float(v) for k, v in step_w.items()}

                if not silence and is_main:
                    print(f"✅ 类别权重已从缓存加载完成（task={len(self._task_weights)}, step={len(self._step_weights)}）")
                return
            except Exception as e:
                if not silence and is_main:
                    print(f"⚠️ 加载类别权重缓存失败，将重新统计: {e}")

        # NOTE: translated from Chinese (cache, stats)
        task_token_counter = Counter()
        step_token_counter = Counter()
        total_samples = 0
        
        if not silence and is_main:
            print("正在遍历训练集统计 task 和 step 分布...")
        for idx in tqdm(
            range(len(train_dataset)),
            desc="统计类别分布",
            disable=(silence and not is_main),
        ):
            try:
                sample = train_dataset[idx]
                labels = sample.get("labels")
                task_region = sample.get("task_region")
                step_region = sample.get("step_region")
                
                if labels is None:
                    continue
                
                total_samples += 1
                
                # NOTE: translated from Chinese (stats)
                if task_region is not None and isinstance(task_region, (list, tuple)) and len(task_region) == 2:
                    ts, te = task_region
                    if ts >= 0 and te > ts and te <= len(labels):
                        task_label_ids = labels[ts:te]
                        if isinstance(task_label_ids, torch.Tensor):
                            task_label_ids = task_label_ids.cpu().tolist()
                        # NOTE: translated from Chinese (filter)
                        valid_task_ids = [tid for tid in task_label_ids if tid != -100]
                        task_token_counter.update(valid_task_ids)
                
                # NOTE: translated from Chinese (stats)
                if step_region is not None and isinstance(step_region, (list, tuple)) and len(step_region) == 2:
                    ss, se = step_region
                    if ss >= 0 and se > ss and se <= len(labels):
                        step_label_ids = labels[ss:se]
                        if isinstance(step_label_ids, torch.Tensor):
                            step_label_ids = step_label_ids.cpu().tolist()
                        # NOTE: translated from Chinese (filter)
                        valid_step_ids = [sid for sid in step_label_ids if sid != -100]
                        step_token_counter.update(valid_step_ids)
                        
            except Exception:
                # NOTE: translated from Chinese
                continue
        
        if not silence and is_main:
            print(f"统计完成：总样本数 = {total_samples}")
            print(f"  - Task token 类型数: {len(task_token_counter)}")
            print(f"  - Step token 类型数: {len(step_token_counter)}")
        
        # NOTE: translated from Chinese (weight)
        # NOTE: translated from Chinese
        #   raw_w[c] = log(total_count / count[c]) + 1
        #   weight[c] = 1.0 + (raw_w[c] - min_raw) / (max_raw - min_raw + 1e-12)
        # NOTE: translated from Chinese
        
        if task_token_counter:
            total_task_tokens = sum(task_token_counter.values())
            raw_task_weights = {}
            for token_id, count in task_token_counter.items():
                raw_w = np.log(total_task_tokens / count) + 1.0
                raw_task_weights[token_id] = float(raw_w)

            raw_vals = np.array(list(raw_task_weights.values()), dtype=np.float32)
            min_raw = float(raw_vals.min())
            max_raw = float(raw_vals.max())
            self._task_weights = {}
            if max_raw <= min_raw + 1e-12:
                # NOTE: translated from Chinese
                for token_id in raw_task_weights.keys():
                    self._task_weights[token_id] = 1.0
            else:
                scale = 1.0 / (max_raw - min_raw + 1e-12)
                for token_id, raw_w in raw_task_weights.items():
                    norm_w = 1.0 + (raw_w - min_raw) * scale  # NOTE: translated from Chinese
                    self._task_weights[token_id] = float(norm_w)
            
            # NOTE: translated from Chinese (weight, stats)
            if not silence and is_main:
                weights_list = list(self._task_weights.values())
                print(f"  Task权重统计: min={min(weights_list):.3f}, max={max(weights_list):.3f}, "
                      f"mean={np.mean(weights_list):.3f}, median={np.median(weights_list):.3f}")
        
        if step_token_counter:
            total_step_tokens = sum(step_token_counter.values())
            raw_step_weights = {}
            for token_id, count in step_token_counter.items():
                raw_w = np.log(total_step_tokens / count) + 1.0
                raw_step_weights[token_id] = float(raw_w)

            raw_vals = np.array(list(raw_step_weights.values()), dtype=np.float32)
            min_raw = float(raw_vals.min())
            max_raw = float(raw_vals.max())
            self._step_weights = {}
            if max_raw <= min_raw + 1e-12:
                for token_id in raw_step_weights.keys():
                    self._step_weights[token_id] = 1.0
            else:
                scale = 1.0 / (max_raw - min_raw + 1e-12)
                for token_id, raw_w in raw_step_weights.items():
                    norm_w = 1.0 + (raw_w - min_raw) * scale
                    self._step_weights[token_id] = float(norm_w)
            
            # NOTE: translated from Chinese (weight, stats)
            if not silence and is_main:
                weights_list = list(self._step_weights.values())
                print(f"  Step权重统计: min={min(weights_list):.3f}, max={max(weights_list):.3f}, "
                      f"mean={np.mean(weights_list):.3f}, median={np.median(weights_list):.3f}")

        # NOTE: translated from Chinese (cache)
        try:
            to_save = {
                "dataset_tag": dataset_tag,
                "task_weights": {str(k): float(v) for k, v in (self._task_weights or {}).items()},
                "step_weights": {str(k): float(v) for k, v in (self._step_weights or {}).items()},
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
            if not silence and is_main:
                print(f"✅ 已将类别权重保存到缓存文件: {cache_path}")
        except Exception as e:
            if not silence and is_main:
                print(f"⚠️ 保存类别权重缓存失败: {e}")
        
        # NOTE: translated from Chinese
        if task_token_counter:
            print("\n  Task token 示例 (最常见的5个):")
            for token_id, count in task_token_counter.most_common(5):
                token_text = tokenizer.decode([token_id])
                weight = self._task_weights.get(token_id, 1.0)
                print(f"    Token '{token_text}' (id={token_id}): count={count}, weight={weight:.3f}")
        
        if step_token_counter:
            print("\n  Step token 示例 (最常见的5个):")
            for token_id, count in step_token_counter.most_common(5):
                token_text = tokenizer.decode([token_id])
                weight = self._step_weights.get(token_id, 1.0)
                print(f"    Token '{token_text}' (id={token_id}): count={count}, weight={weight:.3f}")

    def _compute_hierarchical_bind_loss(self, hidden_states, inputs, logits=None):
        """
        多模态层级绑定 loss（改进版）：
        - trigger-task：对齐 trigger token 向量 与 task span 向量
        - task-step   ：对齐 task span 向量 与 step span 向量
        使用 InfoNCE / 对比学习形式，自动利用 batch 内的负样本
        
        【新增】同时保持 trigger 的判别性，避免 Binding Loss 破坏 trigger 预测能力
        """
        import torch

        device = hidden_states.device
        B, T, H = hidden_states.shape

        trigger_label = inputs.get("trigger_label")       # [B]
        trigger_pos = inputs.get("trigger_position")      # [B]
        task_regions = inputs.get("task_region")          # list of [start, end]
        step_regions = inputs.get("step_region")          # list of [start, end]

        if trigger_label is None or trigger_pos is None:
            return None

        if isinstance(trigger_label, torch.Tensor):
            trigger_label = trigger_label.to(device)
        else:
            trigger_label = torch.tensor(trigger_label, device=device)

        if isinstance(trigger_pos, torch.Tensor):
            trigger_pos = trigger_pos.to(device)
        else:
            trigger_pos = torch.tensor(trigger_pos, device=device)

        # NOTE: translated from Chinese (binding)
        pos_mask = (trigger_label == 1)
        neg_mask = (trigger_label == 0)
        if pos_mask.sum() == 0:
            return None

        # NOTE: translated from Chinese
        trig_vecs_tt, task_vecs_tt = [], []   # NOTE: translated from Chinese (binding)
        task_vecs_ts, step_vecs_ts = [], []   # NOTE: translated from Chinese (binding)
        # NOTE: translated from Chinese (added)
        pos_trig_vecs = []
        neg_trig_vecs = []
        need_disc_loss = (self._bind_trigger_task and self._bind_trigger_disc_weight != 0)

        for b in range(B):
            trig_idx = int(trigger_pos[b].item())
            trig_idx = max(0, min(trig_idx, T - 1))
            trig_vec = hidden_states[b, trig_idx]  # [H]
            
            # NOTE: translated from Chinese (loss)
            if need_disc_loss:
                if bool(pos_mask[b]):
                    pos_trig_vecs.append(trig_vec)
                elif bool(neg_mask[b]):
                    neg_trig_vecs.append(trig_vec)
            
            if not bool(pos_mask[b]):
                continue

            # NOTE: translated from Chinese (binding)
            if self._bind_trigger_task and task_regions is not None:
                # NOTE: translated from Chinese (check)
                if b < len(task_regions):
                    region = task_regions[b]
                    if isinstance(region, (list, tuple)) and len(region) == 2:
                        s, e = int(region[0]), int(region[1])
                        # NOTE: translated from Chinese
                        if s >= 0 and e > s and e <= T:
                            span = hidden_states[b, s:e]          # [L, H]
                            if span.size(0) > 0:  # NOTE: translated from Chinese
                                task_vec = span.mean(dim=0)           # [H]
                                # NOTE: translated from Chinese (check)
                                if torch.isfinite(task_vec).all() and torch.isfinite(trig_vec).all():
                                    trig_vecs_tt.append(trig_vec)
                                    task_vecs_tt.append(task_vec)

            # NOTE: translated from Chinese (binding)
            if self._bind_task_step and task_regions is not None and step_regions is not None:
                # NOTE: translated from Chinese (check)
                if b < len(task_regions) and b < len(step_regions):
                    treg = task_regions[b]
                    sreg = step_regions[b]
                    if (isinstance(treg, (list, tuple)) and len(treg) == 2 and
                        isinstance(sreg, (list, tuple)) and len(sreg) == 2):
                        ts, te = int(treg[0]), int(treg[1])
                        ss, se = int(sreg[0]), int(sreg[1])
                        # NOTE: translated from Chinese
                        if (ts >= 0 and te > ts and te <= T and
                            ss >= 0 and se > ss and se <= T):
                            t_span = hidden_states[b, ts:te]     # [Lt, H]
                            s_span = hidden_states[b, ss:se]     # [Ls, H]
                            if t_span.size(0) > 0 and s_span.size(0) > 0:  # NOTE: translated from Chinese
                                t_vec = t_span.mean(dim=0)           # [H]
                                s_vec = s_span.mean(dim=0)           # [H]
                                # NOTE: translated from Chinese (check)
                                if torch.isfinite(t_vec).all() and torch.isfinite(s_vec).all():
                                    task_vecs_ts.append(t_vec)
                                    step_vecs_ts.append(s_vec)

        loss_tt = None
        loss_ts = None
        loss_trigger_disc = None  # NOTE: translated from Chinese (added, loss)
        temp = getattr(self, "_bind_temperature", 0.07)

        # NOTE: translated from Chinese
        # NOTE: translated from Chinese (loss)
        if self._bind_trigger_task and len(trig_vecs_tt) > 0:
            trig_vecs = torch.stack(trig_vecs_tt, dim=0)    # NOTE: translated from Chinese
            task_vecs = torch.stack(task_vecs_tt, dim=0)    # NOTE: translated from Chinese
            # NOTE: translated from Chinese (check)
            trig_norm = torch.norm(trig_vecs, p=2, dim=-1, keepdim=True)
            task_norm = torch.norm(task_vecs, p=2, dim=-1, keepdim=True)
            trig_vecs = trig_vecs / (trig_norm + 1e-8)
            task_vecs = task_vecs / (task_norm + 1e-8)
            P = trig_vecs.size(0)
            if P > 1:
                # NOTE: translated from Chinese
                sim = trig_vecs @ task_vecs.t() / temp             # [P, P]
                labels = torch.arange(P, device=device)  # NOTE: translated from Chinese
                loss_tt = 0.5 * (
                    F.cross_entropy(sim, labels) +        # NOTE: translated from Chinese
                    F.cross_entropy(sim.t(), labels)       # NOTE: translated from Chinese
                )
            else:
                # NOTE: translated from Chinese
                cos_sim = F.cosine_similarity(trig_vecs[0:1], task_vecs[0:1], dim=-1)
                loss_tt = 1.0 - cos_sim[0]
            # NOTE: translated from Chinese (check)
            if not torch.isfinite(loss_tt):
                loss_tt = None

        # NOTE: translated from Chinese
        # NOTE: translated from Chinese (loss)
        if self._bind_task_step and len(task_vecs_ts) > 0:
            task_vecs = torch.stack(task_vecs_ts, dim=0)    # NOTE: translated from Chinese
            step_vecs = torch.stack(step_vecs_ts, dim=0)    # NOTE: translated from Chinese
            # NOTE: translated from Chinese (check)
            task_norm = torch.norm(task_vecs, p=2, dim=-1, keepdim=True)
            step_norm = torch.norm(step_vecs, p=2, dim=-1, keepdim=True)
            task_vecs = task_vecs / (task_norm + 1e-8)
            step_vecs = step_vecs / (step_norm + 1e-8)
            Q = task_vecs.size(0)
            if Q > 1:
                # NOTE: translated from Chinese
                sim = task_vecs @ step_vecs.t() / temp
                labels = torch.arange(Q, device=device)  # NOTE: translated from Chinese
                loss_ts = 0.5 * (
                    F.cross_entropy(sim, labels) +        # NOTE: translated from Chinese
                    F.cross_entropy(sim.t(), labels)      # NOTE: translated from Chinese
                )
            else:
                cos_sim = F.cosine_similarity(task_vecs[0:1], step_vecs[0:1], dim=-1)
                loss_ts = 1.0 - cos_sim[0]
            # NOTE: translated from Chinese (check)
            if not torch.isfinite(loss_ts):
                loss_ts = None

        # NOTE: translated from Chinese (added, loss)
        # NOTE: translated from Chinese (predict)
        # NOTE: translated from Chinese
        if (self._bind_trigger_task and 
            self._bind_trigger_disc_weight != 0 and 
            len(pos_trig_vecs) > 0 and 
            len(neg_trig_vecs) > 0):
            pos_vecs = torch.stack(pos_trig_vecs, dim=0)  # [P, H]
            neg_vecs = torch.stack(neg_trig_vecs, dim=0)  # [N, H]
            
            # NOTE: translated from Chinese
            pos_center = pos_vecs.mean(dim=0)  # [H]
            neg_center = neg_vecs.mean(dim=0)  # [H]
            
            # NOTE: translated from Chinese
            separation = torch.norm(pos_center - neg_center)
            
            # NOTE: translated from Chinese (loss)
            # NOTE: translated from Chinese (weight)
            loss_trigger_disc = -separation * 0.01  # NOTE: translated from Chinese (weight)
            
            # NOTE: translated from Chinese (check)
            if not torch.isfinite(loss_trigger_disc):
                loss_trigger_disc = None
        else:
            loss_trigger_disc = None

        # NOTE: translated from Chinese (weight)
        return {
            "tt": loss_tt if loss_tt is not None and torch.isfinite(loss_tt) else None,
            "ts": loss_ts if loss_ts is not None and torch.isfinite(loss_ts) else None,
            "trigger_disc": loss_trigger_disc if loss_trigger_disc is not None and torch.isfinite(loss_trigger_disc) else None,
        }
    
    def _compute_task_step_constraint_loss(self, logits, inputs):
        """
        计算 Task-Step 约束损失：对不合法的 task-step 组合施加额外惩罚
        """
        import torch
        
        task_regions = inputs.get("task_region")
        step_regions = inputs.get("step_region")
        labels = inputs.get("labels")
        trigger_label = inputs.get("trigger_label")
        
        if task_regions is None or step_regions is None or labels is None or trigger_label is None:
            return None
        
        # NOTE: translated from Chinese (cache)
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return None
        
        device = logits.device
        batch_size = logits.size(0)
        constraint_losses = []
        
        for b in range(batch_size):
            # NOTE: translated from Chinese (check)
            if trigger_label[b].item() != 1:
                continue
            
            task_region = task_regions[b] if b < len(task_regions) else None
            step_region = step_regions[b] if b < len(step_regions) else None
            
            if (not isinstance(task_region, (list, tuple)) or len(task_region) != 2 or
                not isinstance(step_region, (list, tuple)) or len(step_region) != 2):
                continue
            
            ts, te = int(task_region[0]), int(task_region[1])
            ss, se = int(step_region[0]), int(step_region[1])
            
            # NOTE: translated from Chinese (check)
            if ts < 0 or te <= ts or ss < 0 or se <= ss:
                continue
            
            # NOTE: translated from Chinese
            try:
                task_label_ids = labels[b, ts:te].cpu().tolist()
                step_label_ids = labels[b, ss:se].cpu().tolist()
                
                # NOTE: translated from Chinese (filter)
                task_label_ids = [tid for tid in task_label_ids if tid != -100]
                step_label_ids = [sid for sid in step_label_ids if sid != -100]
                
                if not task_label_ids or not step_label_ids:
                    continue
                
                task_text = tokenizer.decode(task_label_ids, skip_special_tokens=True).strip()
                step_text = tokenizer.decode(step_label_ids, skip_special_tokens=True).strip()
                
                # NOTE: translated from Chinese (check, valid)
                if not self._task_step_mapper.is_valid_task_step_pair(task_text, step_text):
                    # NOTE: translated from Chinese (valid)
                    step_logits = logits[b, ss:se]  # [L, vocab_size]
                    step_labels = labels[b, ss:se]  # [L]
                    
                    # NOTE: translated from Chinese (loss)
                    valid_mask = step_labels != -100
                    if valid_mask.any():
                        valid_logits = step_logits[valid_mask]
                        valid_labels = step_labels[valid_mask]
                        penalty = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
                        if torch.isfinite(penalty):
                            constraint_losses.append(penalty)
            except Exception:
                # NOTE: translated from Chinese
                continue
        
        if not constraint_losses:
            return None
        
        return sum(constraint_losses) / len(constraint_losses)
    
    def _compute_weighted_task_step_loss(self, logits, inputs):
        """
        计算类别加权的 Task 和 Step CE 损失
        
        Args:
            logits: 模型输出的 logits [batch_size, seq_len, vocab_size]
            inputs: 包含 labels, task_region, step_region 的输入字典
            
        Returns:
            task_weighted_loss, step_weighted_loss
        """
        import torch
        import torch.nn.functional as F
        
        task_regions = inputs.get("task_region")
        step_regions = inputs.get("step_region")
        labels = inputs.get("labels")
        
        if labels is None:
            return None, None
        
        device = logits.device
        batch_size = logits.size(0)
        
        task_losses = []
        step_losses = []
        
        for b in range(batch_size):
            task_region = task_regions[b] if task_regions is not None and b < len(task_regions) else None
            step_region = step_regions[b] if step_regions is not None and b < len(step_regions) else None
            
            # NOTE: translated from Chinese
            if (self._task_weights is not None and task_region is not None and 
                isinstance(task_region, (list, tuple)) and len(task_region) == 2):
                ts, te = int(task_region[0]), int(task_region[1])
                
                if ts >= 0 and te > ts and te <= logits.size(1):
                    task_logits = logits[b, ts:te]  # [L, vocab_size]
                    task_labels = labels[b, ts:te]  # [L]
                    
                    # NOTE: translated from Chinese (loss)
                    valid_mask = task_labels != -100
                    if valid_mask.any():
                        valid_logits = task_logits[valid_mask]  # [N, vocab_size]
                        valid_labels = task_labels[valid_mask]  # [N]
                        
                        # NOTE: translated from Chinese (weight)
                        token_weights = torch.ones(len(valid_labels), device=device)
                        for i, label_id in enumerate(valid_labels):
                            label_id_int = int(label_id.item())
                            if label_id_int in self._task_weights:
                                token_weights[i] = self._task_weights[label_id_int]
                        
                        # NOTE: translated from Chinese (loss)
                        # NOTE: translated from Chinese (weight, loss)
                        per_token_loss = F.cross_entropy(
                            valid_logits, valid_labels, reduction='none'
                        )
                        weighted_loss = (per_token_loss * token_weights).mean()
                        
                        if torch.isfinite(weighted_loss):
                            task_losses.append(weighted_loss)
            
            # NOTE: translated from Chinese
            if (self._step_weights is not None and step_region is not None and 
                isinstance(step_region, (list, tuple)) and len(step_region) == 2):
                ss, se = int(step_region[0]), int(step_region[1])
                
                if ss >= 0 and se > ss and se <= logits.size(1):
                    step_logits = logits[b, ss:se]  # [L, vocab_size]
                    step_labels = labels[b, ss:se]  # [L]
                    
                    # NOTE: translated from Chinese (loss)
                    valid_mask = step_labels != -100
                    if valid_mask.any():
                        valid_logits = step_logits[valid_mask]  # [N, vocab_size]
                        valid_labels = step_labels[valid_mask]  # [N]
                        
                        # NOTE: translated from Chinese (weight)
                        token_weights = torch.ones(len(valid_labels), device=device)
                        for i, label_id in enumerate(valid_labels):
                            label_id_int = int(label_id.item())
                            if label_id_int in self._step_weights:
                                token_weights[i] = self._step_weights[label_id_int]
                        
                        # NOTE: translated from Chinese (loss)
                        per_token_loss = F.cross_entropy(
                            valid_logits, valid_labels, reduction='none'
                        )
                        weighted_loss = (per_token_loss * token_weights).mean()
                        
                        if torch.isfinite(weighted_loss):
                            step_losses.append(weighted_loss)
        
        # NOTE: translated from Chinese (loss)
        task_weighted_loss = None
        if task_losses:
            task_weighted_loss = sum(task_losses) / len(task_losses)
        
        step_weighted_loss = None
        if step_losses:
            step_weighted_loss = sum(step_losses) / len(step_losses)
        
        return task_weighted_loss, step_weighted_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss方法，添加边际损失
        """
        # NOTE: translated from Chinese (debug)
        if self._debug_counter < self._debug_print_limit:
            batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
            
            # NOTE: translated from Chinese (cache)
            tokenizer = getattr(self, "_tokenizer", None)
            
            # NOTE: translated from Chinese
            # if "labels" in inputs and tokenizer is not None:
            #     labels = inputs["labels"]
            #     input_ids = inputs.get("input_ids", None)
                
            # NOTE: translated from Chinese
            #         print(f"\n{'='*80}")
            #         print(f"DEBUG Sample {self._debug_counter * batch_size + b_idx + 1}:")
                    
            # NOTE: translated from Chinese
            #         if input_ids is not None:
            #             input_seq = input_ids[b_idx].cpu().tolist()
            #             label_seq = labels[b_idx].cpu().tolist()
                        
            # NOTE: translated from Chinese
            #             pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
            #             valid_len = (input_ids[b_idx] != pad_token_id).sum().item()
            #             if pad_token_id is None or pad_token_id == 0:
            #                 valid_len = len(input_seq)
                        
            # NOTE: translated from Chinese
            #             supervised_positions = [(i, label) for i, label in enumerate(label_seq[:valid_len]) if label != -100]
            # NOTE: translated from Chinese
            #             if supervised_positions:
            # NOTE: translated from Chinese
            #                     input_token = input_seq[pos]
            # NOTE: translated from Chinese
            #                     try:
            #                         tokens = tokenizer.decode([input_token])
            #                         label_tokens = tokenizer.decode([label_id])
            #                     except:
            #                         tokens = f"<{input_token}>"
            #                         label_tokens = f"<{label_id}>"
            #                     print(f"    pos={pos}: input='{tokens.strip()}' -> label='{label_tokens.strip()}' (label_id={label_id})")
                        
            # NOTE: translated from Chinese
            #             full_text = tokenizer.decode(input_seq[:valid_len], skip_special_tokens=False)
            # NOTE: translated from Chinese
                    
            self._debug_counter += batch_size
        
        # NOTE: translated from Chinese (loss)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        try:
            outputs = model(**inputs)
        except Exception as e:
            self._print_failure_context("model_forward", e, inputs, outputs=None)
            raise
        
        if self.label_smoother is not None and labels is not None:
            # NOTE: translated from Chinese (loss)
            loss = self.label_smoother(outputs, labels)
        else:
            # NOTE: translated from Chinese (loss)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            if loss is None:
                # NOTE: translated from Chinese (loss)
                logits = outputs.logits
                labels = inputs.get("labels")
                if labels is not None:
                    # NOTE: translated from Chinese (debug)
                    tokenizer = getattr(self, "_tokenizer", None)
                    
                    if self._debug_counter <= self._debug_print_limit and tokenizer is not None:
                        silence, is_main, _, _ = _get_env_silence_and_rank()
                        if not silence and is_main:
                            print(f"\n{'='*80}")
                            print("DEBUG Loss Calculation:")
                            print(f"  logits shape: {logits.shape}")
                            print(f"  labels shape: {labels.shape}")
                        
                        # NOTE: translated from Chinese
                        valid_mask = labels != -100
                        num_valid = valid_mask.sum().item()
                        if not silence and is_main:
                            print(f"  Valid labels (non -100): {num_valid}/{labels.numel()}")
                        
                        # NOTE: translated from Chinese
                        raw_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), 
                            labels.view(-1), 
                            ignore_index=-100,
                            reduction='none'
                        )
                        actual_loss = raw_loss[valid_mask.view(-1)]
                        if len(actual_loss) > 0 and not silence and is_main:
                            print(f"  Loss per valid token: mean={actual_loss.mean():.4f}, min={actual_loss.min():.4f}, max={actual_loss.max():.4f}")
                    
                    # Guard invalid label ids to avoid device-side asserts in CE.
                    vocab_size = int(logits.size(-1))
                    invalid_mask = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
                    if invalid_mask.any():
                        bad_count = int(invalid_mask.sum().item())
                        bad_vals = labels[invalid_mask].detach().cpu()
                        uniq_bad = bad_vals.unique().tolist()[:16]
                        silence, is_main, world_size, local_rank = _get_env_silence_and_rank()
                        if not (silence and not is_main):
                            print(
                                f"[WARN][CE] rank={local_rank}/{world_size} invalid labels found: "
                                f"count={bad_count}, vocab_size={vocab_size}, unique={uniq_bad}. "
                                f"Will set them to -100 to continue.",
                                flush=True,
                            )
                        labels = labels.clone()
                        labels[invalid_mask] = -100
                    try:
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
                    except Exception as e:
                        self._print_failure_context("ce_loss", e, inputs, outputs)
                        raise
                else:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # NOTE: translated from Chinese
        loss_components = {"ce_loss": float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss)}
        
        # NOTE: translated from Chinese (loss)
        margin_loss_val = None
        if self.margin_loss is not None and "labels" in inputs:
            labels = inputs["labels"]
            logits = outputs.logits
            margin_loss_val = self.margin_loss.compute_margin_loss(logits, labels)
            if margin_loss_val is not None:
                loss_components["margin_loss"] = float(margin_loss_val.item())
                total_loss = loss + self.margin_loss.margin_weight * margin_loss_val
            else:
                total_loss = loss
        else:
            total_loss = loss

        trigger_label = inputs.get("trigger_label")
        trigger_pos = inputs.get("trigger_position")
        cls_loss_val = None
        if (
            self._classification_weight > 0.0
            and self._cls_true_id is not None
            and trigger_label is not None
            and trigger_pos is not None
        ):
            logits = outputs.logits
            batch_size = logits.size(0)
            device = logits.device
            trigger_idx = trigger_pos.to(device)
            tt = trigger_label.to(device)
            seq_len = int(logits.size(1))
            bad_pos_mask = (trigger_idx < 0) | (trigger_idx >= seq_len)
            if bad_pos_mask.any():
                bad_num = int(bad_pos_mask.sum().item())
                bad_vals = trigger_idx[bad_pos_mask].detach().cpu().tolist()[:16]
                silence, is_main, world_size, local_rank = _get_env_silence_and_rank()
                if not (silence and not is_main):
                    print(
                        f"[WARN][trigger_position] rank={local_rank}/{world_size} "
                        f"out-of-range positions: count={bad_num}, seq_len={seq_len}, sample_vals={bad_vals}. "
                        f"Will clamp to valid range.",
                        flush=True,
                    )
                trigger_idx = trigger_idx.clamp(min=0, max=max(0, seq_len - 1))
            batch_indices = torch.arange(batch_size, device=device)
            try:
                trigger_logits = logits[batch_indices, trigger_idx, :]
                two_logits = torch.stack(
                    (
                        trigger_logits[:, self._cls_false_id],
                        trigger_logits[:, self._cls_true_id],
                    ),
                    dim=-1,
                )
                cls_loss_val = F.cross_entropy(two_logits, tt)
            except Exception as e:
                self._print_failure_context("trigger_cls_loss", e, inputs, outputs)
                raise
            if cls_loss_val is not None:
                loss_components["trigger_cls_loss"] = float(cls_loss_val.item())
                total_loss = total_loss + self._classification_weight * cls_loss_val

        # NOTE: translated from Chinese (added, binding)
        bind_loss_val = None
        bind_tt_val = None
        bind_ts_val = None
        if (self._bind_trigger_task or self._bind_task_step):
            hidden = None
            # NOTE: translated from Chinese
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # hidden_states: tuple(layer0,...,last)
                try:
                    hidden = outputs.hidden_states[-1]
                except Exception:
                    hidden = None

            if hidden is not None:
                bind_losses = self._compute_hierarchical_bind_loss(hidden, inputs, outputs.logits)
                if isinstance(bind_losses, dict):
                    bind_tt_val = bind_losses.get("tt")
                    bind_ts_val = bind_losses.get("ts")
                    bind_trigger_disc_val = bind_losses.get("trigger_disc")
                else:
                    # NOTE: translated from Chinese
                    bind_tt_val = bind_losses
                    bind_ts_val = None
                    bind_trigger_disc_val = None

                if bind_tt_val is not None and self._bind_trigger_task and self._bind_tt_weight != 0:
                    loss_components["bind_tt_loss"] = float(bind_tt_val.item())
                    total_loss = total_loss + float(self._bind_tt_weight) * bind_tt_val
                if bind_ts_val is not None and self._bind_task_step and self._bind_ts_weight != 0:
                    loss_components["bind_ts_loss"] = float(bind_ts_val.item())
                    total_loss = total_loss + float(self._bind_ts_weight) * bind_ts_val
                # NOTE: translated from Chinese (added, weight, loss, config)
                if bind_trigger_disc_val is not None and self._bind_trigger_task and self._bind_trigger_disc_weight != 0:
                    loss_components["bind_trigger_disc_loss"] = float(bind_trigger_disc_val.item())
                    total_loss = total_loss + float(self._bind_trigger_disc_weight) * bind_trigger_disc_val
        
        # NOTE: translated from Chinese (added)
        constraint_loss_val = None
        if self._enable_task_step_constraint and self._task_step_mapper is not None:
            constraint_loss_val = self._compute_task_step_constraint_loss(outputs.logits, inputs)
            if constraint_loss_val is not None:
                loss_components["constraint_loss"] = float(constraint_loss_val.item())
                total_loss = total_loss + self._task_step_constraint_weight * constraint_loss_val
        
        # NOTE: translated from Chinese (added, loss)
        task_weighted_loss = None
        step_weighted_loss = None
        if self._class_weighted and (self._task_weights is not None or self._step_weights is not None):
            task_weighted_loss, step_weighted_loss = self._compute_weighted_task_step_loss(outputs.logits, inputs)
            if task_weighted_loss is not None:
                loss_components["task_weighted_loss"] = float(task_weighted_loss.item())
                total_loss = total_loss + task_weighted_loss
            if step_weighted_loss is not None:
                loss_components["step_weighted_loss"] = float(step_weighted_loss.item())
                total_loss = total_loss + step_weighted_loss
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        if not hasattr(self, "_loss_components_history"):
            self._loss_components_history = []
        self._loss_components_history.append(loss_components)
        # NOTE: translated from Chinese
        if len(self._loss_components_history) > 100:
            self._loss_components_history.pop(0)
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        if hasattr(self, "log"):
            try:
                for loss_name, loss_value in loss_components.items():
                    self.log(loss_name, loss_value)
            except Exception:
                pass
        
        # NOTE: translated from Chinese
        if not hasattr(self, "_loss_print_counter"):
            self._loss_print_counter = 0
        self._loss_print_counter += 1
        
        if self._loss_print_counter % 10 == 0:
            loss_str = " | ".join([f"{k}={v:.4f}" for k, v in loss_components.items()])
            print(f"\n[Loss Components] Total={float(total_loss.item()):.4f} ({loss_str})\n", flush=True)
        
        return (total_loss, outputs) if return_outputs else total_loss


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    silence, is_main, world_size, local_rank = _get_env_silence_and_rank()
    # NOTE: translated from Chinese
    disable_bar = (silence and not is_main)

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(
            f,
            desc=f"读取 {os.path.basename(path)}",
            unit="行",
            disable=disable_bar,
        ):
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


SYSTEM_PROMPT_WITH_SCORES = (
    "You are a vision-language model that decides whether to TRIGGER a proactive "
    "response at the CURRENT moment from a small, ordered frame set (oldest→latest). "
    "Return ONLY a strict JSON with keys: is_trigger (bool), scores {urgency,value,priority}, and reasoning (string). "
    "priority must equal max(urgency,value). Use integers 1..3."
)

SYSTEM_PROMPT_BOOL_ONLY = (
    "You are a vision-language model that decides whether to TRIGGER a proactive "
    "response at the CURRENT moment from a small, ordered frame set (oldest→latest). "
    "Return ONLY a strict JSON with keys: is_trigger (bool), and reasoning (string)."
)


# NOTE: translated from Chinese
def build_system_prompt(
    enable_evidence_frames: bool = False,
    enable_reasoning: bool = False,
    enable_confidence: bool = False,
    enable_scores: bool = False,
    use_reasoning_tokens: bool = False,
    predict_steps: int = 0,
    predict_next_action: bool = False,
):
    """根据启用的字段动态构建系统提示"""
    
    prompt = (
        "Decide whether to TRIGGER at the CURRENT moment from a small, ordered frame set.\n"
    )
    
    # NOTE: translated from Chinese
    if use_reasoning_tokens:
        prompt += (
            "If reasoning tokens are enabled, output exactly FOUR lines inside "
            "<|reasoning_start|>...<|reasoning_end|>, each starting with "
            "SCENE:, TEMPORAL:, GRAPH:, DECISION: respectively.\n"
            "Do NOT output any extra free-form text beyond these four lines.\n"
        )
    
    prompt += (
        "Generation order:\n"
        "  1) Output tag-only labels: <|trigger_start|>…<|trigger_end|> and, if IS==true, <|task_start|>…<|task_end|> then <|step_start|>…<|step_end|>.\n"
    )
    
    if predict_steps and predict_steps > 0:
        prompt += (
            f"  2) If IS==true, also predict the next {predict_steps} steps and wrap them inside "
            "<|future_steps_start|>...<|future_steps_end|>, separated by ';'.\n"
        )
    if predict_next_action:
        prompt += "  3) If IS==true, output <|next_action_start|>...<|next_action_end|>.\n"
    if enable_scores:
        score_idx = 2 + (1 if (predict_steps and predict_steps > 0) else 0) + (1 if predict_next_action else 0)
        prompt += (
            f'  {score_idx}) If IS==true, then output exactly one line: '
            'scores: {"urgency":<int>,"value":<int>,"priority":<int>} '
            "(priority must equal max(urgency,value)).\n"
        )
    
    # NOTE: translated from Chinese
    # Keep the output strictly structured; avoid free-form fields.
    
    prompt += (
        "End with <|im_end|>. Do not use tool calls. Return plain text only."
    )
    
    return prompt

# NOTE: translated from Chinese
SYSTEM_PROMPT_COND = build_system_prompt(enable_evidence_frames=True, enable_reasoning=True, 
                                        enable_confidence=True, enable_scores=True)

JSON_SKELETON = (
    "<|trigger_start|>?<|trigger_end|> and, if IS==true, then <|task_start|>?<|task_end|>"
)


def build_user_prompt(
    video_id: str,
    frame_descs: List[str],
    include_guidelines: bool,
    enable_scores: bool,
    predict_steps: int = 0,
    predict_next_action: bool = False,
    use_reasoning_tokens: bool = False,
    history_memory_text: str = "",
) -> str:
    idx_text = ", ".join(frame_descs)
    max_f = max(0, int(len(frame_descs) - 1))
    guide = (
        "Guidelines:\n"
        "- urgency: 3 = emergency/violence/anomaly; 2 = assistance/social; 1 = environment maintenance.\n"
        "- value: equal to urgency for now.\n"
        "- priority: max(urgency,value).\n"
    ) if include_guidelines else ""
    history_block = f"{history_memory_text.rstrip()}\n\n" if str(history_memory_text or "").strip() else ""
    base = (
        f"Video: {video_id}\n"
        f"Frames: {idx_text}\n\n"
        f"Frame IDs are F0..F{max_f} in the same order as listed in Frames.\n"
        f"{history_block}"
        f"{guide}"
        "Output format:\n"
        + ("0) Inside <|reasoning_start|>...<|reasoning_end|>, output exactly FOUR lines:\n"
           "SCENE: ...\nTEMPORAL: ...\nGRAPH: ...\nDECISION: ...\n"
           "(no other text).\n" if use_reasoning_tokens else "")
        + "1) First output <|trigger_start|>...<|trigger_end|> and, if IS==true, <|task_start|>...<|task_end|> then <|step_start|>...<|step_end|>.\n"
    )
    if predict_steps and predict_steps > 0:
        base += (
            f"2) If IS==true, also predict the next {predict_steps} steps and wrap them inside "
            "<|future_steps_start|>...<|future_steps_end|>, separated by ';'.\n"
        )
    if predict_next_action:
        base += "3) If IS==true, output <|next_action_start|>...<|next_action_end|>.\n"
    # scores are optional; only describe when enabled
    if enable_scores:
        score_idx = 2 + (1 if (predict_steps and predict_steps > 0) else 0) + (1 if predict_next_action else 0)
        base += (
            f"{score_idx}) If IS==true, then output exactly one line: scores: "
            '{"urgency":<int>,"value":<int>,"priority":<int>}.\n'
        )
    base += "End with <|im_end|>."
    return base


def build_history_memory_text(completed_steps: List[str], mode: str = "none", recent_k: int = 4) -> str:
    mode = str(mode or "none").strip().lower()
    if mode in {"", "none"}:
        return ""
    cleaned: List[str] = [str(x).strip() for x in (completed_steps or []) if str(x).strip()]
    recent_k = max(1, int(recent_k or 1))
    recent = cleaned[-recent_k:]
    if mode == "recent_only":
        recent_text = "; ".join(recent) if recent else "none"
        return f"Human action memory:\n- recent completed actions: {recent_text}"
    if mode == "set_only":
        completed_text = "; ".join(cleaned) if cleaned else "none"
        return f"Human action memory:\n- completed actions so far: {completed_text}"
    if mode == "oracle_past":
        completed_text = "; ".join(cleaned) if cleaned else "none"
        recent_text = "; ".join(recent) if recent else "none"
        return (
            "Human action memory:\n"
            f"- completed actions so far: {completed_text}\n"
            f"- most recent completed actions: {recent_text}"
        )
    return ""


def build_two_stage_conversation_bundle(
    *,
    video_id: str,
    frame_descs: List[str],
    images: List[Image.Image],
    completed_steps: List[str],
    is_trigger: bool,
    task_name: str,
    step_name: str,
    future_steps: List[str],
    next_action: str,
    predict_steps: int,
    predict_next_action: bool,
    history_mode: str,
    history_recent_k: int,
) -> Dict[str, Any]:
    history_memory_text = build_history_memory_text(
        completed_steps,
        mode=history_mode,
        recent_k=history_recent_k,
    )
    system_prompt_text = build_two_stage_system_prompt(
        predict_steps=predict_steps,
        predict_next_action=predict_next_action,
    )
    state_user_text = build_state_user_prompt(
        video_id,
        frame_descs,
        include_guidelines=False,
        enable_scores=False,
    )
    decision_user_text = build_decision_user_prompt(
        video_id,
        history_memory_text,
        predict_steps=predict_steps,
        predict_next_action=predict_next_action,
    )
    state_target = format_state_output_two_stage(
        is_trigger=is_trigger,
        task_name=task_name,
        step_name=step_name,
    )
    decision_target = format_decision_output_two_stage(
        is_trigger=is_trigger,
        future_steps=future_steps,
        next_action=next_action,
    )
    messages_prompt = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
        {"role": "user", "content": (
            [{"type": "image"} for _ in images] + [{"type": "text", "text": state_user_text}]
        )},
    ]
    messages_full = messages_prompt + [
        {"role": "assistant", "content": [{"type": "text", "text": state_target}]},
        {"role": "user", "content": [{"type": "text", "text": decision_user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": decision_target}]},
    ]
    return {
        "system_prompt_text": system_prompt_text,
        "state_user_text": state_user_text,
        "decision_user_text": decision_user_text,
        "history_memory_text": history_memory_text,
        "state_target": state_target,
        "decision_target": decision_target,
        "assistant_target": state_target + "\n" + decision_target,
        "messages_prompt": messages_prompt,
        "messages_full": messages_full,
    }


def load_two_stage_eval_images(
    window_files: List[str],
    max_image_long_edge: int = 0,
) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in window_files:
        with Image.open(path) as im:
            img = im.convert("RGB").copy()
        if max_image_long_edge and max(img.size) > max_image_long_edge:
            scale = float(max_image_long_edge) / float(max(img.size))
            new_size = (
                max(1, int(round(img.size[0] * scale))),
                max(1, int(round(img.size[1] * scale))),
            )
            img = img.resize(new_size, Image.LANCZOS)
        images.append(img)
    return images


def format_float_token(value: float) -> str:
    token = f"{value:g}"
    token = token.replace(".", "p").replace("-", "m").replace("+", "")
    return token


def build_run_name_from_args(args: argparse.Namespace) -> str:
    model_tag = Path(args.model_name).name
    model_tag = re.sub(r"[^A-Za-z0-9]+", "", model_tag)
    parts = [
        f"model{model_tag}",
        f"ws{args.window_size}",
        f"st{args.window_stride}",
        f"epochs{args.num_train_epochs}",
        f"lr{format_float_token(args.learning_rate)}",
        f"trainbs{args.per_device_train_batch_size}",
        f"evalbs{args.per_device_eval_batch_size}",
        f"ga{args.gradient_accumulation_steps}",
        f"lora{args.lora_rank}",
    ]
    if args.if_score:
        parts.append("withScores")
    if args.reasoning:
        parts.append("reasoning")
    if args.load_in_4bit:
        parts.append("4bit")
    if args.use_trigger_hints:
        parts.append("triggerHints")
    if getattr(args, "belief_state_eval", False):
        parts.append("beliefEval")
    parts.append("twoStage")
    history_mode = str(getattr(args, "history_memory_mode", "none") or "none").strip().lower()
    if history_mode not in {"", "none"}:
        parts.append(f"mem{re.sub(r'[^A-Za-z0-9]+', '', history_mode)}")
        if history_mode in {"oracle_past", "recent_only"}:
            parts.append(f"mr{int(getattr(args, 'history_recent_k', 4) or 4)}")
    exp_tag = str(getattr(args, "experiment_tag", "") or "").strip()
    if exp_tag:
        parts.append(re.sub(r"[^A-Za-z0-9]+", "", exp_tag))
    return "_".join(parts)


def ensure_unique_run_dir(output_dir: str, base_name: str) -> str:
    run_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(run_dir):
        return run_dir
    idx = 2
    while True:
        candidate = os.path.join(output_dir, f"{base_name}_v{idx}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def load_priority_scores(path: Optional[str]) -> Dict[str, Dict[str, int]]:
    """
    从 priority_score.json 加载任务的紧急度/价值评分。
    返回 {task_name: {"urgency": int, "value": int}} 的字典。
    """
    scores: Dict[str, Dict[str, int]] = {}
    if not path:
        return scores
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return scores

    if not isinstance(data, dict):
        return scores

    for task_name, info in data.items():
        if not task_name or not isinstance(info, dict):
            continue
        try:
            urgency = int(info.get("urgency_score", info.get("urgency", 2)))
        except (TypeError, ValueError):
            urgency = 2
        try:
            value = int(info.get("value_score", info.get("value", 2)))
        except (TypeError, ValueError):
            value = 2
        scores[str(task_name).strip()] = {"urgency": urgency, "value": value}
    return scores


def load_vocabulary_from_annotation(annotation_path: Optional[str]) -> Dict[int, str]:
    """
    从 annotation 文件中提取 vocabulary 映射，返回 {id:int -> name:str}
    """
    vocab_map: Dict[int, str] = {}
    if not annotation_path:
        return vocab_map
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"警告：加载 annotation 失败 {annotation_path}: {exc}")
        return vocab_map

    vocab_section = data.get("vocabulary")
    if not isinstance(vocab_section, dict):
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"警告：annotation 文件 {annotation_path} 中未找到 vocabulary 字段")
        return vocab_map

    for idx_key, name in vocab_section.items():
        if not name:
            continue
        try:
            idx_int = int(idx_key)
        except (TypeError, ValueError):
            continue
        vocab_map[idx_int] = str(name)
    return vocab_map


def resolve_priority_scores(
    task_name: str,
    score_map: Dict[str, Dict[str, int]],
    missing_cache: Optional[Set[str]] = None,
    default_trigger_score: int = 2,
) -> Dict[str, int]:
    """
    根据任务名称获取 {urgency,value,priority}。若找不到，则使用默认分数并打印一次警告。
    """
    sanitized = (task_name or "").strip()
    if sanitized and sanitized in score_map:
        entry = score_map[sanitized]
        urgency = int(entry.get("urgency", default_trigger_score))
        value = int(entry.get("value", default_trigger_score))
    else:
        if sanitized and missing_cache is not None and sanitized not in missing_cache:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"警告：任务 {sanitized} 未在 priority_score.json 中找到，使用默认分数 {default_trigger_score}")
            missing_cache.add(sanitized)
        urgency = value = default_trigger_score if sanitized else 1

    priority = max(int(urgency), int(value))
    return {"urgency": int(urgency), "value": int(value), "priority": priority}


def format_trigger_output(
    is_trigger: bool,
    task_name: str,
    step_name: str,
    include_scores: bool,
    priority_scores: Optional[Dict[str, Dict[str, int]]] = None,
    missing_priority_cache: Optional[Set[str]] = None,
    future_steps: Optional[List[str]] = None,
    predict_steps: int = 0,
    next_action: str = "",
    predict_next_action: bool = False,
) -> str:
    """根据预测结果构建标准化输出文本"""
    IS_L, IS_R = "<|trigger_start|>", "<|trigger_end|>"
    TK_L, TK_R = "<|task_start|>", "<|task_end|>"
    ST_L, ST_R = "<|step_start|>", "<|step_end|>"
    FS_L, FS_R = "<|future_steps_start|>", "<|future_steps_end|>"

    parts: List[str] = [f"{IS_L}{'true' if is_trigger else 'false'}{IS_R}"]

    sanitized_task = (task_name or "").strip()
    if is_trigger and sanitized_task:
        parts.append(f"{TK_L}{sanitized_task}{TK_R}")

    sanitized_step = (step_name or "").strip()
    if is_trigger and sanitized_step:
        parts.append(f"{ST_L}{sanitized_step}{ST_R}")

    if is_trigger and predict_steps and predict_steps > 0:
        future_list = [str(s).strip() for s in (future_steps or []) if str(s).strip()]
        if future_list:
            parts.append(f"{FS_L}{'; '.join(future_list)}{FS_R}")

    if is_trigger and predict_next_action:
        NA_L, NA_R = "<|next_action_start|>", "<|next_action_end|>"
        na_str = (next_action or "Wait / None").strip()
        parts.append(f"{NA_L}{na_str}{NA_R}")

    if is_trigger and include_scores:
        scores_obj = resolve_priority_scores(
            sanitized_task,
            priority_scores or {},
            missing_cache=missing_priority_cache,
        )
        parts.append("scores: " + json.dumps(scores_obj, separators=(",", ":")))

    return "\n".join(parts)


def extract_tagged_span(text: str, start_tag: str, end_tag: str) -> str:
    """从生成的文本中提取指定 tag 包裹的内容"""
    if not text:
        return ""
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return ""
    end_idx = text.find(end_tag, start_idx + len(start_tag))
    if end_idx == -1:
        return ""
    return text[start_idx + len(start_tag): end_idx].strip()


def compute_edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """
    计算两个字符串序列的编辑距离（Levenshtein），用于未来步骤预测评价。
    """
    def _norm_list(seq: List[str]) -> List[str]:
        return ["".join(str(s or "").lower().split()) for s in seq if str(s or "").strip()]

    a = _norm_list(seq_a)
    b = _norm_list(seq_b)
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


def collect_future_actions(
    frame_labels: List[int],
    current_idx: int,
    predict_steps: int,
    vocab_map: Dict[int, str],
    missing_vocab_labels: Optional[Set[int]],
    video_id: Optional[str] = None,
) -> List[str]:
    """
    从当前帧之后提取“未来动作”序列：
    - 仅在动作发生变化时计数（去除连续重复的同一动作）
    - 跳过当前动作本身，从下一不同动作开始
    - 返回最多 predict_steps 个后续不同动作名称
    - 若未来不足 predict_steps 个动作，则返回剩余的动作
    """
    if predict_steps <= 0 or not frame_labels:
        return []
    n = len(frame_labels)
    if current_idx < -1:
        current_idx = -1

    # NOTE: translated from Chinese
    try:
        cur_label = int(frame_labels[current_idx]) if 0 <= current_idx < n else None
    except Exception:
        cur_label = None

    future_steps: List[str] = []
    last_label = cur_label

    for j in range(current_idx + 1, n):
        try:
            lbl = int(frame_labels[j])
        except Exception:
            continue
        # NOTE: translated from Chinese
        if lbl <= 0:
            continue
        # NOTE: translated from Chinese
        if last_label is not None and lbl == last_label:
            continue
        name = vocab_map.get(lbl, "")
        if not name:
            if missing_vocab_labels is not None and lbl not in missing_vocab_labels:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    vid_info = f"，视频 {video_id}" if video_id else ""
                    print(f"警告：未来 step 标签 ID {lbl} 未在 vocabulary 中找到{vid_info}")
                missing_vocab_labels.add(lbl)
            name = str(lbl)
        future_steps.append(str(name).strip())
        last_label = lbl
        if len(future_steps) >= predict_steps:
            break
    return future_steps


def collect_completed_steps_before_current_segment(
    frame_labels: List[int],
    current_idx: int,
    vocab_map: Dict[int, str],
) -> List[str]:
    """
    获取当前 step 段开始前已完成的 step 序列（不包含当前正在执行的 step）。
    从 current_idx 往前找到当前 step 段起点，在 [0, seg_start) 上抽取去重后的 step 序列。
    """
    if not frame_labels:
        return []
    n = len(frame_labels)
    if current_idx < 0 or current_idx >= n:
        return []
    cur = int(frame_labels[current_idx])
    seg_start = current_idx
    for i in range(current_idx - 1, -1, -1):
        try:
            lbl = int(frame_labels[i])
        except (TypeError, ValueError):
            continue
        if lbl <= 0:
            continue
        if lbl != cur:
            seg_start = i + 1
            break
        seg_start = i
    completed: List[str] = []
    last_label = -1
    for j in range(0, seg_start):
        try:
            lbl = int(frame_labels[j])
        except (TypeError, ValueError):
            continue
        if lbl <= 0:
            continue
        if lbl == last_label:
            continue
        name = vocab_map.get(lbl, str(lbl))
        if name:
            completed.append(str(name).strip())
        last_label = lbl
    return completed


def select_evidence_positions(frame_labels: List[int], start: int, end: int) -> List[int]:
    """
    FramePick-CoT evidence selector.
    Return window-relative positions k (so the prompt can reference F{k}).
    Rules:
      1) Always include last frame (end).
      2) Include the latest step boundary within the window (i where label changes).
      3) Include the start of the current step segment (within window).
    Deduplicate, sort, keep up to 3.
    """
    if not frame_labels:
        return []
    n = len(frame_labels)
    start = max(0, int(start))
    end = min(int(end), n - 1)
    if end < start:
        return []
    k_last = end - start

    # latest boundary frame within [start+1, end]
    i_b = -1
    for i in range(end, start, -1):
        try:
            a = int(frame_labels[i])
            b = int(frame_labels[i - 1])
        except Exception:
            continue
        if a != b:
            i_b = i
            break
    k_boundary = (i_b - start) if i_b != -1 else -1

    # current segment start within [start, end]
    i_seg = end
    try:
        cur = int(frame_labels[end])
    except Exception:
        cur = None
    if cur is not None:
        for i in range(end - 1, start - 1, -1):
            try:
                lbl = int(frame_labels[i])
            except Exception:
                continue
            if lbl != cur:
                i_seg = i + 1
                break
        i_seg = max(start, min(i_seg, end))
    k_seg = i_seg - start

    ev = [k_last, k_boundary, k_seg]
    ev = [int(x) for x in ev if isinstance(x, int) and x >= 0]
    ev = sorted(set(ev))[:3]
    return ev


def rle_steps(steps: List[str]) -> List[Tuple[str, int]]:
    runs: List[Tuple[str, int]] = []
    if not steps:
        return runs
    cur = str(steps[0])
    cnt = 1
    for s in steps[1:]:
        s0 = str(s)
        if s0 == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = s0, 1
    runs.append((cur, cnt))
    return runs


def pick_brief(items: List[str], k: int = 2) -> List[str]:
    out: List[str] = []
    for x in items or []:
        s = str(x or "").strip()
        if not s:
            continue
        if s.lower() == "none":
            continue
        out.append(s)
    if not out:
        return []
    return out[: max(1, int(k))]


def build_window_steps(frame_labels: List[int], start: int, end: int, vocab_map: Dict[int, str]) -> List[str]:
    steps: List[str] = []
    if not frame_labels:
        return steps
    n = len(frame_labels)
    start = max(0, int(start))
    end = min(int(end), n - 1)
    if end < start:
        return steps
    for i in range(start, end + 1):
        try:
            lbl = int(frame_labels[i])
        except Exception:
            steps.append("None")
            continue
        if lbl <= 0:
            steps.append("None")
            continue
        name = str(vocab_map.get(lbl, str(lbl))).strip()
        steps.append(name if name else "None")
    return steps


def derive_enabled_and_parallel_steps(
    task_name: str,
    completed_steps: List[str],
    current_step: str,
    human_future_steps: List[str],
    annotation_path: Optional[str],
    tg_cache: Dict[str, "TaskGraphManager"],
) -> Tuple[List[str], List[str]]:
    """
    Reuse existing graph feasibility logic:
    - enabled_steps: legal robot actions under current completed steps and immediate human step.
    - parallel_steps: subset from other thread if possible; fallback to first enabled actions.
    """
    if not annotation_path or not os.path.exists(annotation_path):
        return [], []
    task_name = str(task_name or "").strip()
    if not task_name:
        return [], []
    try:
        if task_name not in tg_cache:
            tg_cache[task_name] = TaskGraphManager(annotation_path, task_name)
        tg = tg_cache[task_name]
    except Exception:
        return [], []

    human_immediate = [str(s).strip() for s in (human_future_steps[:1] or []) if str(s).strip()]
    try:
        legal_raw = tg.get_legal_robot_actions(list(completed_steps or []), human_immediate)
    except Exception:
        legal_raw = []
    cur = str(current_step or "").strip()
    enabled: List[str] = []
    seen: Set[str] = set()
    for a in legal_raw or []:
        s = str(a or "").strip()
        if not s or s in seen:
            continue
        if s == cur:
            continue
        if s.lower() in {"wait / none", "terminate"}:
            continue
        enabled.append(s)
        seen.add(s)
    enabled = enabled[:3]

    parallel: List[str] = []
    try:
        thread_map = getattr(getattr(tg, "graph", None), "thread_map", {}) or {}
    except Exception:
        thread_map = {}
    cur_tid = thread_map.get(cur) if cur else None
    for s in enabled:
        tid = thread_map.get(s)
        if cur_tid and tid and tid != cur_tid:
            parallel.append(s)
    if not parallel:
        parallel = enabled[:2]
    else:
        parallel = parallel[:2]
    return enabled, parallel


def build_reasoning_4sent(
    idx: int,
    env_desc: str,
    window_steps: List[str],
    cur_step: str,
    prev_step: Optional[str],
    done_steps: List[str],
    enabled_steps: List[str],
    parallel_steps: List[str],
    trig: bool,
    trig_reason: Optional[str],
    human_future: List[str],
    robot_next: str,
    evid_frames: Optional[List[str]] = None,
) -> str:
    rnd = random.Random(int(idx))

    env_text = str(env_desc or "N/A").strip()
    # Avoid duplicated punctuation like ".."
    env_text = env_text.rstrip(" .")
    if not env_text:
        env_text = "N/A"

    scene_tpl = [
        "SCENE: {env}.",
    ]
    s1 = rnd.choice(scene_tpl).format(env=env_text)

    runs_all = rle_steps(window_steps or [])
    runs = [(a, c) for a, c in runs_all if str(a).strip() and str(a).strip().lower() != "none"]
    # Keep only labeled step names, no "xN" suffix.
    run_names: List[str] = []
    for a, _c in runs:
        s = str(a).strip()
        if not s:
            continue
        if run_names and run_names[-1] == s:
            continue
        run_names.append(s)
    runs_str = " -> ".join(run_names[-3:]) if run_names else "N/A"
    temporal_tpl = [
        "TEMPORAL: Within the window, the step trajectory is {runs}.",
    ]
    cur_step_txt = str(cur_step or "").strip()
    if runs_str == "N/A":
        s2 = "TEMPORAL: N/A"
    elif prev_step:
        s2 = rnd.choice(temporal_tpl).format(
            runs=runs_str,
            cur=(cur_step_txt if cur_step_txt and cur_step_txt.lower() != "none" else "N/A"),
            prev=str(prev_step or "N/A"),
        )
    else:
        s2 = rnd.choice(temporal_tpl[:2]).format(
            runs=runs_str,
            cur=(cur_step_txt if cur_step_txt and cur_step_txt.lower() != "none" else "N/A"),
            prev="N/A",
        )

    done_b = pick_brief(done_steps or [], 2)
    ena_b = pick_brief(enabled_steps or [], 3)
    par_b = pick_brief(parallel_steps or [], 2)
    done_str = ", ".join(done_b) if done_b else "no key prerequisite step is confirmed as done"
    ena_str = ", ".join(ena_b) if ena_b else "no next step is enabled under the constraints"
    par_str = ", ".join(par_b) if par_b else "no safe parallel opportunity is identified"
    graph_tpl = [
        "GRAPH: Given {done}, the task-graph constraints enable {ena}, and a parallel opportunity is {par}.",
        "GRAPH: With completed steps being {done}, feasible next steps include {ena}, and parallel candidates include {par}.",
    ]
    if not trig:
        s3 = "GRAPH: N/A"
    else:
        s3 = rnd.choice(graph_tpl).format(done=done_str, ena=ena_str, par=par_str)

    hf_b = pick_brief(human_future or [], 3)
    hf_str = ", ".join(hf_b) if hf_b else "N/A"
    reason = str(trig_reason or "the current state suggests intervention may be beneficial").strip()
    decision_tpl = [
        "DECISION: Trigger={trig}; the human may do {hf}, and the robot chooses {act}.",
        "DECISION: We set Trigger={trig}; predicted human next steps are {hf}, so the robot action is {act}.",
        "DECISION: Trigger={trig}; forecasted human actions: {hf}; robot next action: {act}.",
    ]
    if not trig:
        s4 = "DECISION: Trigger=false"
    else:
        s4 = rnd.choice(decision_tpl).format(
            trig=str(bool(trig)).lower(),
            reason=reason,
            hf=hf_str,
            act=str(robot_next or "Wait / None"),
        )
    return "\n".join([s1, s2, s3, s4])


def teacher_next_action(
    task_name: str,
    completed_steps: List[str],
    current_step: str,
    future_steps_list: List[str],
    annotation_path: Optional[str],
    tg_cache: Dict[str, "TaskGraphManager"],
) -> str:
    """
    对齐 one-step eval 的 entropy selector 口径生成 next_action：
    - legal（含 no-preemption）过滤在 get_legal_robot_actions 阶段完成
    - candidates = future_steps_list ∩ legal_now ∩ real_action_node（保持 future 顺序）
    - 选择 argmin entropy（human_history = completed_steps, robot_history = []），并用位置做 tie-break
    """
    # Align with eval_onestep_end2end.py default.
    immediate_M = 1

    def _is_real_action_node(step: str, graph_env: Optional["TaskGraphManager"] = None) -> bool:
        s = str(step or "").strip()
        if not s or s == "Wait / None":
            return False
        if s.lower() == "terminate":
            return False
        if graph_env is not None:
            node = (getattr(graph_env, "name2node", None) or {}).get(s)
            if not node:
                return False
            try:
                if bool(node.get("is_midlevel", False)):
                    return False
            except Exception:
                pass
        return True

    def _shannon_entropy_from_counts(counts: Dict[str, int]) -> float:
        tot = sum(int(v) for v in counts.values() if v and v > 0)
        if tot <= 0:
            return 0.0
        h = 0.0
        for v in counts.values():
            if not v or v <= 0:
                continue
            p = v / tot
            h -= p * math.log2(p)
        return float(h)

    def _hmix_entropy(graph_env: "TaskGraphManager", human_history: List[str], candidate_action: str) -> float:
        # Simple Hmix implementation consistent with eval suggestion:
        # entropy over thread_id distribution of (human_history + robot_history + [candidate_action]).
        counts: Dict[str, int] = {}
        seq = list(human_history or []) + [candidate_action]
        for a in seq:
            tid = (getattr(graph_env, "thread_map", None) or {}).get(a)
            if not tid:
                continue
            counts[tid] = int(counts.get(tid, 0)) + 1
        return _shannon_entropy_from_counts(counts)

    def _entropy(graph_env: "TaskGraphManager", completed: List[str], a: str) -> float:
        """
        Prefer EntropyPlanner interface (eval signature). If it mismatches, fall back to local Hmix entropy.
        """
        try:
            return float(EntropyPlanner(graph_env).calculate_entropy(robot_history=[], candidate_action=a, human_history=list(completed)))
        except Exception:
            return float(_hmix_entropy(graph_env, list(completed), a))

    if not annotation_path or not os.path.exists(annotation_path):
        if len(future_steps_list) > 1:
            return future_steps_list[1]
        return "Wait / None"
    task_name = (task_name or "").strip()
    if not task_name:
        if len(future_steps_list) > 1:
            return future_steps_list[1]
        return "Wait / None"
    try:
        if task_name not in tg_cache:
            tg_cache[task_name] = TaskGraphManager(annotation_path, task_name)
        tg = tg_cache[task_name]
    except Exception:
        if len(future_steps_list) > 1:
            return future_steps_list[1]
        return "Wait / None"

    # Legal set at this decision point (no-preemption enforced here).
    # Candidates should include the whole future list; legal will exclude immediate head via human_immediate.
    human_immediate = [str(s).strip() for s in (future_steps_list[: max(0, int(immediate_M))] or []) if str(s).strip()]
    try:
        legal_raw = tg.get_legal_robot_actions(list(completed_steps), list(human_immediate))
        legal_now = set([a for a in (legal_raw or []) if _is_real_action_node(str(a), tg)])
    except Exception:
        legal_now = set()
    # Also avoid suggesting the current in-progress human step.
    legal_now.discard(str(current_step or "").strip())

    # Helper: decide pool then select min-entropy with stable tie-break.
    def _select_min_entropy(pool: List[str], cand_order: Optional[List[str]] = None) -> str:
        if not pool:
            return "Wait / None"
        # Filter out too-early Terminate if any other option exists.
        if any(str(s).strip().lower() != "terminate" for s in pool):
            pool = [s for s in pool if str(s).strip().lower() != "terminate"]
        if not pool:
            return "Wait / None"
        # stable tie-break: (entropy, pos_in_candidates, lex)
        order = list(cand_order or pool)
        best_a = pool[0]
        best_key = (float("inf"), 10**9, str(best_a))
        for a in pool:
            e = _entropy(tg, list(completed_steps), a)
            try:
                pos = order.index(a)
            except Exception:
                pos = 10**9
            key = (float(e), int(pos), str(a))
            if key < best_key:
                best_key = key
                best_a = a
        return best_a if best_a else "Wait / None"

    # 1) candidates = future_steps_list ∩ legal_now (preserve future order).
    cand0: List[str] = []
    seen: Set[str] = set()
    for x in future_steps_list or []:
        s = str(x or "").strip()
        if not s or s == "Wait / None":
            continue
        if s in seen:
            continue
        seen.add(s)
        if s in legal_now and _is_real_action_node(s, tg):
            cand0.append(s)

    # Environment-feasible candidates: must be node + prereq ok at decision time (eval does this too).
    def _feasible(xs: List[str]) -> List[str]:
        out: List[str] = []
        for a in xs:
            node = (getattr(tg, "name2node", None) or {}).get(a)
            if node is None:
                continue
            cond = node.get("activation_condition", "TRUE")
            try:
                if tg.check_condition(cond, list(completed_steps)):
                    out.append(a)
            except Exception:
                continue
        return out

    feasible_cand = _feasible(cand0)
    if feasible_cand:
        return _select_min_entropy(feasible_cand, cand_order=cand0)

    # 2) fallback: choose min-entropy from legal_now itself.
    feasible_legal = _feasible(list(legal_now))
    if feasible_legal:
        # tie-break within legal fallback: lex after entropy (no meaningful pos)
        return _select_min_entropy(sorted(set(feasible_legal)), cand_order=sorted(set(feasible_legal)))
    return "Wait / None"


def predict_trigger_with_logits(
    model: "torch.nn.Module",
    processor_or_tokenizer: Any,
    example: Dict[str, Any],
    include_scores: bool,
    fallback_task: str = "",
    fallback_step: str = "",
    priority_scores: Optional[Dict[str, Dict[str, int]]] = None,
    missing_priority_cache: Optional[Set[str]] = None,
) -> Tuple[str, int, str, str, List[str], str]:
    """
    使用教师强制方式，根据 logits 判定 <|trigger|> 标签，并返回格式化输出。
    """
    import torch

    processor = processor_or_tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else None
    tokenizer = processor.tokenizer if processor is not None else processor_or_tokenizer

    device = next(model.parameters()).device

    input_ids = example["prompt_input_ids"].unsqueeze(0).to(device)
    attention_mask = example["prompt_attention_mask"].unsqueeze(0).to(device)
    inputs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}

    prompt_pixel = example.get("prompt_pixel_values")
    if isinstance(prompt_pixel, torch.Tensor):
        inputs["pixel_values"] = prompt_pixel.to(device=device, dtype=torch.bfloat16)
    prompt_grid = example.get("prompt_image_grid_thw")
    if isinstance(prompt_grid, torch.Tensor):
        inputs["image_grid_thw"] = prompt_grid.to(device)

    trigger_start_id = tokenizer.convert_tokens_to_ids("<|trigger_start|>")
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    true_id = true_ids[0] if true_ids else tokenizer.convert_tokens_to_ids("true")
    false_id = false_ids[0] if false_ids else tokenizer.convert_tokens_to_ids("false")

    with torch.no_grad():
        # NOTE: translated from Chinese
        _ = model(**inputs)

        trigger_token = torch.tensor([[trigger_start_id]], device=device)
        next_input_ids = torch.cat([input_ids, trigger_token], dim=1)
        next_attention = torch.cat([attention_mask, torch.ones_like(trigger_token)], dim=1)
        next_inputs = {"input_ids": next_input_ids, "attention_mask": next_attention}
        if "pixel_values" in inputs:
            next_inputs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            next_inputs["image_grid_thw"] = inputs["image_grid_thw"]

        outputs = model(**next_inputs)
        logits = outputs.logits[0, -1]

    logit_true = float(logits[true_id])
    logit_false = float(logits[false_id])
    pred_is = 1 if logit_true > logit_false else 0

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,
            "pad_token_id": pad_token_id,
        }
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with autocast_ctx:
            gen_outputs = model.generate(**inputs, **gen_kwargs)
    gen_tokens = gen_outputs[0, input_ids.size(1):]
    normalized_text = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

    parsed_task = extract_tagged_span(normalized_text, "<|task_start|>", "<|task_end|>")
    parsed_step = extract_tagged_span(normalized_text, "<|step_start|>", "<|step_end|>")
    parsed_future = extract_tagged_span(normalized_text, "<|future_steps_start|>", "<|future_steps_end|>")
    parsed_next_action = extract_tagged_span(normalized_text, "<|next_action_start|>", "<|next_action_end|>")
    task_name = parsed_task if pred_is == 1 else ""
    step_name_output = parsed_step if pred_is == 1 else ""
    future_steps_output: List[str] = []
    if pred_is == 1 and parsed_future:
        for seg in re.split(r"[;\n]+", parsed_future):
            seg = seg.strip()
            if seg:
                future_steps_output.append(seg)

    if pred_is == 1 and not task_name:
        task_name = (fallback_task or "").strip()
    if pred_is == 1 and not step_name_output:
        step_name_output = (fallback_step or "").strip()

    if example.get("ts_two_stage"):
        state_formatted = format_state_output_two_stage(
            bool(pred_is),
            task_name,
            step_name_output,
        )
        if not pred_is:
            decision_formatted = format_decision_output_two_stage(False, [], "")
            return (
                state_formatted + "\n" + decision_formatted,
                pred_is,
                task_name,
                step_name_output,
                [],
                "",
            )
        if processor is None:
            raise RuntimeError("two-stage prediction requires processor, not bare tokenizer")

        window_files = [str(x) for x in (example.get("ts_window_files") or []) if str(x).strip()]
        max_edge = int(example.get("ts_max_image_long_edge", 0) or 0)
        images = load_two_stage_eval_images(window_files, max_image_long_edge=max_edge)
        messages_decision_prompt = [
            {"role": "system", "content": [{"type": "text", "text": str(example.get('ts_system_prompt_text', '') or '')}]},
            {"role": "user", "content": (
                [{"type": "image"} for _ in images] + [{"type": "text", "text": str(example.get('ts_state_user_text', '') or '')}]
            )},
            {"role": "assistant", "content": [{"type": "text", "text": state_formatted}]},
            {"role": "user", "content": [{"type": "text", "text": str(example.get('ts_decision_user_text', '') or '')}]},
        ]
        decision_text_prompt = tokenizer.apply_chat_template(
            messages_decision_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        decision_inputs = processor(text=[decision_text_prompt], images=[images], return_tensors="pt")
        if "pixel_values" in decision_inputs:
            decision_inputs["pixel_values"] = decision_inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
        if "image_grid_thw" in decision_inputs:
            decision_inputs["image_grid_thw"] = decision_inputs["image_grid_thw"].to(device)
        decision_inputs["input_ids"] = decision_inputs["input_ids"].to(device)
        decision_inputs["attention_mask"] = decision_inputs["attention_mask"].to(device)
        decision_prompt_len = int(decision_inputs["input_ids"].shape[1])

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 192,
                "do_sample": False,
                "pad_token_id": pad_token_id,
            }
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with autocast_ctx:
                decision_outputs = model.generate(**decision_inputs, **gen_kwargs)

        decision_tokens = decision_outputs[0, decision_prompt_len:]
        decision_text = tokenizer.decode(decision_tokens, skip_special_tokens=False).strip()
        decision_future = extract_tagged_span(decision_text, "<|future_steps_start|>", "<|future_steps_end|>")
        decision_next_action = extract_tagged_span(decision_text, "<|next_action_start|>", "<|next_action_end|>")
        future_steps_output = []
        if decision_future:
            for seg in re.split(r"[;\n]+", decision_future):
                seg = seg.strip()
                if seg:
                    future_steps_output.append(seg)
        decision_formatted = decision_text if decision_text else format_decision_output_two_stage(
            True,
            future_steps_output,
            decision_next_action,
        )
        return (
            state_formatted + "\n" + decision_formatted,
            pred_is,
            task_name,
            step_name_output,
            future_steps_output,
            decision_next_action,
        )

    formatted = normalized_text if normalized_text else format_trigger_output(
        bool(pred_is),
        task_name,
        step_name_output,
        include_scores,
        priority_scores=priority_scores,
        missing_priority_cache=missing_priority_cache,
    )
    return formatted, pred_is, task_name, step_name_output, future_steps_output, parsed_next_action


def predict_trigger_with_logits_batch(
    model: "torch.nn.Module",
    processor_or_tokenizer: Any,
    examples: List[Dict[str, Any]],
    include_scores: bool,
    fallback_task: str = "",
    fallback_step: str = "",
    priority_scores: Optional[Dict[str, Dict[str, int]]] = None,
    missing_priority_cache: Optional[Set[str]] = None,
) -> List[Tuple[str, int, str, str, List[str]]]:
    """
    批量版本的 trigger 预测与生成，一次性对多个样本进行 forward + generate。
    返回列表，每个元素为 (formatted_text, pred_is, task_name, step_name_output)。
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if not examples:
        return []

    if any(ex.get("ts_two_stage") for ex in examples):
        results: List[Tuple[str, int, str, str, List[str]]] = []
        for ex in examples:
            formatted, pred_is, task_name, step_name, future_steps, _ = predict_trigger_with_logits(
                model,
                processor_or_tokenizer,
                ex,
                include_scores=include_scores,
                fallback_task=fallback_task,
                fallback_step=fallback_step,
                priority_scores=priority_scores,
                missing_priority_cache=missing_priority_cache,
            )
            results.append((formatted, pred_is, task_name, step_name, future_steps))
        return results

    tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer
    device = next(model.parameters()).device

    # NOTE: translated from Chinese
    input_ids_list = []
    attn_mask_list = []
    pixel_values_list = []
    image_grid_list = []
    input_lens: List[int] = []

    for ex in examples:
        ids = ex["prompt_input_ids"]
        attn = ex["prompt_attention_mask"]
        if not isinstance(ids, torch.Tensor) or not isinstance(attn, torch.Tensor):
            raise ValueError("expect prompt_input_ids / prompt_attention_mask to be torch.Tensor in batch examples")
        ids = ids.to(device)
        attn = attn.to(device)
        input_ids_list.append(ids)
        attn_mask_list.append(attn)
        input_lens.append(int(ids.size(0)))

        pv = ex.get("prompt_pixel_values")
        if isinstance(pv, torch.Tensor):
            pixel_values_list.append(pv.to(device=device, dtype=torch.bfloat16))
        else:
            pixel_values_list.append(None)

        grid = ex.get("prompt_image_grid_thw")
        if isinstance(grid, torch.Tensor):
            image_grid_list.append(grid.to(device))
        else:
            image_grid_list.append(None)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attn_mask_list, batch_first=True, padding_value=0)

    inputs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # NOTE: translated from Chinese
    def _all_same_shape(tensors: List[Optional[torch.Tensor]]) -> bool:
        shapes = [tuple(t.shape) for t in tensors if t is not None]
        return len(shapes) > 0 and len(set(shapes)) == 1

    has_pixels = any(pv is not None for pv in pixel_values_list)
    has_grids = any(g is not None for g in image_grid_list)

    # NOTE: translated from Chinese
    if (has_pixels and not _all_same_shape(pixel_values_list)) or (has_grids and not _all_same_shape(image_grid_list)):
        results: List[Tuple[str, int, str, str, List[str]]] = []
        for ex in examples:
            formatted, pred_is, task_name, step_name, future_steps, parsed_next_action = predict_trigger_with_logits(
                model,
                processor_or_tokenizer,
                ex,
                include_scores=include_scores,
                fallback_task=fallback_task,
                fallback_step=fallback_step,
                priority_scores=priority_scores,
                missing_priority_cache=missing_priority_cache,
            )
            results.append((formatted, pred_is, task_name, step_name, future_steps))
        return results

    # NOTE: translated from Chinese
    if has_pixels:
        inputs["pixel_values"] = torch.stack([pv for pv in pixel_values_list if pv is not None], dim=0)
    if has_grids:
        inputs["image_grid_thw"] = torch.stack([g for g in image_grid_list if g is not None], dim=0)

    trigger_start_id = tokenizer.convert_tokens_to_ids("<|trigger_start|>")
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    true_id = true_ids[0] if true_ids else tokenizer.convert_tokens_to_ids("true")
    false_id = false_ids[0] if false_ids else tokenizer.convert_tokens_to_ids("false")

    # NOTE: translated from Chinese
    with torch.no_grad():
        _ = model(**inputs)

        bs = input_ids.size(0)
        trigger_token = torch.full((bs, 1), trigger_start_id, device=device, dtype=torch.long)
        next_input_ids = torch.cat([inputs["input_ids"], trigger_token], dim=1)
        next_attention = torch.cat(
            [inputs["attention_mask"], torch.ones_like(trigger_token, dtype=inputs["attention_mask"].dtype)],
            dim=1,
        )
        next_inputs = {
            "input_ids": next_input_ids,
            "attention_mask": next_attention,
        }
        if "pixel_values" in inputs:
            next_inputs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            next_inputs["image_grid_thw"] = inputs["image_grid_thw"]

        outputs = model(**next_inputs)
        logits = outputs.logits[:, -1, :]  # [B, vocab]

    logit_true = logits[:, true_id]
    logit_false = logits[:, false_id]
    pred_is_batch = (logit_true > logit_false).long().tolist()

    # NOTE: translated from Chinese
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,
            "pad_token_id": pad_token_id,
        }
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with autocast_ctx:
            gen_outputs = model.generate(**inputs, **gen_kwargs)

    results: List[Tuple[str, int, str, str, List[str]]] = []
    for b_idx, (ex, pred_is, in_len) in enumerate(zip(examples, pred_is_batch, input_lens)):
        # NOTE: translated from Chinese
        gen_tokens = gen_outputs[b_idx, in_len:]
        normalized_text = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

        parsed_task = extract_tagged_span(normalized_text, "<|task_start|>", "<|task_end|>")
        parsed_step = extract_tagged_span(normalized_text, "<|step_start|>", "<|step_end|>")
        parsed_future = extract_tagged_span(normalized_text, "<|future_steps_start|>", "<|future_steps_end|>")
        task_name = parsed_task if pred_is == 1 else ""
        step_name_output = parsed_step if pred_is == 1 else ""
        future_steps_output: List[str] = []
        if pred_is == 1 and parsed_future:
            for seg in re.split(r"[;\n]+", parsed_future):
                seg = seg.strip()
                if seg:
                    future_steps_output.append(seg)

        if pred_is == 1 and not task_name:
            task_name = (fallback_task or "").strip()
        if pred_is == 1 and not step_name_output:
            step_name_output = (fallback_step or "").strip()

        formatted = normalized_text if normalized_text else format_trigger_output(
            bool(pred_is),
            task_name,
            step_name_output,
            include_scores,
            priority_scores=priority_scores,
            missing_priority_cache=missing_priority_cache,
        )
        results.append((formatted, pred_is, task_name, step_name_output, future_steps_output))

    return results


def build_trigger_map(
    trigger_json_path: Optional[str],
    candidate_ids: Set[str],
) -> Dict[str, str]:
    """
    仅为出现的视频构建 id→trigger_en 的小字典，避免整文件载入内存。
    假设结构与示例一致：最外层有 "annotations": { "<id>": { ..., "trigger_en": "..." } }。
    采用逐行状态机，粗略解析，鲁棒性足以应对该文件结构。
    """
    if not trigger_json_path or not candidate_ids:
        return {}
    result: Dict[str, str] = {}
    current_id: Optional[str] = None
    want_block = False
    try:
        with open(trigger_json_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                # NOTE: translated from Chinese
                m = re.match(r'"([A-Za-z0-9_]+)":\s*\{', line)
                if m:
                    vid = m.group(1)
                    if vid in candidate_ids:
                        current_id = vid
                        want_block = True
                    else:
                        current_id = None
                        want_block = False
                    continue
                if want_block and current_id:
                    # NOTE: translated from Chinese
                    te = re.search(r'"trigger_en"\s*:\s*"(.*)"\s*,?', line)
                    if te:
                        # NOTE: translated from Chinese
                        text = te.group(1)
                        result[current_id] = text
                        # NOTE: translated from Chinese
                        want_block = False
                        current_id = None
                # NOTE: translated from Chinese
                if len(result) == len(candidate_ids):
                    break
    except Exception:
        # NOTE: translated from Chinese
        pass
    return result


def extract_base_id(video_id: str) -> str:
    # e.g. TSU_P02T01C06_01 -> TSU_P02T01C06
    return video_id.rsplit("_", 1)[0]


def load_keyframe_dirs_from_annotation(annotation_path: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    从 annotation 文件中的 keyframe_paths 字段构建:
      - video_id/base_id -> keyframes_dir 的映射
    例如:
      "SITE_A_T09S01A02cam01": "/.../keyframes/T09S01A02/T09S01A02cam01"
    """
    video_to_dir: Dict[str, str] = {}
    base_to_dir: Dict[str, str] = {}
    if not annotation_path or not os.path.exists(annotation_path):
        return video_to_dir, base_to_dir
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return video_to_dir, base_to_dir

    kp_map = data.get("keyframe_paths", {}) or {}
    if not isinstance(kp_map, dict):
        return video_to_dir, base_to_dir

    for vid, kfdir in kp_map.items():
        if not isinstance(vid, str):
            continue
        if not isinstance(kfdir, str):
            continue
        kfdir = kfdir.strip()
        if not kfdir or not os.path.isdir(kfdir):
            continue
        video_to_dir.setdefault(vid, kfdir)
        base_id = extract_base_id(vid)
        if base_id:
            base_to_dir.setdefault(base_id, kfdir)
    return video_to_dir, base_to_dir


def load_scenarios_from_annotation(annotation_path: Optional[str]) -> Dict[str, str]:
    """
    Load scenario text from annotation["annotations"] and build a flexible id->scenario map.
    Supports keys such as annotation id, take_uid, video_id and their base ids.
    """
    out: Dict[str, str] = {}
    if not annotation_path or not os.path.exists(annotation_path):
        return out
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return out
    annotations = data.get("annotations", {}) or {}
    if not isinstance(annotations, dict):
        return out

    def _put(key: str, value: str) -> None:
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            return
        out.setdefault(k, v)
        out.setdefault(extract_base_id(k), v)

    for ann_key, ann_val in annotations.items():
        if not isinstance(ann_val, dict):
            continue
        scenario = (
            ann_val.get("scenario")
            or ann_val.get("scenario_en")
            or ann_val.get("environment")
            or ann_val.get("env_desc")
            or ""
        )
        scenario = str(scenario or "").strip()
        if not scenario:
            continue
        _put(str(ann_key), scenario)
        _put(str(ann_val.get("take_uid", "") or ""), scenario)
        _put(str(ann_val.get("video_id", "") or ""), scenario)
    return out


def resolve_scenario_text(
    row: Dict[str, Any],
    video_id: str,
    scenario_map: Dict[str, str],
) -> str:
    """
    Resolve per-sample scenario text with priority:
    row fields -> video id map -> base id map -> "N/A".
    """
    for key in ("scenario", "scenario_en", "environment", "env_desc"):
        val = str(row.get(key, "") or "").strip()
        if val:
            return val
    vid = str(video_id or "").strip()
    base = extract_base_id(vid)
    return str(scenario_map.get(vid) or scenario_map.get(base) or "N/A")


def _find_subseq(haystack: List[int], needle: List[int], start: int = 0) -> int:
    """在haystack中查找needle子序列的起始位置"""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    for i in range(start, n - m + 1):
        if haystack[i:i+m] == needle:
            return i
    return -1


class MaxTokensAfterPattern(LogitsProcessor):
    """一旦检测到给定 pattern（如 'scores:'）出现在生成序列里，就只允许往后生成至多 max_tokens；
       达到上限后强烈偏置 '}' 与 <|im_end|>。batch=1 版本。"""
    def __init__(self, tokenizer, start_len: int, pattern: str, max_tokens: int, eos_id: int, bias: float = 5.0):
        self.tok = tokenizer
        self.start_len = start_len
        self.pat = tokenizer.encode(pattern, add_special_tokens=False)
        self.max_tokens = max(1, int(max_tokens))
        self.eos_id = eos_id
        self.bias = float(bias)
        self.trigger_pos = -1
        self.gen_after = 0

    def __call__(self, input_ids, scores):
        if input_ids.size(0) != 1:
            return scores
        cur = input_ids[0].tolist()
        # NOTE: translated from Chinese
        if self.trigger_pos == -1:
            pos = _find_subseq(cur, self.pat, start=self.start_len)
            if pos != -1:
                self.trigger_pos = pos + len(self.pat)
                self.gen_after = max(0, len(cur) - self.trigger_pos)
        else:
            # NOTE: translated from Chinese
            self.gen_after = max(self.gen_after, len(cur) - self.trigger_pos)
            if self.gen_after >= self.max_tokens:
                # NOTE: translated from Chinese
                try:
                    rbrace = self.tok.encode("}", add_special_tokens=False)[0]
                    scores[0, rbrace] = scores[0, rbrace] + self.bias
                except Exception:
                    pass
                scores[0, self.eos_id] = scores[0, self.eos_id] + self.bias
        return scores


class MinTokensBeforeTags(LogitsProcessor):
    """
    在生成满 min_new_tokens 之前，阻止开始输出指定 tag（支持部分前缀匹配，防止逐字拼出 [[IS]]）。
    仅支持 batch_size=1 的解码（你当前评估就是逐样本生成）。
    """
    def __init__(self, tokenizer, start_len: int, min_new_tokens: int, tags: List[str]):
        self.tok = tokenizer
        self.start_len = int(start_len)
        self.min_new_tokens = int(min_new_tokens)
        # NOTE: translated from Chinese
        variants = []
        for t in tags:
            variants += [t, " " + t, "\n" + t]
        self.patterns = [self.tok.encode(v, add_special_tokens=False) for v in variants]

    @staticmethod
    def _match_len(tail: List[int], pat: List[int]) -> int:
        k = min(len(tail), len(pat))
        while k > 0 and tail[-k:] != pat[:k]:
            k -= 1
        return k

    def __call__(self, input_ids, scores):
        # NOTE: translated from Chinese
        if input_ids.size(0) != 1 or self.min_new_tokens <= 0:
            return scores
        gen_len = input_ids.size(1) - self.start_len
        if gen_len < self.min_new_tokens:
            tail = input_ids[0, self.start_len:].tolist()
            for pat in self.patterns:
                if not pat:
                    continue
                k = self._match_len(tail, pat)  # NOTE: translated from Chinese
                if k < len(pat):
                    next_id = pat[k]            # NOTE: translated from Chinese
                    scores[0, next_id] = -1e9   # NOTE: translated from Chinese
        return scores


def build_tool_ban_ids(tokenizer) -> List[List[int]]:
    """
    构建工具调用相关标记的黑名单，用于禁止模型生成这些标记
    """
    ban_strs = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<|tool_call|>", "<|tool_response|>",
        "<tool>", "</tool>",
        "<function_calls>", "</function_calls>",
    ]
    bad = []
    for s in ban_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids and all(t != tokenizer.unk_token_id for t in ids):
            bad.append(ids)
    return bad


class ConditionalFormatProcessor(LogitsProcessor):
    """
    条件格式处理器：根据已生成的内容控制后续生成
    - 如果已经生成了 false，则阻止生成 task 和 scores
    - 如果已经生成了 true，则允许生成 task 和 scores
    """
    def __init__(self, tokenizer, start_len: int):
        self.tok = tokenizer
        self.start_len = int(start_len)
        
        # NOTE: translated from Chinese
        self.false_pattern = self.tok.encode("false", add_special_tokens=False)
        self.true_pattern = self.tok.encode("true", add_special_tokens=False)
        self.task_start_pattern = self.tok.encode("<|task_start|>", add_special_tokens=False)
        self.scores_pattern = self.tok.encode("scores:", add_special_tokens=False)
        
    def __call__(self, input_ids, scores):
        if input_ids.size(0) != 1:
            return scores
            
        # NOTE: translated from Chinese
        current_seq = input_ids[0, self.start_len:].tolist()
        
        # NOTE: translated from Chinese (check)
        if self._contains_pattern(current_seq, self.false_pattern):
            # NOTE: translated from Chinese
            if self._is_about_to_generate(current_seq, self.task_start_pattern):
                task_start_id = self.task_start_pattern[0]
                scores[0, task_start_id] = -1e9
            if self._is_about_to_generate(current_seq, self.scores_pattern):
                scores_id = self.scores_pattern[0]
                scores[0, scores_id] = -1e9
                
        return scores
    
    def _contains_pattern(self, seq: List[int], pattern: List[int]) -> bool:
        """检查序列中是否包含指定模式"""
        if not pattern:
            return False
        for i in range(len(seq) - len(pattern) + 1):
            if seq[i:i+len(pattern)] == pattern:
                return True
        return False
    
    def _is_about_to_generate(self, seq: List[int], pattern: List[int]) -> bool:
        """检查是否即将生成指定模式"""
        if not pattern:
            return False
        # NOTE: translated from Chinese (check)
        for i in range(1, min(len(pattern), len(seq) + 1)):
            if len(seq) >= i and seq[-i:] == pattern[:i]:
                return True
        return False


def resolve_frame_path(dir_path: str, idx: int) -> Optional[str]:
    candidates = [
        f"{idx}.jpg",
        f"{idx:06d}.jpg",
        f"{idx:05d}.jpg",
        f"{idx:04d}.jpg",
        f"frame_{idx:06d}.jpg",
        f"img_{idx:06d}.jpg",
        f"frame_{idx}.jpg",
        f"img_{idx}.jpg",
    ]
    for name in candidates:
        p = os.path.join(dir_path, name)
        if os.path.exists(p):
            return p
    return None


def load_keyframes_index(path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    递归解析复杂结构的映射文件，返回：
    - video_id -> keyframes_dir（若提供）
    - base_id  -> keyframes_dir（若提供）
    - video_id -> [frame_paths...]（若提供的是路径列表）
    """
    video_to_dir: Dict[str, str] = {}
    base_to_dir: Dict[str, str] = {}
    video_to_paths: Dict[str, List[str]] = {}
    if not path or not os.path.exists(path):
        return video_to_dir, base_to_dir, video_to_paths

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return video_to_dir, base_to_dir, video_to_paths

    def register_identifier(identifier: str, directory: str) -> None:
        if not identifier:
            return
        video_to_dir.setdefault(identifier, directory)
        base_id = extract_base_id(identifier)
        base_to_dir.setdefault(base_id, directory)
        # NOTE: translated from Chinese
        for prefix in ("TSU_", "UCF_CRIME_", "EGO_EXO4D_", "ego_exo4d_", "EGO_", "ego_"):
            if identifier.startswith(prefix):
                stripped = identifier[len(prefix):]
                if stripped:
                    video_to_dir.setdefault(stripped, directory)
                    base_to_dir.setdefault(stripped, directory)
                    base_to_dir.setdefault(extract_base_id(stripped), directory)
            if base_id.startswith(prefix):
                stripped_base = base_id[len(prefix):]
                if stripped_base:
                    base_to_dir.setdefault(stripped_base, directory)

    def ingest(video_id: str, meta: Dict[str, Any]) -> None:
        kfdir = meta.get("keyframes_dir") or meta.get("keyframe_dir") or meta.get("keyframe_folder")
        if isinstance(kfdir, str) and os.path.isdir(kfdir):
            register_identifier(video_id, kfdir)
            take_uid = meta.get("take_uid") or meta.get("take_id")
            if isinstance(take_uid, str):
                register_identifier(take_uid, kfdir)
        # NOTE: translated from Chinese
        kfpaths = meta.get("keyframes") or meta.get("keyframe_paths") or meta.get("frames")
        if isinstance(kfpaths, list) and all(isinstance(x, str) for x in kfpaths):
            video_to_paths[video_id] = [p for p in kfpaths if isinstance(p, str)]

    def traverse(obj: Any) -> None:
        if isinstance(obj, dict):
            # NOTE: translated from Chinese
            if "video_id" in obj and isinstance(obj["video_id"], (str, int)):
                vid = str(obj["video_id"]) or ""
                if vid:
                    ingest(vid, obj)
            # NOTE: translated from Chinese
            for v in obj.values():
                traverse(v)
        elif isinstance(obj, list):
            for v in obj:
                traverse(v)

    traverse(data)
    return video_to_dir, base_to_dir, video_to_paths


def load_keyframes_from_annotation(annotation_path: Optional[str]) -> Dict[str, List[str]]:
    """
    从 annotation 文件中额外加载 keyframe_paths 映射：
    - 直接使用最外层的 video_id 作为键（例如 SITE_A_T09S01A02cam01_03）
    - 同时基于 base_id（去掉最后一段，例如 SITE_A_T09S01A02cam01）注册一份，便于 _01/_02/_03 复用
    返回:
        video_id_or_base_id -> [frame_paths...]
    """
    video_to_paths: Dict[str, List[str]] = {}
    if not annotation_path or not os.path.exists(annotation_path):
        return video_to_paths
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return video_to_paths

    if not isinstance(data, dict):
        return video_to_paths

    for vid, meta in data.items():
        if not isinstance(meta, dict):
            continue
        kfpaths = meta.get("keyframe_paths")
        if not (isinstance(kfpaths, list) and all(isinstance(p, str) for p in kfpaths)):
            continue
        paths = [p for p in kfpaths if isinstance(p, str)]
        if not paths:
            continue
        # NOTE: translated from Chinese
        video_to_paths.setdefault(str(vid), paths)
        # NOTE: translated from Chinese
        base_id = extract_base_id(str(vid))
        if base_id:
            video_to_paths.setdefault(base_id, paths)
    return video_to_paths


def build_task_lookup(top_map_path: str) -> Dict[str, str]:
    """
    构建 video_id -> 顶层任务名 的映射。
    要求：映射文件的最上层 key 为任务名；任意深度处包含若干含有 "video_id" 的条目。
    """
    out: Dict[str, str] = {}
    if not top_map_path or not os.path.exists(top_map_path):
        return out
    try:
        with open(top_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return out

    def traverse(node: Any, top_key: str) -> None:
        if isinstance(node, dict):
            if "video_id" in node and isinstance(node["video_id"], (str, int)):
                out[str(node["video_id"])]= top_key
            for v in node.values():
                traverse(v, top_key)
        elif isinstance(node, list):
            for v in node:
                traverse(v, top_key)

    if isinstance(data, dict):
        for k, v in data.items():
            traverse(v, str(k))
    return out

class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        window_size: int,
        window_stride: int,
        use_trigger_hints: bool,
        trigger_json_path: Optional[str],
        tokenizer: Any,
        if_score: bool,
        frame_root: Optional[str] = None,
        # NOTE: translated from Chinese
        annotation_path: Optional[str] = None,
        processor: Any = None,
        record_path: Optional[str] = None,
        preprocessed_data_dir: Optional[str] = None,
        preprocessed_data_file: Optional[str] = None,
        preprocessed_train_file: Optional[str] = None,
        preprocessed_val_file: Optional[str] = None,
        priority_score_path: Optional[str] = None,
        # NOTE: translated from Chinese
        enable_evidence_frames: bool = False,
        enable_reasoning: bool = False,
        enable_confidence: bool = False,
        enable_scores: bool = False,
        use_reasoning_tokens: bool = False,
        random_seed: int = 42,
        max_image_long_edge: int = 896,
        predict_steps: int = 0,
        predict_next_action: bool = False,
        history_memory_mode: str = "none",
        history_recent_k: int = 4,
    ) -> None:
        super().__init__()
        # NOTE: translated from Chinese (cache)
        self._jsonl_path = jsonl_path
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self._annotation_path = annotation_path
        self._tg_cache: Dict[str, "TaskGraphManager"] = {}
        self.predict_next_action = bool(predict_next_action)
        self.history_memory_mode = str(history_memory_mode or "none").strip().lower()
        self.history_recent_k = max(1, int(history_recent_k or 1))
        silence, is_main, _, _ = _get_env_silence_and_rank()
        silence, is_main, _, _ = _get_env_silence_and_rank()
        # NOTE: translated from Chinese
        silence, is_main, _, _ = _get_env_silence_and_rank()
        self._record_path = record_path
        self.if_score = if_score
        self._frame_root = frame_root
        self._processor = processor
        self._vocab_id_to_name: Dict[int, str] = load_vocabulary_from_annotation(annotation_path)
        self._missing_vocab_labels: Set[int] = set()
        self.priority_scores: Dict[str, Dict[str, int]] = load_priority_scores(priority_score_path)
        self._missing_priority_tasks: Set[str] = set()
        self._frame_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._frame_cache_max = 512
        self._frame_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._frame_cache_max = 512
        # NOTE: translated from Chinese (task)
        self.task_lookup: Dict[str, str] = {}
        self.video_task_names: Dict[str, str] = {}
        self.video_scenarios: Dict[str, str] = load_scenarios_from_annotation(annotation_path)
        self.gt_tasks: List[str] = []
        self.gt_steps: List[str] = []
        self.gt_future_steps: List[List[str]] = []
        self.gt_next_actions: List[str] = []
        self.predict_steps = max(0, int(predict_steps or 0))
        
        # NOTE: translated from Chinese
        self.enable_evidence_frames = enable_evidence_frames
        self.enable_reasoning = enable_reasoning
        self.enable_confidence = enable_confidence
        self.enable_scores = enable_scores
        self.use_reasoning_tokens = use_reasoning_tokens
        self._random_seed = random_seed
        self._max_image_long_edge = int(max_image_long_edge) if max_image_long_edge else 0
        self._frame_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._frame_cache_max = 512
        self._random_seed = random_seed
        self._frame_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._frame_cache_max = 512
        self._random_seed = random_seed
        self._frame_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._frame_cache_max = 512
        self._random_seed = random_seed
        self._random_seed = random_seed
        
        # NOTE: translated from Chinese
        preprocessed_file = None
        
        # NOTE: translated from Chinese
        if preprocessed_train_file and "train" in jsonl_path.lower():
            preprocessed_file = preprocessed_train_file
        elif preprocessed_val_file and "val" in jsonl_path.lower():
            preprocessed_file = preprocessed_val_file
        elif preprocessed_data_file:
            # NOTE: translated from Chinese
            preprocessed_file = preprocessed_data_file
        elif preprocessed_data_dir:
            # NOTE: translated from Chinese
            import hashlib
            param_str = (
                f"{jsonl_path}_{window_size}_{window_stride}_{use_trigger_hints}_{if_score}_"
                f"{frame_root}_{self.enable_evidence_frames}_{self.enable_reasoning}_"
                f"{self.enable_confidence}_{self.enable_scores}_{os.environ.get('LABELS_ONLY','0')}_"
                f"maxedge:{self._max_image_long_edge}_predSteps:{self.predict_steps}_"
                f"predNextAction:{self.predict_next_action}_"
                f"histMode:{self.history_memory_mode}_histRecent:{self.history_recent_k}_"
                f"reasoningSchema:4sent_v1"
            )
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            preprocessed_file = os.path.join(preprocessed_data_dir, f"preprocessed_{param_hash}.pkl")
        
        # NOTE: translated from Chinese
        if preprocessed_file and os.path.exists(preprocessed_file):
            if not silence and is_main:
                print(f"正在加载预处理数据: {preprocessed_file}")
            with open(preprocessed_file, "rb") as f:
                data_dict = pickle.load(f)
                self.data = data_dict["data"]
                self.targets = data_dict["targets"]
                self.gt_tasks = data_dict.get("gt_tasks", [""] * len(self.data))
                self.gt_steps = data_dict.get("gt_steps", [""] * len(self.data))
                self.gt_future_steps = data_dict.get("gt_future_steps", [[] for _ in range(len(self.data))])
                self.gt_next_actions = data_dict.get("gt_next_actions", [""] * len(self.data))
            if not silence and is_main:
                print(f"预处理数据加载完成，共 {len(self.data)} 个样本")
            return
        
        # NOTE: translated from Chinese
        lock_file = None
        if preprocessed_file:
            lock_file = preprocessed_file + ".lock"
        
        should_process = True
        if lock_file:
            # NOTE: translated from Chinese
            try:
                lock_fd = open(lock_file, 'w')
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not silence and is_main:
                    print("获取到数据预处理锁，开始处理原始数据...")
            except (IOError, OSError):
                # NOTE: translated from Chinese
                if not silence and is_main:
                    print("等待其他进程完成数据预处理...")
                lock_fd = None
                should_process = False
                
                # NOTE: translated from Chinese
                max_wait = 300  # NOTE: translated from Chinese
                wait_time = 0
                while wait_time < max_wait:
                    if os.path.exists(preprocessed_file):
                        break
                    time.sleep(1)
                    wait_time += 1
                
                if os.path.exists(preprocessed_file):
                    if not silence and is_main:
                        print(f"检测到预处理数据已生成，正在加载: {preprocessed_file}")
                    with open(preprocessed_file, "rb") as f:
                        data_dict = pickle.load(f)
                        self.data = data_dict["data"]
                        self.targets = data_dict["targets"]
                        self.gt_tasks = data_dict.get("gt_tasks", [""] * len(self.data))
                        self.gt_steps = data_dict.get("gt_steps", [""] * len(self.data))
                        self.gt_future_steps = data_dict.get("gt_future_steps", [[] for _ in range(len(self.data))])
                        self.gt_next_actions = data_dict.get("gt_next_actions", [""] * len(self.data))
                    if not silence and is_main:
                        print(f"预处理数据加载完成，共 {len(self.data)} 个样本")
                    return
                else:
                    if not silence and is_main:
                        print("等待超时，开始处理数据...")
                    should_process = True
        
        if should_process:
            if not silence and is_main:
                print("开始处理原始数据...")
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        self._video_to_paths: Dict[str, List[str]] = {}
        self._video_to_dir: Dict[str, str] = {}
        self._base_to_dir: Dict[str, str] = {}
        if annotation_path:
            # NOTE: translated from Chinese
            anno_vdir, anno_bdir = load_keyframe_dirs_from_annotation(annotation_path)
            if anno_vdir:
                self._video_to_dir.update(anno_vdir)
            if anno_bdir:
                self._base_to_dir.update(anno_bdir)
            # NOTE: translated from Chinese
            anno_paths = load_keyframes_from_annotation(annotation_path)
            if anno_paths:
                self._video_to_paths.update(anno_paths)
        if self._record_path:
            os.makedirs(os.path.dirname(self._record_path), exist_ok=True)
            # NOTE: translated from Chinese
            with open(self._record_path, "w", encoding="utf-8") as _f:
                pass
        rows = read_jsonl(jsonl_path)
        # NOTE: translated from Chinese
        video_ids: Set[str] = set()
        for r in rows:
            vid = r.get("video_id")
            if isinstance(vid, str):
                video_ids.add(vid)
        # NOTE: translated from Chinese

        # NOTE: translated from Chinese
        samples: List[Dict[str, Any]] = []
        self.targets: List[int] = []  # NOTE: translated from Chinese
        for r in tqdm(
            rows,
            desc="处理视频数据",
            unit="视频",
            disable=(silence and not is_main),
        ):
            vid = r["video_id"]
            frame_labels: List[int] = r.get("frame_labels", [])
            frame_task_labels: List[int] = r.get("frame_task_labels", [])
            n = len(frame_labels)
            if n == 0:
                continue
            base_id = extract_base_id(vid)
            # NOTE: translated from Chinese
            dir_path: Optional[str] = None
            video_to_dir = getattr(self, "_video_to_dir", None)
            base_to_dir = getattr(self, "_base_to_dir", None)
            video_to_paths = getattr(self, "_video_to_paths", None)
            base_id = extract_base_id(vid)
            if video_to_dir or base_to_dir:
                dir_path = (video_to_dir.get(vid) if video_to_dir else None) or (
                    base_to_dir.get(base_id) if base_to_dir else None
                )
            if dir_path is None:
                _frame_root = getattr(self, "_frame_root", None)
                if _frame_root:
                    # NOTE: translated from Chinese
                    candidates = [vid, base_id, base_id.replace("TSU_", "")]
                    for cand in candidates:
                        p = os.path.join(_frame_root, cand)
                        if os.path.isdir(p):
                            dir_path = p
                            break
            files_with_idx: List[Tuple[str, int]] = []
            # NOTE: translated from Chinese
            if video_to_paths:
                path_list = video_to_paths.get(vid) or video_to_paths.get(base_id) or video_to_paths.get(base_id.replace("TSU_", ""))
                if path_list:
                    tmp: List[Tuple[str, int]] = []
                    for pth in path_list:
                        try:
                            fname = os.path.basename(pth)
                            m = re.search(r"(\d+)", fname)
                            if not m:
                                continue
                            idx_int = int(m.group(1))
                            if os.path.isfile(pth):
                                tmp.append((pth, idx_int))
                        except Exception:
                            continue
                    tmp.sort(key=lambda x: x[1])
                    files_with_idx.extend(tmp)
            # NOTE: translated from Chinese
            if not files_with_idx and dir_path and os.path.isdir(dir_path):
                try:
                    tmp: List[Tuple[str, int]] = []
                    for nm in os.listdir(dir_path):
                        m = re.search(r"(\d+)", nm)
                        if not m:
                            continue
                        idx_int = int(m.group(1))
                        pth = os.path.join(dir_path, nm)
                        if os.path.isfile(pth):
                            tmp.append((pth, idx_int))
                    tmp.sort(key=lambda x: x[1])
                    files_with_idx.extend(tmp)
                except Exception:
                    pass

            # If we cannot resolve any frame file paths for this video, skip it in lazy mode.
            # Otherwise, TRL may crash during trainer init when it samples the first item.
            if not files_with_idx:
                if not silence and is_main:
                    print(
                        f"[WARN] Lazy dataset: no frame files found, skip video. "
                        f"video_id={vid}, base_id={base_id}, dir_path={dir_path}, "
                        f"frame_root={self._frame_root}, annotation_path={annotation_path}"
                    )
                continue
            # NOTE: translated from Chinese
            trigger_hint = None

            # NOTE: translated from Chinese
            if not files_with_idx:
                raise RuntimeError(f"未找到任何帧文件: video_id={vid}, base_id={base_id}, dir_path={dir_path}")

            # NOTE: translated from Chinese
            window_count = len(range(0, n, window_stride))
            for end in tqdm(range(0, n, window_stride), desc=f"处理视频 {vid} 的窗口", total=window_count, unit="窗口", leave=False):
                start = max(0, end - window_size + 1)
                window_files = files_with_idx[start : end + 1] if files_with_idx else []
                # NOTE: translated from Chinese
                images: List[Image.Image] = []
                ok = True
                # NOTE: translated from Chinese
                # NOTE: translated from Chinese
                frame_descs: List[str] = []
                if window_files:
                    idx0 = window_files[0][1]
                    for j, (pth, idx_int) in tqdm(
                        enumerate(window_files),
                        desc="加载图像",
                        unit="帧",
                        leave=False,
                        total=len(window_files),
                        disable=len(window_files) <= 1,
                    ):
                        t = (idx_int - idx0) / 25.0
                        frame_descs.append(f"F{j} [idx={idx_int} t={t:.2f}s]")
                        # NOTE: translated from Chinese
                        try:
                            images.append(self._load_image_cached(pth))
                        except Exception as e:
                            ok = False
                            err_msg = (
                                f"加载帧图像失败: video_id={vid}, path={pth}, "
                                f"idx={idx_int}, window=({start},{end}), error={e}"
                            )
                            raise RuntimeError(err_msg) from e

                # NOTE: translated from Chinese
                if not images or not ok:
                    raise RuntimeError(
                        f"窗口未正确加载到任何图像: video_id={vid}, "
                        f"base_id={base_id}, window=({start},{end}), "
                        f"window_files={len(window_files)}"
                    )

                label_value = -1
                if frame_labels and end < len(frame_labels):
                    try:
                        label_value = int(frame_labels[end])
                    except (TypeError, ValueError):
                        label_value = -1
                is_trigger = label_value > 0

                # NOTE: translated from Chinese
                step_name = ""
                if is_trigger:
                    step_name = self._vocab_id_to_name.get(label_value, "")
                    if not step_name:
                        if label_value not in self._missing_vocab_labels:
                            print(f"警告：step 标签 ID {label_value} 未在 vocabulary 中找到，视频 {vid}")
                            self._missing_vocab_labels.add(label_value)
                        step_name = str(label_value)

                # NOTE: translated from Chinese
                task_label = -1
                if frame_task_labels and end < len(frame_task_labels):
                    try:
                        task_label = int(frame_task_labels[end])
                    except (TypeError, ValueError):
                        task_label = -1
                task_name_output = ""
                if is_trigger and task_label > 0:
                    task_name_output = self._vocab_id_to_name.get(task_label, "")
                    if not task_name_output:
                        # NOTE: translated from Chinese
                        if task_label not in self._missing_vocab_labels:
                            print(f"警告：task 标签 ID {task_label} 未在 vocabulary 中找到，视频 {vid}")
                            self._missing_vocab_labels.add(task_label)
                        task_name_output = str(task_label)
                    # NOTE: translated from Chinese
                    self.video_task_names.setdefault(vid, task_name_output)
                    if base_id:
                        self.video_task_names.setdefault(base_id, task_name_output)
                else:
                    # NOTE: translated from Chinese
                    task_name_output = self.video_task_names.get(vid, "") or self.video_task_names.get(base_id, "")
                    if not task_name_output:
                        task_name_output = self.task_lookup.get(vid, "") or self.task_lookup.get(base_id, "")
                task_name_output = task_name_output.strip() if isinstance(task_name_output, str) else str(task_name_output).strip()

                # NOTE: translated from Chinese (step)
                future_steps_list: List[str] = collect_future_actions(
                    frame_labels=frame_labels,
                    current_idx=end,
                    predict_steps=self.predict_steps,
                    vocab_map=self._vocab_id_to_name,
                    missing_vocab_labels=self._missing_vocab_labels,
                    video_id=vid,
                )
                completed_steps: List[str] = collect_completed_steps_before_current_segment(
                    frame_labels, end, self._vocab_id_to_name
                )
                next_action = ""
                if is_trigger and self.predict_next_action:
                    next_action = teacher_next_action(
                        task_name_output, completed_steps, step_name,
                        future_steps_list, self._annotation_path, self._tg_cache,
                    )
                enabled_steps, parallel_steps = derive_enabled_and_parallel_steps(
                    task_name=task_name_output,
                    completed_steps=completed_steps,
                    current_step=step_name,
                    human_future_steps=future_steps_list,
                    annotation_path=self._annotation_path,
                    tg_cache=self._tg_cache,
                )
                evidence_pos = select_evidence_positions(frame_labels, start, end)
                if not evidence_pos:
                    evidence_pos = [max(0, end - start)]
                evidence_frames = [f"F{k}" for k in evidence_pos]
                window_steps = build_window_steps(frame_labels, start, end, self._vocab_id_to_name)
                prev_step = completed_steps[-1] if completed_steps else None
                env_desc = resolve_scenario_text(r, vid, self.video_scenarios)
                trigger_reason = str(r.get("trigger_en", "") or r.get("trigger_reason", "") or "").strip() or None

                # NOTE: translated from Chinese
                # print(f"images: {images}")
                # print(f"frame_descs: {frame_descs}")
                # print(f"vid: {vid}")
                # print(f"end: {end}")
                # print(f"start: {start}")
                # print(f"window_files: {window_files}")
                # print(f"idx0: {idx0}")
                # print(f"idx_int: {idx_int}")
                # print(f"t: {t}")

                # NOTE: translated from Chinese
                if not images:
                    continue

                sample_idx = len(samples)
                use_labels_only = bool(os.environ.get("LABELS_ONLY", "0") == "1")

                bundle = build_two_stage_conversation_bundle(
                    video_id=vid,
                    frame_descs=frame_descs,
                    images=images,
                    completed_steps=completed_steps,
                    is_trigger=bool(is_trigger),
                    task_name=task_name_output,
                    step_name=step_name,
                    future_steps=future_steps_list if is_trigger else [],
                    next_action=next_action if is_trigger else "",
                    predict_steps=self.predict_steps,
                    predict_next_action=self.predict_next_action,
                    history_mode=self.history_memory_mode,
                    history_recent_k=self.history_recent_k,
                )
                dynamic_system_prompt = bundle["system_prompt_text"]
                history_memory_text = bundle["history_memory_text"]
                messages_prompt = bundle["messages_prompt"]
                messages_full = bundle["messages_full"]
                assistant_target = bundle["assistant_target"]

                text_prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
                text_full = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)

                enc_full = self._processor(text=[text_full], images=[images], return_tensors="pt")
                enc_prompt_mm = self._processor(text=[text_prompt], images=[images], return_tensors="pt")
                if "pixel_values" in enc_full:
                    enc_full["pixel_values"] = enc_full["pixel_values"].to(torch.bfloat16)
                if "pixel_values" in enc_prompt_mm:
                    enc_prompt_mm["pixel_values"] = enc_prompt_mm["pixel_values"].to(torch.bfloat16)

                ids_prompt = enc_prompt_mm["input_ids"]
                ids_full = enc_full["input_ids"]

                # NOTE: translated from Chinese
                full_ids = ids_full.squeeze(0)                  # [T]
                prompt_len = int(ids_prompt.shape[1])
                # NOTE: translated from Chinese
                labels = ids_full.clone().detach()  # NOTE: translated from Chinese
                labels[0, :prompt_len] = -100  # NOTE: translated from Chinese
                attention_mask = torch.ones_like(ids_full, dtype=torch.long)

                # NOTE: translated from Chinese
                RE_L, RE_R = "<|reasoning_start|>", "<|reasoning_end|>"
                IS_L, IS_R = "<|trigger_start|>", "<|trigger_end|>"
                TK_L, TK_R = "<|task_start|>", "<|task_end|>"
                ST_L, ST_R = "<|step_start|>", "<|step_end|>"
            
                is_l_ids = tokenizer.encode(IS_L, add_special_tokens=False)
                is_r_ids = tokenizer.encode(IS_R, add_special_tokens=False)
                tk_l_ids = tokenizer.encode(TK_L, add_special_tokens=False)
                tk_r_ids = tokenizer.encode(TK_R, add_special_tokens=False)
                st_l_ids = tokenizer.encode(ST_L, add_special_tokens=False)
                st_r_ids = tokenizer.encode(ST_R, add_special_tokens=False)
                st_l_ids = tokenizer.encode(ST_L, add_special_tokens=False)
                st_r_ids = tokenizer.encode(ST_R, add_special_tokens=False)
                re_l_ids = tokenizer.encode(RE_L, add_special_tokens=False)
                re_r_ids = tokenizer.encode(RE_R, add_special_tokens=False)
                scores_ids = tokenizer.encode("scores:", add_special_tokens=False)
                fs_l_ids = tokenizer.encode("<|future_steps_start|>", add_special_tokens=False)
                fs_r_ids = tokenizer.encode("<|future_steps_end|>", add_special_tokens=False)
                NA_L, NA_R = "<|next_action_start|>", "<|next_action_end|>"
                na_l_ids = tokenizer.encode(NA_L, add_special_tokens=False)
                na_r_ids = tokenizer.encode(NA_R, add_special_tokens=False)
                eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

                full_list = full_ids.tolist()

                def find(a, pat, start=0):
                    return _find_subseq(a, pat, start=start)

                def supervise_span(start_idx, end_idx_exclusive):
                    if start_idx != -1 and end_idx_exclusive != -1 and end_idx_exclusive > start_idx:
                        labels[0, start_idx:end_idx_exclusive] = ids_full[0, start_idx:end_idx_exclusive]

                # NOTE: translated from Chinese
                s_re_l = s_re_r = -1
                s_is = e_is = -1
                s_tk = e_tk = -1
                s_st = e_st = -1
                s_sc = e_sc = -1
                s_fs = e_fs = -1
                s_na = e_na = -1
                
                # NOTE: translated from Chinese
                if self.use_reasoning_tokens:
                    # NOTE: translated from Chinese
                    s_re_l = find(full_list, re_l_ids, start=prompt_len)
                    s_re_r = find(full_list, re_r_ids, start=max(prompt_len, s_re_l if s_re_l!=-1 else prompt_len))
                    if s_re_l != -1 and s_re_r != -1 and s_re_r > s_re_l:
                        # Supervise the full 4-sentence reasoning block.
                        supervise_span(s_re_l, s_re_r + len(re_r_ids))

                # NOTE: translated from Chinese
                s_is = find(full_list, is_l_ids, start=prompt_len)
                e_is = find(full_list, is_r_ids, start=prompt_len)
                if s_is != -1 and e_is != -1:
                    supervise_span(s_is, e_is + len(is_r_ids))

                # NOTE: translated from Chinese
                s_tk = find(full_list, tk_l_ids, start=prompt_len)
                e_tk = find(full_list, tk_r_ids, start=prompt_len)
                if s_tk != -1 and e_tk != -1:
                    supervise_span(s_tk, e_tk + len(tk_r_ids))

                # NOTE: translated from Chinese
                s_st = find(full_list, st_l_ids, start=prompt_len)
                e_st = find(full_list, st_r_ids, start=prompt_len)
                if s_st != -1 and e_st != -1:
                    supervise_span(s_st, e_st + len(st_r_ids))

                # NOTE: translated from Chinese
                if is_trigger:
                    s_sc = find(full_list, scores_ids, start=prompt_len)
                    if s_sc != -1:
                        # NOTE: translated from Chinese
                        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
                        if len(nl_ids) == 1:
                            nl_id = nl_ids[0]
                            for k in range(s_sc+1, len(full_list)):
                                if full_list[k] == nl_id:
                                    e_sc = k
                                    break
                        if e_sc == -1:
                            if eos_id in full_list:
                                e_sc = full_list.index(eos_id)
                            else:
                                e_sc = len(full_list)
                        supervise_span(s_sc, e_sc)
                
                # NOTE: translated from Chinese (config)
                if self.predict_steps > 0 and future_steps_list:
                    s_fs = find(full_list, fs_l_ids, start=prompt_len)
                    e_fs = find(full_list, fs_r_ids, start=prompt_len)
                    if s_fs != -1 and e_fs != -1:
                        supervise_span(s_fs, e_fs + len(fs_r_ids))
                if is_trigger and self.predict_next_action:
                    s_na = find(full_list, na_l_ids, start=prompt_len)
                    e_na = find(full_list, na_r_ids, start=prompt_len)
                    if s_na != -1 and e_na != -1:
                        supervise_span(s_na, e_na + len(na_r_ids))
                
                # NOTE: translated from Chinese (debug)
                if len(samples) == 0:  # NOTE: translated from Chinese
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    if not silence and is_main:
                        print(f"DEBUG: Special tokens encoding:")
                        print(f"  {RE_L} -> {re_l_ids}")
                        print(f"  {RE_R} -> {re_r_ids}")
                        print(f"  {IS_L} -> {is_l_ids}")
                        print(f"  {IS_R} -> {is_r_ids}")
                        print(f"  {TK_L} -> {tk_l_ids}")
                        print(f"  {TK_R} -> {tk_r_ids}")
                        print(f"  assistant_target: {assistant_target}")

                    # NOTE: translated from Chinese (debug)
                    start_is = find(full_list, is_l_ids, start=prompt_len)
                    end_is   = find(full_list, is_r_ids, start=prompt_len)
                    is_region = [start_is, end_is + len(is_r_ids)] if start_is != -1 and end_is != -1 else [-1, -1]
                
                    start_tk = find(full_list, tk_l_ids, start=prompt_len)
                    end_tk   = find(full_list, tk_r_ids, start=prompt_len)
                    task_region = [start_tk, end_tk + len(tk_r_ids)] if start_tk != -1 and end_tk != -1 else [-1, -1]
                    start_st = find(full_list, st_l_ids, start=prompt_len)
                    end_st = find(full_list, st_r_ids, start=prompt_len)
                    step_region = [start_st, end_st + len(st_r_ids)] if start_st != -1 and end_st != -1 else [-1, -1]
                    trigger_token_pos = -1
                if start_is != -1:
                    trigger_token_pos = start_is + len(is_l_ids)

                    item = {
                    "input_ids": ids_full.squeeze(0),
                    "labels": labels.squeeze(0),
                    "attention_mask": attention_mask.squeeze(0),
                    # NOTE: translated from Chinese
                    "prompt_input_ids": ids_prompt.squeeze(0),
                    "prompt_attention_mask": torch.ones_like(ids_prompt.squeeze(0), dtype=torch.long),
                    # NOTE: translated from Chinese (debug)
                    "debug_prompt_text": text_prompt,
                    "debug_target_text": assistant_target,
                    "is_region": is_region,
                    "task_region": task_region,
                    "step_region": step_region,
                    "trigger_label": torch.tensor(int(is_trigger), dtype=torch.long),
                    "trigger_position": torch.tensor(int(trigger_token_pos if trigger_token_pos >= 0 else max(prompt_len, 0)), dtype=torch.long),
                    "future_steps": future_steps_list,
                    "next_action": next_action if is_trigger else "",
                    "ts_two_stage": True,
                    "ts_system_prompt_text": dynamic_system_prompt,
                    "ts_state_user_text": bundle["state_user_text"],
                    "ts_decision_user_text": bundle["decision_user_text"],
                    "ts_window_files": [pth for (pth, _) in window_files],
                    "ts_max_image_long_edge": int(getattr(self, "_max_image_long_edge", 0) or 0),
                    }
                if "pixel_values" in enc_full:
                    item["pixel_values"] = enc_full["pixel_values"].squeeze(0)
                if "image_grid_thw" in enc_full:
                    grid = enc_full["image_grid_thw"]
                    if isinstance(grid, torch.Tensor):
                        # NOTE: translated from Chinese
                        if grid.dim() >= 3:
                            grid = grid.squeeze(0)
                        grid = grid.view(-1, 3)
                    item["image_grid_thw"] = grid
                    # NOTE: translated from Chinese
                if "pixel_values" in enc_prompt_mm:
                    item["prompt_pixel_values"] = enc_prompt_mm["pixel_values"].squeeze(0)
                if "image_grid_thw" in enc_prompt_mm:
                    pgrid = enc_prompt_mm["image_grid_thw"]
                    if isinstance(pgrid, torch.Tensor):
                        if pgrid.dim() >= 3:
                            pgrid = pgrid.squeeze(0)
                        pgrid = pgrid.view(-1, 3)
                    item["prompt_image_grid_thw"] = pgrid
                    samples.append(item)
                    self.targets.append(1 if is_trigger else 0)
                    self.gt_tasks.append(task_name_output if is_trigger else "")
                    self.gt_steps.append(step_name if is_trigger else "")
                    self.gt_future_steps.append(future_steps_list if is_trigger else [])
                    self.gt_next_actions.append(next_action if is_trigger else "")
                if self._record_path:
                    out_label = {"is_trigger": bool(is_trigger)}
                    if (self.enable_scores or self.if_score) and is_trigger:
                        out_label["scores"] = resolve_priority_scores(
                            task_name_output,
                            self.priority_scores,
                            missing_cache=self._missing_priority_tasks,
                        )
                    with open(self._record_path, "a", encoding="utf-8") as rf:
                        wf_obj = {
                            "video_id": vid,
                            "frame_descs": frame_descs,
                            "prompt": text_full,
                            "label": out_label,
                            "prompt_text": text_prompt,
                            "assistant_target": assistant_target,
                            "is_region": is_region,
                            "task_region": task_region,
                            "task_name": task_name_output if is_trigger else "",
                            "step_name": step_name if is_trigger else "",
                            "step_region": step_region,
                            "future_steps": future_steps_list if is_trigger else [],
                            "next_action": next_action if is_trigger else "",
                        }
                        rf.write(json.dumps(wf_obj, ensure_ascii=False) + "\n")

        self.data = samples
        
        # NOTE: translated from Chinese
        if preprocessed_file:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"正在保存预处理数据到: {preprocessed_file}")
            os.makedirs(os.path.dirname(preprocessed_file), exist_ok=True)
            with open(preprocessed_file, "wb") as f:
                pickle.dump({
                    "data": self.data,
                    "targets": self.targets,
                    "gt_tasks": self.gt_tasks,
                    "gt_steps": self.gt_steps,
                    "gt_future_steps": self.gt_future_steps,
                    "gt_next_actions": self.gt_next_actions,
                }, f)
            if not silence and is_main:
                print(f"预处理数据保存完成，共 {len(self.data)} 个样本")
            
            # NOTE: translated from Chinese
            if 'lock_fd' in locals() and lock_fd:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                    lock_fd.close()
                    # NOTE: translated from Chinese
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                    if not silence and is_main:
                        print("数据预处理锁已释放")
                except Exception as e:
                    if not silence and is_main:
                        print(f"释放锁时出错: {e}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def _load_image_cached(self, path: str) -> Image.Image:
        cached = self._frame_cache.get(path)
        if cached is not None:
            self._frame_cache.move_to_end(path)
            return cached.copy()
        with Image.open(path) as im:
            img = im.convert("RGB").copy()
        mle = getattr(self, "_max_image_long_edge", 0)
        if isinstance(mle, int) and mle > 0:
            w, h = img.size
            m = max(w, h)
            if m > mle:
                scale = float(mle) / float(m)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), Image.LANCZOS)
        self._frame_cache[path] = img
        if len(self._frame_cache) > self._frame_cache_max:
            self._frame_cache.popitem(last=False)
        return img.copy()


class LazySlidingWindowDataset(Dataset):
    """
    懒加载版本的 SlidingWindowDataset
    __init__ 时只记录元数据，__getitem__ 时才加载图像和 tokenize
    适合大规模数据集，配合 DataLoader 的 num_workers 并行加载
    """
    
    def __init__(
        self,
        jsonl_path: str,
        window_size: int,
        window_stride: int,
        use_trigger_hints: bool,
        trigger_json_path: Optional[str],
        tokenizer: Any,
        if_score: bool,
        frame_root: Optional[str] = None,
        # NOTE: translated from Chinese
        annotation_path: Optional[str] = None,
        processor: Any = None,
        record_path: Optional[str] = None,
        preprocessed_data_dir: Optional[str] = None,
        preprocessed_data_file: Optional[str] = None,
        preprocessed_train_file: Optional[str] = None,
        preprocessed_val_file: Optional[str] = None,
        priority_score_path: Optional[str] = None,
        enable_evidence_frames: bool = False,
        enable_reasoning: bool = False,
        enable_confidence: bool = False,
        enable_scores: bool = False,
        use_reasoning_tokens: bool = False,
        random_seed: int = 42,
        max_image_long_edge: int = 896,
        predict_steps: int = 0,
        predict_next_action: bool = False,
        history_memory_mode: str = "none",
        history_recent_k: int = 4,
        distill_label_map_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        # NOTE: translated from Chinese (cache)
        self._jsonl_path = jsonl_path
        self._annotation_path = annotation_path
        self._tg_cache: Dict[str, "TaskGraphManager"] = {}
        self.predict_next_action = bool(predict_next_action)
        self.history_memory_mode = str(history_memory_mode or "none").strip().lower()
        self.history_recent_k = max(1, int(history_recent_k or 1))
        self.tokenizer = tokenizer
        self._processor = processor
        self.if_score = if_score
        self._frame_root = frame_root
        self.window_size = window_size
        self.window_stride = window_stride
        
        # NOTE: translated from Chinese
        self.enable_evidence_frames = enable_evidence_frames
        self.enable_reasoning = enable_reasoning
        self.enable_confidence = enable_confidence
        self.enable_scores = enable_scores
        self.use_reasoning_tokens = use_reasoning_tokens
        self._random_seed = random_seed
        self._frame_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._frame_cache_max = 512
        self._max_image_long_edge = int(max_image_long_edge) if max_image_long_edge else 0
        self.predict_steps = max(0, int(predict_steps or 0))
        self._lazy_label_cache: Dict[int, Dict[str, Any]] = {}
        self._lazy_label_cache_updates: Dict[int, Dict[str, Any]] = {}
        self._lazy_label_cache_flush_every = 8

        self._distill_label_map: Dict[int, str] = {}
        if distill_label_map_path:
            import json as _cd_json
            with open(distill_label_map_path, "r", encoding="utf-8") as _cd_f:
                _cd_raw_map = _cd_json.load(_cd_f)
            self._distill_label_map = {
                int(_k): str(_v) for _k, _v in _cd_raw_map.items() if str(_v or "").strip()
            }
            print(
                f"[distill] loaded {len(self._distill_label_map)} teacher labels from "
                f"{distill_label_map_path}",
                flush=True,
            )

        # NOTE: translated from Chinese
        self._vocab_id_to_name: Dict[int, str] = load_vocabulary_from_annotation(annotation_path)
        self._missing_vocab_labels: Set[int] = set()
        
        # NOTE: translated from Chinese
        self.priority_scores: Dict[str, Dict[str, int]] = load_priority_scores(priority_score_path)
        self._missing_priority_tasks: Set[str] = set()
        
        # NOTE: translated from Chinese (task)
        self.task_lookup: Dict[str, str] = {}
        self.video_task_names: Dict[str, str] = {}
        self.video_scenarios: Dict[str, str] = load_scenarios_from_annotation(annotation_path)
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        self._video_to_paths: Dict[str, List[str]] = {}
        self._video_to_dir: Dict[str, str] = {}
        self._base_to_dir: Dict[str, str] = {}
        if annotation_path:
            anno_vdir, anno_bdir = load_keyframe_dirs_from_annotation(annotation_path)
            if anno_vdir:
                self._video_to_dir.update(anno_vdir)
            if anno_bdir:
                self._base_to_dir.update(anno_bdir)
            anno_paths = load_keyframes_from_annotation(annotation_path)
            if anno_paths:
                self._video_to_paths.update(anno_paths)

        # Build a persistent lazy-label cache key for both train/val splits.
        import hashlib
        cache_root = Path(preprocessed_data_dir).resolve() if preprocessed_data_dir else (ROOT / "cot_sft_v2" / "data" / "lazy_label_cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_signature = {
            "jsonl": str(jsonl_path),
            "window_size": int(window_size),
            "window_stride": int(window_stride),
            "predict_steps": int(self.predict_steps),
            "predict_next_action": bool(self.predict_next_action),
            "history_memory_mode": str(self.history_memory_mode),
            "history_recent_k": int(self.history_recent_k),
            "annotation": str(annotation_path or ""),
            "frame_root": str(frame_root or ""),
        }
        cache_key = hashlib.md5(json.dumps(cache_signature, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        self._lazy_label_cache_prefix = str(cache_root / f"lazy_labels_{cache_key}")
        self._load_lazy_label_cache()
        
        # NOTE: translated from Chinese
        rows = read_jsonl(jsonl_path)
        
        # NOTE: translated from Chinese
        self.samples_meta: List[Dict[str, Any]] = []
        self.targets: List[int] = []
        self.gt_tasks: List[str] = []
        self.gt_steps: List[str] = []
        self.gt_future_steps: List[List[str]] = []
        self.gt_next_actions: List[str] = []
        
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"正在构建样本元数据（懒加载模式）...")
            print("说明：懒加载模式仅在此阶段扫描元数据；图像读取与tokenize会在__getitem__按需执行。")
        for r in tqdm(
            rows,
            desc="扫描视频",
            unit="视频",
            disable=(silence and not is_main),
        ):
            vid = r["video_id"]
            frame_labels: List[int] = r.get("frame_labels", [])
            frame_task_labels: List[int] = r.get("frame_task_labels", [])
            n = len(frame_labels)
            if n == 0:
                continue
            
            base_id = extract_base_id(vid)
            
            # NOTE: translated from Chinese
            dir_path: Optional[str] = None
            if self._video_to_dir or self._base_to_dir:
                dir_path = (self._video_to_dir.get(vid) if self._video_to_dir else None) or (
                    self._base_to_dir.get(base_id) if self._base_to_dir else None
                )
            if dir_path is None and self._frame_root:
                candidates = [vid, base_id, base_id.replace("TSU_", "")]
                for cand in candidates:
                    p = os.path.join(self._frame_root, cand)
                    if os.path.isdir(p):
                        dir_path = p
                        break
            
            files_with_idx: List[Tuple[str, int]] = []
            if self._video_to_paths:
                path_list = self._video_to_paths.get(vid) or self._video_to_paths.get(base_id) or self._video_to_paths.get(base_id.replace("TSU_", ""))
                if path_list:
                    tmp: List[Tuple[str, int]] = []
                    for pth in path_list:
                        try:
                            fname = os.path.basename(pth)
                            m = re.search(r"(\d+)", fname)
                            if not m:
                                continue
                            idx_int = int(m.group(1))
                            if os.path.isfile(pth):
                                tmp.append((pth, idx_int))
                        except Exception:
                            continue
                    tmp.sort(key=lambda x: x[1])
                    files_with_idx.extend(tmp)
            
            if not files_with_idx and dir_path and os.path.isdir(dir_path):
                try:
                    tmp: List[Tuple[str, int]] = []
                    for nm in os.listdir(dir_path):
                        m = re.search(r"(\d+)", nm)
                        if not m:
                            continue
                        idx_int = int(m.group(1))
                        pth = os.path.join(dir_path, nm)
                        if os.path.isfile(pth):
                            tmp.append((pth, idx_int))
                    tmp.sort(key=lambda x: x[1])
                    files_with_idx.extend(tmp)
                except Exception:
                    pass
            
            # NOTE: translated from Chinese
            for end in range(0, n, window_stride):
                start = max(0, end - window_size + 1)
                window_files = files_with_idx[start : end + 1] if files_with_idx else []
                
                # NOTE: translated from Chinese
                label_value = -1
                if frame_labels and end < len(frame_labels):
                    try:
                        label_value = int(frame_labels[end])
                    except (TypeError, ValueError):
                        label_value = -1
                is_trigger = label_value > 0
                
                # NOTE: translated from Chinese
                step_name = ""
                if is_trigger:
                    step_name = self._vocab_id_to_name.get(label_value, "")
                    if not step_name:
                        if label_value not in self._missing_vocab_labels:
                            self._missing_vocab_labels.add(label_value)
                        step_name = str(label_value)
                
                # NOTE: translated from Chinese
                task_label = -1
                if frame_task_labels and end < len(frame_task_labels):
                    try:
                        task_label = int(frame_task_labels[end])
                    except (TypeError, ValueError):
                        task_label = -1
                task_name_output = ""
                if is_trigger and task_label > 0:
                    task_name_output = self._vocab_id_to_name.get(task_label, "")
                    if not task_name_output:
                        if task_label not in self._missing_vocab_labels:
                            self._missing_vocab_labels.add(task_label)
                        task_name_output = str(task_label)
                    # NOTE: translated from Chinese
                    self.video_task_names.setdefault(vid, task_name_output)
                    if base_id:
                        self.video_task_names.setdefault(base_id, task_name_output)
                else:
                    task_name_output = self.video_task_names.get(vid, "") or self.video_task_names.get(base_id, "")
                    if not task_name_output:
                        task_name_output = self.task_lookup.get(vid, "") or self.task_lookup.get(base_id, "")
                task_name_output = task_name_output.strip() if isinstance(task_name_output, str) else str(task_name_output).strip()

                env_desc = resolve_scenario_text(r, vid, self.video_scenarios)
                trigger_reason = str(r.get("trigger_en", "") or r.get("trigger_reason", "") or "").strip()
                
                # NOTE: translated from Chinese
                meta = {
                    "video_id": vid,
                    "window_files": window_files,
                    "start": start,
                    "end": end,
                    "is_trigger": is_trigger,
                    "step_name": step_name,
                    "task_name": task_name_output,
                    "frame_labels": frame_labels,
                    "scenario": env_desc,
                    "trigger_reason": trigger_reason,
                }
                
                self.samples_meta.append(meta)
                self.targets.append(1 if is_trigger else 0)
                self.gt_tasks.append(task_name_output if is_trigger else "")
                self.gt_steps.append(step_name if is_trigger else "")
                self.gt_future_steps.append([])
                self.gt_next_actions.append("")
        
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"样本元数据构建完成，共 {len(self.samples_meta)} 个样本")
        if len(self.samples_meta) == 0:
            raise RuntimeError(
                f"LazySlidingWindowDataset built 0 samples. "
                f"Check --annotation/--frame_root paths. jsonl_path={jsonl_path}"
            )

    def _lazy_cache_writer_path(self) -> str:
        _, _, _, local_rank = _get_env_silence_and_rank()
        worker_info = torch.utils.data.get_worker_info()
        worker_tag = f"w{worker_info.id}" if worker_info is not None else "wmain"
        return f"{self._lazy_label_cache_prefix}.rank{local_rank}.{worker_tag}.pkl"

    def _load_lazy_label_cache(self) -> None:
        silence, is_main, _, _ = _get_env_silence_and_rank()
        prefix = Path(self._lazy_label_cache_prefix)
        parent = prefix.parent
        stem = prefix.name
        cache_files = sorted(parent.glob(f"{stem}.rank*.w*.pkl"))
        loaded = 0
        for fp in cache_files:
            try:
                with open(fp, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        if isinstance(k, int) and isinstance(v, dict) and k not in self._lazy_label_cache:
                            self._lazy_label_cache[k] = v
                            loaded += 1
            except Exception:
                continue
        if not silence and is_main and loaded > 0:
            print(f"懒加载标签缓存已加载: {loaded} 条 ({len(cache_files)} 个分片)")

    def _flush_lazy_label_cache(self, force: bool = False) -> None:
        if not self._lazy_label_cache_updates:
            return
        if (not force) and len(self._lazy_label_cache_updates) < self._lazy_label_cache_flush_every:
            return
        out_path = self._lazy_cache_writer_path()
        existing: Dict[int, Dict[str, Any]] = {}
        if os.path.exists(out_path):
            try:
                with open(out_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    existing = data
            except Exception:
                existing = {}
        existing.update(self._lazy_label_cache_updates)
        tmp = f"{out_path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(existing, f)
        os.replace(tmp, out_path)
        self._lazy_label_cache_updates.clear()
    
    def __len__(self) -> int:
        return len(self.samples_meta)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """懒加载：在这里才真正加载图像并进行 tokenize"""
        meta = self.samples_meta[idx]
        
        vid = meta["video_id"]
        window_files = meta["window_files"]
        start = meta["start"]
        end = meta["end"]
        is_trigger = meta["is_trigger"]
        step_name = meta["step_name"]
        task_name_output = meta["task_name"]
        frame_labels: List[int] = meta.get("frame_labels", [])
        cached_labels = self._lazy_label_cache.get(idx)
        if cached_labels is None:
            completed_steps: List[str] = collect_completed_steps_before_current_segment(
                frame_labels, end, self._vocab_id_to_name
            )
            future_steps_list: List[str] = collect_future_actions(
                frame_labels=frame_labels,
                current_idx=end,
                predict_steps=self.predict_steps,
                vocab_map=self._vocab_id_to_name,
                missing_vocab_labels=self._missing_vocab_labels,
                video_id=vid,
            )
            next_action = ""
            if is_trigger and self.predict_next_action:
                next_action = teacher_next_action(
                    task_name_output, completed_steps, step_name,
                    future_steps_list, self._annotation_path, self._tg_cache,
                )
                if self._distill_label_map:
                    _cd_distilled = self._distill_label_map.get(idx)
                    if _cd_distilled:
                        next_action = _cd_distilled
            enabled_steps, parallel_steps = derive_enabled_and_parallel_steps(
                task_name=task_name_output,
                completed_steps=completed_steps,
                current_step=step_name,
                human_future_steps=future_steps_list,
                annotation_path=self._annotation_path,
                tg_cache=self._tg_cache,
            )
            window_steps = build_window_steps(frame_labels, start, end, self._vocab_id_to_name)
            prev_step = completed_steps[-1] if completed_steps else None
            evidence_pos = select_evidence_positions(frame_labels, start, end)
            cached_labels = {
                "completed_steps": completed_steps,
                "future_steps": future_steps_list,
                "next_action": next_action,
                "enabled_steps": enabled_steps,
                "parallel_steps": parallel_steps,
                "window_steps": window_steps,
                "prev_step": prev_step,
                "evidence_pos": evidence_pos,
            }
            self._lazy_label_cache[idx] = cached_labels
            self._lazy_label_cache_updates[idx] = cached_labels
            self._flush_lazy_label_cache(force=False)
        completed_steps = cached_labels.get("completed_steps", [])
        future_steps_list = cached_labels.get("future_steps", [])
        next_action = cached_labels.get("next_action", "")
        enabled_steps = cached_labels.get("enabled_steps", [])
        parallel_steps = cached_labels.get("parallel_steps", [])
        window_steps = cached_labels.get("window_steps", [])
        prev_step = cached_labels.get("prev_step", None)
        evidence_pos = cached_labels.get("evidence_pos", [])
        # Keep these labels in sync for eval code paths that read dataset fields directly.
        if idx < len(self.gt_future_steps):
            self.gt_future_steps[idx] = future_steps_list if is_trigger else []
        if idx < len(self.gt_next_actions):
            self.gt_next_actions[idx] = next_action if is_trigger else ""
        
        # NOTE: translated from Chinese
        images: List[Image.Image] = []
        frame_descs: List[str] = []
        if window_files:
            idx0 = window_files[0][1]
            for j, (pth, idx_int) in enumerate(window_files):
                t = (idx_int - idx0) / 25.0
                frame_descs.append(f"F{j} [idx={idx_int} t={t:.2f}s]")
                try:
                    images.append(self._load_image_cached(pth))
                except Exception as e:
                    raise RuntimeError(f"加载帧图像失败: video_id={vid}, path={pth}, error={e}") from e
        else:
            # NOTE: translated from Chinese
            raise RuntimeError(f"未找到任何帧文件: video_id={vid}, window=({start},{end})")
        
        # NOTE: translated from Chinese
        use_labels_only = bool(os.environ.get("LABELS_ONLY", "0") == "1")
        bundle = build_two_stage_conversation_bundle(
            video_id=vid,
            frame_descs=frame_descs,
            images=images,
            completed_steps=completed_steps,
            is_trigger=bool(is_trigger),
            task_name=task_name_output,
            step_name=step_name,
            future_steps=future_steps_list if is_trigger else [],
            next_action=next_action if is_trigger else "",
            predict_steps=self.predict_steps,
            predict_next_action=self.predict_next_action,
            history_mode=self.history_memory_mode,
            history_recent_k=self.history_recent_k,
        )
        dynamic_system_prompt = bundle["system_prompt_text"]
        history_memory_text = bundle["history_memory_text"]
        messages_prompt = bundle["messages_prompt"]
        messages_full = bundle["messages_full"]
        assistant_target = bundle["assistant_target"]
        
        # Tokenize
        text_prompt = self.tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        text_full = self.tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        
        # NOTE: translated from Chinese
        enc_full = self._processor(text=[text_full], images=[images], return_tensors="pt")
        enc_prompt_mm = self._processor(text=[text_prompt], images=[images], return_tensors="pt")
        if "pixel_values" in enc_full:
            enc_full["pixel_values"] = enc_full["pixel_values"].to(torch.bfloat16)
        if "pixel_values" in enc_prompt_mm:
            enc_prompt_mm["pixel_values"] = enc_prompt_mm["pixel_values"].to(torch.bfloat16)
        ids_prompt = enc_prompt_mm["input_ids"]
        ids_full = enc_full["input_ids"]
        
        # NOTE: translated from Chinese
        full_ids = ids_full.squeeze(0)
        prompt_len = int(ids_prompt.shape[1])
        labels = ids_full.clone().detach()
        labels[0, :prompt_len] = -100
        attention_mask = torch.ones_like(ids_full, dtype=torch.long)
        
        # NOTE: translated from Chinese
        RE_L, RE_R = "<|reasoning_start|>", "<|reasoning_end|>"
        IS_L, IS_R = "<|trigger_start|>", "<|trigger_end|>"
        TK_L, TK_R = "<|task_start|>", "<|task_end|>"
        ST_L, ST_R = "<|step_start|>", "<|step_end|>"
        
        is_l_ids = self.tokenizer.encode(IS_L, add_special_tokens=False)
        is_r_ids = self.tokenizer.encode(IS_R, add_special_tokens=False)
        tk_l_ids = self.tokenizer.encode(TK_L, add_special_tokens=False)
        tk_r_ids = self.tokenizer.encode(TK_R, add_special_tokens=False)
        st_l_ids = self.tokenizer.encode(ST_L, add_special_tokens=False)
        st_r_ids = self.tokenizer.encode(ST_R, add_special_tokens=False)
        re_l_ids = self.tokenizer.encode(RE_L, add_special_tokens=False)
        re_r_ids = self.tokenizer.encode(RE_R, add_special_tokens=False)
        scores_ids = self.tokenizer.encode("scores:", add_special_tokens=False)
        fs_l_ids = self.tokenizer.encode("<|future_steps_start|>", add_special_tokens=False)
        fs_r_ids = self.tokenizer.encode("<|future_steps_end|>", add_special_tokens=False)
        NA_L, NA_R = "<|next_action_start|>", "<|next_action_end|>"
        na_l_ids = self.tokenizer.encode(NA_L, add_special_tokens=False)
        na_r_ids = self.tokenizer.encode(NA_R, add_special_tokens=False)
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        full_list = full_ids.tolist()
        
        def find(a, pat, start=0):
            return _find_subseq(a, pat, start=start)
        
        def supervise_span(start_idx, end_idx_exclusive):
            if start_idx != -1 and end_idx_exclusive != -1 and end_idx_exclusive > start_idx:
                labels[0, start_idx:end_idx_exclusive] = ids_full[0, start_idx:end_idx_exclusive]
        
        # NOTE: translated from Chinese
        if self.use_reasoning_tokens:
            s_re_l = find(full_list, re_l_ids, start=prompt_len)
            s_re_r = find(full_list, re_r_ids, start=max(prompt_len, s_re_l if s_re_l!=-1 else prompt_len))
            if s_re_l != -1 and s_re_r != -1 and s_re_r > s_re_l:
                supervise_span(s_re_l, s_re_r + len(re_r_ids))
        
        # NOTE: translated from Chinese
        s_is = find(full_list, is_l_ids, start=prompt_len)
        e_is = find(full_list, is_r_ids, start=prompt_len)
        if s_is != -1 and e_is != -1:
            supervise_span(s_is, e_is + len(is_r_ids))
        
        # NOTE: translated from Chinese
        s_tk = find(full_list, tk_l_ids, start=prompt_len)
        e_tk = find(full_list, tk_r_ids, start=prompt_len)
        if s_tk != -1 and e_tk != -1:
            supervise_span(s_tk, e_tk + len(tk_r_ids))
        
        # NOTE: translated from Chinese
        s_st = find(full_list, st_l_ids, start=prompt_len)
        e_st = find(full_list, st_r_ids, start=prompt_len)
        if s_st != -1 and e_st != -1:
            supervise_span(s_st, e_st + len(st_r_ids))
        
        # NOTE: translated from Chinese
        s_sc = find(full_list, scores_ids, start=prompt_len) if is_trigger else -1
        if s_sc != -1:
            nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            e_sc = -1
            if len(nl_ids) == 1:
                nl_id = nl_ids[0]
                for k in range(s_sc+1, len(full_list)):
                    if full_list[k] == nl_id:
                        e_sc = k
                        break
            if e_sc == -1:
                if eos_id in full_list:
                    e_sc = full_list.index(eos_id)
                else:
                    e_sc = len(full_list)
            supervise_span(s_sc, e_sc)
        
        # NOTE: translated from Chinese (config)
        if self.predict_steps > 0 and future_steps_list:
            s_fs = find(full_list, fs_l_ids, start=prompt_len)
            e_fs = find(full_list, fs_r_ids, start=prompt_len)
            if s_fs != -1 and e_fs != -1:
                supervise_span(s_fs, e_fs + len(fs_r_ids))
        if is_trigger and self.predict_next_action:
            s_na = find(full_list, na_l_ids, start=prompt_len)
            e_na = find(full_list, na_r_ids, start=prompt_len)
            if s_na != -1 and e_na != -1:
                supervise_span(s_na, e_na + len(na_r_ids))
        
        # NOTE: translated from Chinese
        is_region = [s_is, e_is + len(is_r_ids)] if s_is != -1 and e_is != -1 else [-1, -1]
        task_region = [s_tk, e_tk + len(tk_r_ids)] if s_tk != -1 and e_tk != -1 else [-1, -1]
        step_region = [s_st, e_st + len(st_r_ids)] if s_st != -1 and e_st != -1 else [-1, -1]
        trigger_token_pos = -1
        if s_is != -1:
            trigger_token_pos = s_is + len(is_l_ids)
        
        # NOTE: translated from Chinese
        item = {
            "input_ids": ids_full.squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "prompt_input_ids": ids_prompt.squeeze(0),
            "prompt_attention_mask": torch.ones_like(ids_prompt.squeeze(0), dtype=torch.long),
            "debug_prompt_text": text_prompt,
            "debug_target_text": assistant_target,
            "is_region": is_region,
            "task_region": task_region,
            "step_region": step_region,
            "trigger_label": torch.tensor(int(is_trigger), dtype=torch.long),
            "trigger_position": torch.tensor(
                int(trigger_token_pos if trigger_token_pos >= 0 else max(prompt_len, 0)),
                dtype=torch.long,
            ),
            "future_steps": future_steps_list,
            "next_action": next_action if is_trigger else "",
            "ts_two_stage": True,
            "ts_system_prompt_text": dynamic_system_prompt,
            "ts_state_user_text": bundle["state_user_text"],
            "ts_decision_user_text": bundle["decision_user_text"],
            "ts_window_files": [pth for (pth, _) in window_files],
            "ts_max_image_long_edge": int(getattr(self, "_max_image_long_edge", 0) or 0),
        }
        
        if "pixel_values" in enc_full:
            item["pixel_values"] = enc_full["pixel_values"].squeeze(0)
        if "image_grid_thw" in enc_full:
            grid = enc_full["image_grid_thw"]
            if isinstance(grid, torch.Tensor):
                if grid.dim() >= 3:
                    grid = grid.squeeze(0)
                grid = grid.view(-1, 3)
            item["image_grid_thw"] = grid
        
        if "pixel_values" in enc_prompt_mm:
            item["prompt_pixel_values"] = enc_prompt_mm["pixel_values"].squeeze(0)
        if "image_grid_thw" in enc_prompt_mm:
            pgrid = enc_prompt_mm["image_grid_thw"]
            if isinstance(pgrid, torch.Tensor):
                if pgrid.dim() >= 3:
                    pgrid = pgrid.squeeze(0)
                pgrid = pgrid.view(-1, 3)
            item["prompt_image_grid_thw"] = pgrid
        
        return item

    def _load_image_cached(self, path: str) -> Image.Image:
        cached = self._frame_cache.get(path)
        if cached is not None:
            self._frame_cache.move_to_end(path)
            return cached.copy()
        with Image.open(path) as im:
            img = im.convert("RGB").copy()
        mle = getattr(self, "_max_image_long_edge", 0)
        if isinstance(mle, int) and mle > 0:
            w, h = img.size
            m = max(w, h)
            if m > mle:
                scale = float(mle) / float(m)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), Image.LANCZOS)
        self._frame_cache[path] = img
        if len(self._frame_cache) > self._frame_cache_max:
            self._frame_cache.popitem(last=False)
        return img.copy()


def make_collate_fn(pad_token_id: int):
    """
    创建数据批处理函数，用于将变长序列对齐到相同长度
    这对监督学习loss计算至关重要，因为CrossEntropyLoss需要固定长度的输入
    """
    SKIP_TO_GPU = {
        "prompt_pixel_values",
        "prompt_image_grid_thw",
        "prompt_input_ids",
        "prompt_attention_mask",
        "debug_prompt_text",
        "debug_target_text",
    }

    def _pad_1d(x: torch.Tensor, length: int, pad: int) -> torch.Tensor:
        """将1D张量填充到指定长度"""
        if x.size(0) >= length:
            return x
        return torch.cat([x, x.new_full((length - x.size(0),), pad)], dim=0)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批处理函数：将多个样本组合成一个batch
        关键：确保所有序列长度一致，这对loss计算很重要
        """
        # NOTE: translated from Chinese
        max_len = 0
        for b in batch:
            if isinstance(b.get("input_ids"), torch.Tensor):
                max_len = max(max_len, int(b["input_ids"].size(0)))
        
        out: Dict[str, Any] = {}
        for k in batch[0].keys():
            if k in SKIP_TO_GPU:
                continue  # NOTE: translated from Chinese (debug)
            # NOTE: translated from Chinese
            if k in ("input_ids", "attention_mask", "labels"):
                arr = []
                for b in batch:
                    t = b[k]
                    if k == "input_ids":
                        # NOTE: translated from Chinese
                        arr.append(_pad_1d(t, max_len, pad_token_id))
                    elif k == "attention_mask":
                        # NOTE: translated from Chinese
                        arr.append(_pad_1d(t, max_len, 0))
                    else:  # labels
                        # NOTE: translated from Chinese
                        arr.append(_pad_1d(t, max_len, -100))
                out[k] = torch.stack(arr, dim=0)
            elif k in ("prompt_input_ids", "prompt_attention_mask"):
                # NOTE: translated from Chinese
                pmax = 0
                for b in batch:
                    if isinstance(b.get(k), torch.Tensor):
                        pmax = max(pmax, int(b[k].size(0)))
                parr = []
                for b in batch:
                    t = b[k]
                    if k == "prompt_input_ids":
                        parr.append(_pad_1d(t, pmax, pad_token_id))
                    else:
                        parr.append(_pad_1d(t, pmax, 0))
                out[k] = torch.stack(parr, dim=0)
            elif isinstance(batch[0][k], torch.Tensor):
                # NOTE: translated from Chinese
                # NOTE: translated from Chinese
                # NOTE: translated from Chinese
                if k in ("pixel_values", "image_grid_thw"):
                    if len(batch) == 1:
                        out[k] = batch[0][k]
                    else:
                        out[k] = torch.cat([b[k] for b in batch], dim=0)
                else:
                    out[k] = torch.stack([b[k] for b in batch], dim=0)
                if k == "pixel_values":
                    out[k] = out[k].to(dtype=torch.bfloat16, copy=False)
            elif k in ("trigger_label", "trigger_position"):
                out[k] = torch.stack([b[k] if isinstance(b[k], torch.Tensor) else torch.tensor(b[k], dtype=torch.long) for b in batch], dim=0)
            else:
                out[k] = [b[k] for b in batch]
        return out

    return collate_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="<REDACTED_PATH>",
    )
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--frame_root", type=str, required=False, default="")
    parser.add_argument("--preprocessed_data_dir", type=str, required=False, default="", 
                        help="预处理数据保存目录，如果提供则尝试加载预处理数据")
    parser.add_argument("--preprocessed_data_file", type=str, required=False, default="", 
                        help="指定预处理数据文件名，如果提供则直接使用该文件")
    parser.add_argument("--preprocessed_train_file", type=str, required=False, default="", 
                        help="指定训练集预处理数据文件")
    parser.add_argument("--preprocessed_val_file", type=str, required=False, default="", 
                        help="指定验证集预处理数据文件")
    parser.add_argument("--keyframes_map", type=str, required=False, default="<REPO_ROOT>/datasets/task_video_mapping_final_complete.json")
    parser.add_argument("--annotation", type=str, required=False,
                        default="<REDACTED_PATH>",
                        help="annotation 文件路径，将从其中的 vocabulary 字段解析 step 名称")
    parser.add_argument("--priority_scores", type=str, required=False, default="<REDACTED_PATH>",
                        help="根据任务名称查找紧急度/价值评分的 JSON 文件")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--use_trigger_hints", action="store_true")
    parser.add_argument(
        "--trigger_json",
        type=str,
        default="<REPO_ROOT>/old/4_keysteps_tsu_all_v2_annotations_only_filtered_25fps_with_zh_with_triggers_with_zh_social_updated.json",
        help="可选的大型标注文件路径，若提供且启用 --use_trigger_hints，将从其中读取 trigger_en",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=32)
    # NOTE: argparse type=bool is unsafe (bool("False")==True). Use flags instead.
    parser.add_argument("--bf16", dest="bf16", action="store_true", help="Enable bf16 mixed precision.")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false", help="Disable bf16 mixed precision.")
    parser.set_defaults(bf16=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="<REPO_ROOT>/save/l2_sft",
    )
    parser.add_argument(
    "--ddp_timeout",
    type=int,
    default=14400,  # 4 hours，按你最长 eval 可能等待的时间设
    help="Timeout (seconds) for torch.distributed.init_process_group / collectives."
    )
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--record_prompts", action="store_true")
    parser.add_argument("--if_score", action="store_true", help="若设置，则标签与输出包含 scores 字段；否则仅输出 is_trigger")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_generation_batch_size", type=int, default=1,
                        help="生成评估时的batch大小（仅影响逐样本generate阶段）")
    parser.add_argument("--debug", action="store_true", help="输出详细调试信息到控制台与记录文件")
    parser.add_argument("--debug_samples", type=int, default=5, help="每个阶段最多打印的调试样本数")
    parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="每隔多少个 epoch 执行一次评估（1 表示每个 epoch 都评估）")
    parser.add_argument("--eval_round_k", type=int, default=0,
                        help=">0 时启用轮次对齐评估：每个 rank 每轮只评估 k 个样本，然后全 rank 同步")
    parser.add_argument("--labels_only", action="store_true", help="只生成标签：[[IS]]…[[/IS]]，若触发则追加[[TASK]]…[[/TASK]]；不输出 reasoning 等自由字段")
    parser.add_argument("--eval_only", action="store_true", help="仅执行评估，跳过训练阶段")
    parser.add_argument("--train_jump", action="store_true",
                        help="跳跃式训练：根据 run_dir 里已存在的 epoch_N checkpoint 自动跳过对应 epoch 的训练，并在评测点加载对应 checkpoint 做评估")
    parser.add_argument("--checkpoint_path", type=str, default="", help="评估时加载的 checkpoint 路径（仅在 --eval_only 时使用）")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="从指定 checkpoint 继续训练（恢复优化器状态、学习率调度器等）")
    parser.add_argument("--load_from_checkpoint", type=str, default="", help="从指定 checkpoint 加载模型权重（不恢复训练状态，从头开始训练）")
    parser.add_argument("--min_reason_tokens", type=int, default=50,
                        help="在生成 [[IS]] / [[TASK]] 前，至少先写这么多新 token（用于硬性拉长 reasoning）")
    parser.add_argument("--predict_next_action", action="store_true", help="Supervise next action tag.")
    parser.add_argument("--predict_steps", type=int, default=0,
                        help="预测未来 step 的数量，0 表示不预测，>0 时启用")
    parser.add_argument(
        "--onestep_workers",
        type=int,
        default=16,
        help="one-step 评估并发分片数（默认16）。",
    )
    parser.add_argument("--belief_state_eval", action="store_true",
                        help="启用基于 task graph 的 belief-state 后处理，用于稳定 task/step/future 预测")
    parser.add_argument("--belief_task_history_window", type=int, default=6,
                        help="belief-state 中 task 历史投票窗口大小")
    parser.add_argument("--belief_step_history_window", type=int, default=4,
                        help="belief-state 中 step 历史投票窗口大小")
    parser.add_argument("--belief_future_topk", type=int, default=5,
                        help="belief-state 过滤后保留的 future steps 上限")
    parser.add_argument("--history_memory_mode", type=str, default="none",
                        choices=["none", "oracle_past", "recent_only", "set_only"],
                        help="历史动作记忆模式：none / oracle_past / recent_only / set_only")
    parser.add_argument("--history_recent_k", type=int, default=4,
                        help="当启用历史动作记忆时，保留最近多少个已完成动作")
    parser.add_argument("--experiment_tag", type=str, default="",
                        help="实验标签，会写入 run name 方便区分不同时期实验")
    parser.add_argument("--limit_embedding_training", action="store_true",
                        help="仅训练特定token（special/label/end）的embedding/lm_head行；默认训练全部token")
    parser.add_argument("--reasoning", action="store_true",
                        help="启用reasoning功能：在输出前添加<|reasoning_start|>...<|reasoning_end|>外壳")
    
    # NOTE: translated from Chinese
    parser.add_argument("--evidence_frames", action="store_true", help="在输出中包含 evidence_frames 字段")
    parser.add_argument("--confidence", action="store_true", help="在输出中包含 confidence 字段")
    parser.add_argument("--scores", action="store_true", help="在输出中包含 scores 字段")
    parser.add_argument("--scores_max_tokens", type=int, default=24,
                        help="从出现 'scores:' 起最多允许继续生成的 token 数（到达后强烈偏置 <|im_end|>）")
    
    # NOTE: translated from Chinese
    parser.add_argument("--use_margin_loss", action="store_true", 
                        help="启用边际损失来提高正负样本的分离度")
    parser.add_argument("--margin_value", type=float, default=0.5,
                        help="边际值，控制正负样本之间的最小距离")
    parser.add_argument("--margin_weight", type=float, default=0.1,
                        help="边际损失的权重，控制边际损失在总损失中的比重")
    parser.add_argument("--pos_weight", type=float, default=2.0,
                        help="正样本的权重（默认更高，因为我们要提高正例准确率）")
    parser.add_argument("--neg_weight", type=float, default=1.0,
                        help="负样本的权重")
    
    # NOTE: translated from Chinese (binding)
    parser.add_argument("--bind_trigger_task", action="store_true",
                        help="启用 trigger-task 多模态绑定 loss")
    parser.add_argument("--bind_task_step", action="store_true",
                        help="启用 task-step 多模态绑定 loss")
    parser.add_argument("--bind_loss_weight", type=float, default=0.1,
                        help="层级绑定 loss 的整体权重（若单独未指定则同时用于 trig2task 与 task2step）")
    parser.add_argument("--bind_tt_weight", type=float, default=None,
                        help="trigger-task 绑定 loss 权重（默认沿用 bind_loss_weight）")
    parser.add_argument("--bind_ts_weight", type=float, default=0.5,
                        help="task-step 绑定 loss 权重（默认 0.5）")
    parser.add_argument("--bind_trigger_disc_weight", type=float, default=0.1,
                        help="trigger 判别性损失权重（默认 0.1），用于保持正负样本 trigger 向量的分离度")
    
    # NOTE: translated from Chinese
    parser.add_argument("--enable_task_step_constraint", action="store_true",
                        help="启用 task-step 约束（强制学习合法的 task-step 组合）")
    parser.add_argument("--task_step_constraint_weight", type=float, default=0.5,
                        help="Task-Step 约束 loss 的权重，默认 0.5")
    
    # NOTE: translated from Chinese
    parser.add_argument("--lazy_loading", action="store_true",
                        help="启用懒加载模式：__init__时只记录元数据，__getitem__时才加载图像和tokenize")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="DataLoader的num_workers数量，建议懒加载模式下设置为4或更高")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true",
                        help="禁用gradient checkpointing以提升速度（可能增加显存占用）")
    parser.add_argument("--trigger_loss_weight", type=float, default=1.0,
                        help="触发 true/false 分类额外损失的权重，默认 1.0，可设为 0 关闭")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--disable_find_unused_parameters", action="store_true",
                        help="多卡训练时禁用 find_unused_parameters（默认多卡自动开启以避免梯度同步错误）")
    parser.add_argument("--max_image_long_edge", type=int, default=896,
                        help="图像最长边缩放到该值（保持长宽比，为0表示不缩放）")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="最多保留的 checkpoint 数量（默认3，设为2可只保留最新和最好的）")
    parser.add_argument("--save_only_model", action="store_true",
                        help="只保存模型权重（不保存优化器状态），可大幅减少 checkpoint 大小")
    parser.add_argument("--class_weighted", action="store_true",
                        help="启用类别加权：根据task和step的训练数据规模对CE损失进行加权，处理long-tail问题")
    parser.add_argument("--silence", action="store_true",
                        help="静默模式：数据加载和统计阶段不打印任何日志，只保留一个进程的进度条")
    
    args = parser.parse_args()

    # NOTE: translated from Chinese
    if not args.reasoning:
        args.min_reason_tokens = 0

    if args.eval_only and args.train_jump:
        raise ValueError("--eval_only 与 --train_jump 不能同时使用。")
    if args.train_jump and not args.resume_from_checkpoint:
        raise ValueError("--train_jump 需要同时提供 --resume_from_checkpoint（例如 run_dir/epoch_1）。")

    # NOTE: translated from Chinese (stats)
    if args.silence:
        os.environ["L2_SILENCE"] = "1"
    else:
        os.environ.pop("L2_SILENCE", None)

    # NOTE: translated from Chinese
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir = os.path.abspath(args.output_dir)
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = os.path.abspath(args.resume_from_checkpoint)

    run_name = build_run_name_from_args(args)
    marker_path = os.path.join(args.output_dir, f".{run_name}_run_dir")
    launch_start_ts = time.time()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    silence, is_main, _, _ = _get_env_silence_and_rank()

    def _write_marker_atomic(path: str, value: str) -> None:
        """Best-effort atomic marker write to avoid empty/partial reads."""
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as mf:
            mf.write(value)
            mf.flush()
            try:
                os.fsync(mf.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)

    # NOTE: translated from Chinese
    resume_run_dir = ""
    if args.resume_from_checkpoint:
        # Always reuse the checkpoint's run directory when resuming so every rank
        # writes eval shards and checkpoints back into the same run.
        resume_run_dir = os.path.abspath(os.path.dirname(args.resume_from_checkpoint))

    if dist.is_available() and dist.is_initialized():
        run_dir_obj = [None]
        if local_rank == 0:
            if resume_run_dir:
                run_dir_obj[0] = resume_run_dir
            else:
                run_dir_obj[0] = ensure_unique_run_dir(args.output_dir, run_name)
            os.makedirs(run_dir_obj[0], exist_ok=True)
        dist.broadcast_object_list(run_dir_obj, src=0)
        run_dir = run_dir_obj[0]
        # NOTE: translated from Chinese
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception as e:
            if not silence and is_main:
                print(f"[WARN] 创建 run_dir 失败: {run_dir}, error={e}")
        _safe_dist_barrier("run_dir_ready")
        if local_rank == 0:
            # NOTE: translated from Chinese
            try:
                _write_marker_atomic(marker_path, run_dir)
            except Exception:
                pass
        if local_rank == 0 and not silence:
            try:
                ws = dist.get_world_size()
            except Exception:
                ws = -1
            print(f"[RunDir] distributed run_dir={run_dir} world_size={ws}")
    else:
        # NOTE: translated from Chinese
        if local_rank == 0:
            if resume_run_dir:
                run_dir = resume_run_dir
            else:
                run_dir = ensure_unique_run_dir(args.output_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            try:
                # Prevent non-zero ranks from accidentally reading a stale marker
                # left by a previous interrupted run with the same run_name.
                if os.path.exists(marker_path):
                    os.remove(marker_path)
            except Exception:
                pass
            try:
                _write_marker_atomic(marker_path, run_dir)
            except Exception:
                pass
            if not silence:
                print(f"运行目录: {run_dir}")
        else:
            wait_time = 0.0
            run_dir = ""
            while True:
                if os.path.exists(marker_path):
                    try:
                        marker_mtime = os.path.getmtime(marker_path)
                    except Exception:
                        marker_mtime = 0.0
                    # Ignore stale markers from previous interrupted launches.
                    # Non-zero ranks can otherwise attach to an older run_dir
                    # before rank0 refreshes the marker for the current launch.
                    if marker_mtime + 5.0 >= launch_start_ts:
                        try:
                            with open(marker_path, 'r', encoding='utf-8') as mf:
                                run_dir = mf.read().strip()
                            # Guard against reading a marker written by another run
                            # that happens to share the same output root.
                            if run_dir:
                                base_name = os.path.basename(run_dir.rstrip('/'))
                                if not (
                                    base_name == run_name
                                    or base_name.startswith(f'{run_name}_v')
                                ):
                                    run_dir = ''
                            else:
                                run_dir = ''
                        except Exception:
                            run_dir = ''
                    else:
                        run_dir = ''
                    if run_dir:
                        break
                time.sleep(0.1)
                wait_time += 0.1
                if wait_time > 30:
                    raise RuntimeError(
                        f"等待主进程创建运行目录超时或 marker 为空: {marker_path}"
                    )
            if not run_dir:
                raise RuntimeError(f"读取到空 run_dir marker: {marker_path}")
        os.makedirs(run_dir, exist_ok=True)
        if local_rank == 0 and not silence:
            print(f"[RunDir] standalone run_dir={run_dir}")

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", "")).lower()
    is_qwen3_vl = "qwen3_vl" in model_type
    is_qwen25_vl = "qwen2_5_vl" in model_type or "qwen2.5" in model_type or "qwen2_5" in model_type
    if is_qwen25_vl:
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif is_qwen3_vl:
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError(
                "当前 transformers 版本缺少 Qwen3VLForConditionalGeneration，无法加载 Qwen3-VL。"
            )
        model_cls = Qwen3VLForConditionalGeneration
    else:
        # Fallback: try a multimodal auto model first, then causal LM
        model_cls = AutoModelForVision2Seq

    quant_cfg = getattr(config, "quantization_config", None)
    if isinstance(quant_cfg, dict):
        is_fp8_quant = str(quant_cfg.get("quant_method", "")).lower() == "fp8"
    else:
        quant_cfg_name = type(quant_cfg).__name__.lower() if quant_cfg is not None else ""
        is_fp8_quant = bool(quant_cfg is not None and "fp8" in quant_cfg_name)
    use_bnb = bool(args.load_in_4bit and quant_cfg is None)
    if args.load_in_4bit and quant_cfg is not None and not silence:
        print(
            "[Quant] 检测到模型自带量化配置，禁用 BitsAndBytes 以避免冲突: "
            f"{type(quant_cfg).__name__}"
        )

    bnb_config = None
    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Prefer fast tokenizer when available; fallback to slow if fast init fails.
    tokenizer_file = os.path.join(args.model_name, "tokenizer.json")
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=True,
            tokenizer_file=tokenizer_file if os.path.exists(tokenizer_file) else None,
        )
    except Exception as exc:
        print(f"[Tokenizer] Fast tokenizer init failed, fallback to slow: {exc}")
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=False,
        )
    tokenizer = processor.tokenizer
    
    # NOTE: translated from Chinese
    special_tokens = [
        "<|trigger_start|>",
        "<|trigger_end|>", 
        "<|task_start|>",
        "<|task_end|>",
        "<|step_start|>",
        "<|step_end|>",
    ]
    
    if args.predict_steps and args.predict_steps > 0:
        special_tokens.extend([
            "<|future_steps_start|>",
            "<|future_steps_end|>",
        ])
    if getattr(args, "predict_next_action", False):
        special_tokens.extend([
            "<|next_action_start|>",
            "<|next_action_end|>",
        ])
    
    # NOTE: translated from Chinese
    if args.reasoning:
        special_tokens.extend([
            "<|reasoning_start|>",
            "<|reasoning_end|>"
        ])
        if not silence and is_main:
            print("已启用reasoning功能，添加reasoning special tokens")
    
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if not silence and is_main:
        print(f"已添加special tokens: {special_tokens}")
    
    # NOTE: translated from Chinese
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not silence and is_main:
            print(f"  {token} -> id: {token_id}, encoded: {encoded}")
        if token_id == tokenizer.unk_token_id:
            print(f"  警告: {token} 被识别为UNK token!")
        if len(encoded) != 1:
            print(f"  警告: {token} 编码为多个token: {encoded}")
        # NOTE: translated from Chinese
        assert len(encoded) == 1, f"{token} 不是单token，请检查是否真的被加入词表"


    # NOTE: translated from Chinese
    record_dir = os.path.join(run_dir, "record") if args.record_prompts or args.debug else None
    record_train = os.path.join(record_dir, "train.jsonl") if record_dir else None
    record_val = os.path.join(record_dir, "val.jsonl") if record_dir else None

    # NOTE: translated from Chinese
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        local_rank = -1
        world_size = 1
    # NOTE: translated from Chinese
    try:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            os.environ["WORLD_SIZE"] = str(world_size)
    except Exception:
        pass
    
    ddp_find_unused_parameters = world_size > 1 and not getattr(args, "disable_find_unused_parameters", False)
    if world_size > 1:
        state = "开启" if ddp_find_unused_parameters else "关闭"
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"分布式训练：world_size={world_size}，自动{state} find_unused_parameters")

    # 先加载模型，再加载数据集，减少“先扫数据很久才发现模型问题”的等待成本。
    from_pretrained_kwargs = dict(
        torch_dtype=torch.bfloat16,
    )
    if bnb_config is not None:
        from_pretrained_kwargs["quantization_config"] = bnb_config
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        local_rank = -1
    if local_rank >= 0 and torch.cuda.is_available():
        # In DDP + 4bit/8bit, each rank must load the quantized model on its own device.
        # Otherwise accelerate.prepare() raises device mismatch for non-zero ranks.
        from_pretrained_kwargs["device_map"] = {"": local_rank}
    elif local_rank == -1:
        # For some multimodal models, sharding across GPUs (device_map="auto") can
        # cause device-mismatch errors inside the vision tower. Default to single-GPU
        # placement when possible; allow auto-sharding only when explicitly desired.
        smoke_enabled_env = bool(int(os.environ.get("SMOKE_TEST_SAMPLES", "0") or "0"))
        force_single = smoke_enabled_env or (model_type == "qwen3_vl")
        if force_single and torch.cuda.is_available():
            from_pretrained_kwargs["device_map"] = {"": 0}
        else:
            from_pretrained_kwargs["device_map"] = "auto"

    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main:
        print(f"开始加载模型（先于数据集加载）... device_map={from_pretrained_kwargs.get('device_map', None)}")
    try:
        model = model_cls.from_pretrained(
            args.model_name,
            **from_pretrained_kwargs,
        )
    except ValueError as e:
        # Some configs are not supported by a chosen AutoModel class; try a fallback.
        if model_cls is AutoModelForVision2Seq:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, **from_pretrained_kwargs)
        else:
            raise
    if use_bnb:
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except TypeError:
            model = prepare_model_for_kbit_training(model)
    
    # NOTE: translated from Chinese
    DatasetClass = LazySlidingWindowDataset if args.lazy_loading else SlidingWindowDataset
    loading_mode = "懒加载" if args.lazy_loading else "预处理"
    silence, is_main, _, _ = _get_env_silence_and_rank()
    train_dataset = None
    if not args.eval_only:
        if not silence:
            # NOTE: translated from Chinese
            print(f"进程 {local_rank}/{world_size} 正在加载训练数据集（{loading_mode}模式）...")
        train_dataset = DatasetClass(
            jsonl_path=args.train_json,
            window_size=args.window_size,
            window_stride=args.window_stride,
            use_trigger_hints=args.use_trigger_hints,
            trigger_json_path=args.trigger_json if args.use_trigger_hints else None,
            tokenizer=tokenizer,
            if_score=args.if_score,
            frame_root=args.frame_root if args.frame_root else None,
            annotation_path=args.annotation if getattr(args, "annotation", "") else None,
            processor=processor,
            record_path=record_train,
            preprocessed_data_dir=args.preprocessed_data_dir if args.preprocessed_data_dir else None,
            preprocessed_data_file=args.preprocessed_data_file if args.preprocessed_data_file else None,
            preprocessed_train_file=args.preprocessed_train_file if args.preprocessed_train_file else None,
            preprocessed_val_file=args.preprocessed_val_file if args.preprocessed_val_file else None,
            priority_score_path=args.priority_scores if getattr(args, "priority_scores", "") else None,
            # NOTE: translated from Chinese
            enable_evidence_frames=args.evidence_frames,
            enable_reasoning=args.reasoning,
            enable_confidence=args.confidence,
            enable_scores=args.scores,
            use_reasoning_tokens=args.reasoning,
            random_seed=args.seed,
            max_image_long_edge=args.max_image_long_edge,
            predict_steps=args.predict_steps,
            predict_next_action=getattr(args, "predict_next_action", False),
            history_memory_mode=getattr(args, "history_memory_mode", "none"),
            history_recent_k=getattr(args, "history_recent_k", 4),
        )
        if not silence:
            print(f"进程 {local_rank} 训练数据集加载完成（{loading_mode}模式），共 {len(train_dataset)} 个样本")
    else:
        if not silence and is_main:
            print("eval_only 模式：跳过训练数据集加载。")
    
    if not silence:
        print(f"进程 {local_rank}/{world_size} 正在加载验证数据集（{loading_mode}模式）...")
    # NOTE: translated from Chinese
    eval_dataset = DatasetClass(
        jsonl_path=args.val_json,
        window_size=args.window_size,
        window_stride=args.window_stride,
        use_trigger_hints=args.use_trigger_hints,
        trigger_json_path=args.trigger_json if args.use_trigger_hints else None,
        tokenizer=tokenizer,
        if_score=args.if_score,
        frame_root=args.frame_root if args.frame_root else None,
        annotation_path=args.annotation if getattr(args, "annotation", "") else None,
        processor=processor,
        record_path=record_val,
        preprocessed_data_dir=args.preprocessed_data_dir if args.preprocessed_data_dir else None,
        preprocessed_data_file=args.preprocessed_data_file if args.preprocessed_data_file else None,
        preprocessed_train_file=args.preprocessed_train_file if args.preprocessed_train_file else None,
        preprocessed_val_file=args.preprocessed_val_file if args.preprocessed_val_file else None,
        priority_score_path=args.priority_scores if getattr(args, "priority_scores", "") else None,
        # NOTE: translated from Chinese
        enable_evidence_frames=args.evidence_frames,
        enable_reasoning=args.reasoning,
        enable_confidence=args.confidence,
        enable_scores=args.scores,
        use_reasoning_tokens=args.reasoning,
        random_seed=args.seed,
        max_image_long_edge=args.max_image_long_edge,
        predict_steps=args.predict_steps,
        predict_next_action=getattr(args, "predict_next_action", False),
        history_memory_mode=getattr(args, "history_memory_mode", "none"),
        history_recent_k=getattr(args, "history_recent_k", 4),
    )
    if not silence:
        print(f"进程 {local_rank} 验证数据集加载完成，共 {len(eval_dataset)} 个样本")
    # NOTE: translated from Chinese

    def _print_debug_gt_samples(ds_name: str, ds_obj: Any, max_n: int = 3) -> None:
        if not getattr(args, "debug", False):
            return
        if local_rank != 0:
            return
        try:
            n_total = len(ds_obj)
        except Exception:
            n_total = 0
        n_show = min(int(max_n), int(n_total))
        print(f"[DEBUG] {ds_name} first {n_show} GT labels:")
        for i in range(n_show):
            try:
                gt_trigger = (
                    int(ds_obj.targets[i])
                    if hasattr(ds_obj, "targets") and i < len(ds_obj.targets)
                    else -1
                )
                gt_task = (
                    str(ds_obj.gt_tasks[i])
                    if hasattr(ds_obj, "gt_tasks") and i < len(ds_obj.gt_tasks)
                    else ""
                )
                gt_step = (
                    str(ds_obj.gt_steps[i])
                    if hasattr(ds_obj, "gt_steps") and i < len(ds_obj.gt_steps)
                    else ""
                )
                gt_future = (
                    ds_obj.gt_future_steps[i]
                    if hasattr(ds_obj, "gt_future_steps") and i < len(ds_obj.gt_future_steps)
                    else []
                )
                gt_next_action = (
                    str(ds_obj.gt_next_actions[i])
                    if hasattr(ds_obj, "gt_next_actions") and i < len(ds_obj.gt_next_actions)
                    else ""
                )
                print(
                    f"  - idx={i} trigger={gt_trigger} "
                    f"task='{gt_task}' step='{gt_step}' "
                    f"future_steps={gt_future} next_action='{gt_next_action}'"
                )
                # Print the exact supervised target text (contains reasoning + special tokens).
                try:
                    ex = ds_obj[i]
                    target_text = str(ex.get("debug_target_text", "") or "")
                except Exception:
                    target_text = ""
                if target_text:
                    print("    supervised_target_text:")
                    for ln in target_text.splitlines():
                        print(f"      {ln}")
                else:
                    print("    supervised_target_text: <unavailable>")
            except Exception as e:
                print(f"  - idx={i} <failed to read GT label: {e}>")

    if getattr(args, "debug", False) and train_dataset is not None:
        _print_debug_gt_samples("train", train_dataset, max_n=3)
    if getattr(args, "debug", False):
        _print_debug_gt_samples("val", eval_dataset, max_n=3)

    # === Smoke test mode: run a tiny subset to validate end-to-end training ===
    # Enable by: export SMOKE_TEST_SAMPLES=10
    try:
        smoke_n = int(os.environ.get("SMOKE_TEST_SAMPLES", "0") or "0")
    except Exception:
        smoke_n = 0
    smoke_enabled = bool(smoke_n and smoke_n > 0)
    if smoke_enabled and train_dataset is not None:
        try:
            from torch.utils.data import Subset

            n_train = min(int(smoke_n), len(train_dataset))
            n_eval = min(int(smoke_n), len(eval_dataset))
            train_dataset = Subset(train_dataset, list(range(n_train)))
            eval_dataset = Subset(eval_dataset, list(range(n_eval)))
            # Keep epochs minimal for smoke test; max_steps will also be set later.
            args.num_train_epochs = 1
            args.eval_every_n_epochs = 1
            if not silence and is_main:
                print(
                    f"[SmokeTest] Enabled: SMOKE_TEST_SAMPLES={smoke_n}, "
                    f"train={n_train}, eval={n_eval}, epochs=1"
                )
        except Exception as e:
            smoke_enabled = False
            if not silence and is_main:
                print(f"[SmokeTest] Failed to enable smoke test: {e}")

    # NOTE: FP8 base model may propagate float8 activations into LoRA path; disable LoRA dropout to
    # avoid fused_dropout(float8) unsupported error.
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.0 if ("is_fp8_quant" in locals() and is_fp8_quant) else 0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],  # NOTE: translated from Chinese
    )

    use_gradient_checkpointing = not getattr(args, "disable_gradient_checkpointing", False)
    gc_kwargs = {"use_reentrant": False} if use_gradient_checkpointing else None

    _train_cfg_kwargs = dict(
        output_dir=run_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        ddp_timeout=args.ddp_timeout,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        optim="adamw_torch",
        packing=False,
        dataset_num_proc=1,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,  # NOTE: translated from Chinese
        dataloader_pin_memory=True if args.dataloader_num_workers > 0 else False,  # NOTE: translated from Chinese
        seed=args.seed,
    )
    # FP8 base may produce float8 grads in some paths; disable grad clipping to avoid foreach_norm errors.
    if "is_fp8_quant" in locals() and is_fp8_quant:
        _train_cfg_kwargs["max_grad_norm"] = 0.0
    # Respect WANDB disable flags; avoid no-tty login prompts in non-interactive runs.
    try:
        wandb_mode = (os.environ.get("WANDB_MODE", "") or "").strip().lower()
        wandb_disabled = (os.environ.get("WANDB_DISABLED", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        if wandb_mode == "disabled" or wandb_disabled:
            _train_cfg_kwargs["report_to"] = []
    except Exception:
        pass
    if "smoke_enabled" in locals() and smoke_enabled:
        # One optimizer step is enough to verify the pipeline.
        _train_cfg_kwargs["max_steps"] = 1
        _train_cfg_kwargs["save_strategy"] = "no"
        _train_cfg_kwargs["do_eval"] = False
        _train_cfg_kwargs["eval_strategy"] = "no"
        _train_cfg_kwargs["report_to"] = []
    training_config = SFTConfig(**_train_cfg_kwargs)

    # Allow LoRA fine-tuning on FP8-quantized base models by disabling the Trainer's
    # "quantized model not trainable" guard. Base weights remain frozen; only adapters train.
    try:
        qc = getattr(getattr(model, "config", None), "quantization_config", None)
        if isinstance(qc, dict):
            _is_fp8 = str(qc.get("quant_method", "")).lower() == "fp8"
            _qc_name = "dict(fp8)" if _is_fp8 else "dict"
        else:
            _qc_name = type(qc).__name__ if qc is not None else "None"
            _is_fp8 = bool(qc is not None and "fp8" in _qc_name.lower())
        if qc is not None and _is_fp8:
            if not silence and is_main:
                print(f"[Quant] Detected FP8 quantized base ({_qc_name}); disabling hf_quantizer guard for LoRA.")
            if getattr(model, "hf_quantizer", None) is not None:
                try:
                    model.hf_quantizer = None
                except Exception:
                    pass
            for attr in ("is_quantized", "_is_quantized"):
                try:
                    setattr(model, attr, False)
                except Exception:
                    pass
    except Exception:
        pass
    # NOTE: translated from Chinese
    try:
        old_vocab_size = int(model.get_input_embeddings().weight.size(0))
        new_vocab_size = int(len(tokenizer))
        if new_vocab_size > old_vocab_size:
            model.resize_token_embeddings(new_vocab_size)
            print(f"已调整模型词表大小: {old_vocab_size} -> {new_vocab_size}")
        # NOTE: translated from Chinese
        _dbg_tokens = ["<|trigger_start|>", "<|trigger_end|>", "<|task_start|>", "<|task_end|>"]
        _dbg_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in _dbg_tokens}
        print(f"Special tokens -> ids: {_dbg_ids}")
    except Exception as e:
        print(f"警告：无法调整模型词表或打印special token id: {e}")
    # NOTE: translated from Chinese (added)
    # NOTE: translated from Chinese
    if training_config.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(use_reentrant=False)
        except Exception:
            pass
    else:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    try:
        if hasattr(model, "config"):
            # NOTE: translated from Chinese (binding)
            model.config.use_cache = not training_config.gradient_checkpointing
            model.config.output_hidden_states = True
    except Exception:
        pass
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    # NOTE: translated from Chinese (weight)
    if args.load_from_checkpoint:
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"正在从 checkpoint 加载模型权重（不恢复训练状态）: {args.load_from_checkpoint}")
        # NOTE: translated from Chinese (weight)
        # NOTE: translated from Chinese (weight)
        try:
            model = PeftModel.from_pretrained(model, args.load_from_checkpoint)
            if not silence and is_main:
                print(f"已成功加载 checkpoint 模型权重: {args.load_from_checkpoint}")
        except Exception as e:
            raise RuntimeError(
                f"无法从 checkpoint 加载模型权重: {args.load_from_checkpoint}\n"
                f"错误: {e}\n"
                "请确保 checkpoint 路径正确且包含有效的 adapter 权重文件（adapter_model.safetensors 或 adapter_model.bin）。"
            ) from e

    # NOTE: translated from Chinese (loss)
    margin_loss = None
    if args.use_margin_loss:
        margin_loss = MarginLoss(
            margin=args.margin_value,
            margin_weight=args.margin_weight,
            pos_weight=args.pos_weight,
            neg_weight=args.neg_weight,
        )
        print(
            f"启用边际损失: margin={args.margin_value}, margin_weight={args.margin_weight}, "
            f"pos_weight={args.pos_weight}, neg_weight={args.neg_weight}"
        )
    # TRL's SFTTrainer still expects a non-None iterable train_dataset even in eval-only workflows.
    # Use eval_dataset as a lightweight placeholder to avoid scanning real train data.
    trainer_train_dataset = train_dataset if train_dataset is not None else eval_dataset
    trainer = MarginSFTTrainer(
        model=model,
        args=training_config,
        peft_config=peft_config,
        train_dataset=trainer_train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id or tokenizer.eos_token_id),
        margin_loss=margin_loss,
        processor=processor,
        trigger_loss_weight=args.trigger_loss_weight,
        # NOTE: translated from Chinese (added, binding)
        bind_trigger_task=args.bind_trigger_task,
        bind_task_step=args.bind_task_step,
        bind_loss_weight=args.bind_loss_weight,
        bind_tt_weight=args.bind_tt_weight,
        bind_ts_weight=args.bind_ts_weight,
        bind_trigger_disc_weight=args.bind_trigger_disc_weight,
        # NOTE: translated from Chinese (added)
        enable_task_step_constraint=args.enable_task_step_constraint,
        task_step_constraint_weight=args.task_step_constraint_weight,
        annotation_path=args.annotation if getattr(args, "annotation", "") else None,
        # NOTE: translated from Chinese (added)
        class_weighted=args.class_weighted,
    )

    # NOTE: translated from Chinese
    try:
        if bool(getattr(args, "limit_embedding_training", False)):
            # NOTE: translated from Chinese
            special_tokens_list = ["<|trigger_start|>", "<|trigger_end|>", "<|task_start|>", "<|task_end|>", "<|step_start|>", "<|step_end|>"]
            if args.predict_steps and args.predict_steps > 0:
                special_tokens_list.extend(["<|future_steps_start|>", "<|future_steps_end|>"])
            if getattr(args, "predict_next_action", False):
                special_tokens_list.extend(["<|next_action_start|>", "<|next_action_end|>"])
            if args.reasoning:
                special_tokens_list.extend(["<|reasoning_start|>", "<|reasoning_end|>"])
            # NOTE: translated from Chinese
            label_tokens = ["true", "false", "1", "0"]  # NOTE: translated from Chinese
            other_important_tokens = ["<|im_end|>", "<|endoftext|>"]  # NOTE: translated from Chinese
            
            all_trainable_tokens = special_tokens_list + label_tokens + other_important_tokens
            trainable_token_ids = []
            
            for token in all_trainable_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    trainable_token_ids.append(token_id)
                    if not silence and is_main:
                        print(f"  可训练token: {token} -> {token_id}")
                else:
                    if not silence and is_main:
                        print(f"  警告: {token} 被识别为UNK token")

            # NOTE: translated from Chinese
            peft_or_base_model = trainer.model

            # NOTE: translated from Chinese
            embed_layer = peft_or_base_model.get_input_embeddings()
            lm_head_layer = None
            try:
                lm_head_layer = peft_or_base_model.get_output_embeddings()
            except Exception:
                pass

            # NOTE: translated from Chinese (weight)
            if hasattr(embed_layer, "weight"):
                embed_layer.weight.requires_grad_(True)
            if lm_head_layer is not None and hasattr(lm_head_layer, "weight"):
                lm_head_layer.weight.requires_grad_(True)

            # NOTE: translated from Chinese
            vocab_size = int(embed_layer.weight.size(0))
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=embed_layer.weight.device)
            for tid in trainable_token_ids:
                if isinstance(tid, int) and 0 <= tid < vocab_size:
                    mask[tid] = True

            def grad_mask_hook(grad):
                # grad: [vocab_size, hidden]
                return grad * mask[:, None].to(grad.dtype)

            embed_layer.weight.register_hook(grad_mask_hook)
            if lm_head_layer is not None and lm_head_layer is not embed_layer:
                lm_head_layer.weight.register_hook(grad_mask_hook)

            # NOTE: translated from Chinese
            if not silence and is_main:
                print(f"已启用扩展梯度掩码，放开 {len(trainable_token_ids)} 个token的训练")
                print(f"Special tokens: {special_tokens_list}")
                print(f"Label tokens: {label_tokens}")
                print(f"Other tokens: {other_important_tokens}")
        else:
            # NOTE: translated from Chinese (weight)
            peft_or_base_model = trainer.model
            embed_layer = peft_or_base_model.get_input_embeddings()
            lm_head_layer = None
            try:
                lm_head_layer = peft_or_base_model.get_output_embeddings()
            except Exception:
                pass
            if hasattr(embed_layer, "weight"):
                embed_layer.weight.requires_grad_(True)
            if lm_head_layer is not None and hasattr(lm_head_layer, "weight"):
                lm_head_layer.weight.requires_grad_(True)
            if not silence and is_main:
                print("Embedding/lm_head：训练全部token（未启用行级梯度掩码）")
    except Exception as _e:
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"警告：设置embedding/lm_head训练策略失败：{_e}")

    # NOTE: translated from Chinese
    try:
        trainer.args.scores_max_tokens = int(args.scores_max_tokens)
    except Exception:
        pass

    # NOTE: translated from Chinese
    try:
        trainer.args.user_debug = bool(args.debug)
        trainer.args.user_debug_samples = int(args.debug_samples)
        trainer.args.eval_every_n_epochs = int(args.eval_every_n_epochs)
        trainer.args.eval_generation_batch_size = int(args.eval_generation_batch_size)
    except Exception:
        pass

    # NOTE: translated from Chinese
    if args.labels_only:
        os.environ["LABELS_ONLY"] = "1"
    else:
        os.environ.pop("LABELS_ONLY", None)

    # NOTE: translated from Chinese
    def compute_metrics_fn(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        计算监督学习的评估指标
        原理：使用teacher-forcing的下一token分类，抽取labels中首个非-100的token与logits对应位置
        """
        import numpy as np
        import torch
        logits = eval_pred.predictions  # NOTE: translated from Chinese (predict)
        labels = eval_pred.label_ids   # NOTE: translated from Chinese
        
        # NOTE: translated from Chinese
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            
        # NOTE: translated from Chinese
        first_pos = []
        for y in labels:
            idx = np.where(y != -100)[0]  # NOTE: translated from Chinese
            first_pos.append(int(idx[0]) if len(idx) else -1)  # NOTE: translated from Chinese
            
        # NOTE: translated from Chinese (predict)
        preds = []
        gts = []
        for i, pos in enumerate(first_pos):
            if pos == -1:
                continue
            p = int(logits[i, pos].argmax(-1))  # NOTE: translated from Chinese (predict)
            preds.append(p)
            gts.append(int(labels[i, pos]))     # NOTE: translated from Chinese
            
        # NOTE: translated from Chinese
        if not gts:
            return {"val_acc": 0.0}
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        acc = float(np.mean(np.array(preds) == np.array(gts)))
        return {"val_acc": acc}

    trainer.compute_metrics = compute_metrics_fn

    # NOTE: translated from Chinese
    class LossComponentsLoggingCallback(TrainerCallback):
        def __init__(self, trainer_ref):
            self.trainer_ref = trainer_ref
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """在每次 logging 时显示所有 loss 组件"""
            if logs is None:
                return
            
            # NOTE: translated from Chinese
            trainer = self.trainer_ref
            if trainer is not None and hasattr(trainer, "_loss_components_history") and trainer._loss_components_history:
                recent_components = trainer._loss_components_history[-1]
                
                # NOTE: translated from Chinese
                loss_parts = []
                if "ce_loss" in recent_components:
                    loss_parts.append(f"CE={recent_components['ce_loss']:.4f}")
                if "margin_loss" in recent_components:
                    loss_parts.append(f"Margin={recent_components['margin_loss']:.4f}")
                if "trigger_cls_loss" in recent_components:
                    loss_parts.append(f"TriggerCls={recent_components['trigger_cls_loss']:.4f}")
                if "bind_loss" in recent_components:
                    loss_parts.append(f"Bind={recent_components['bind_loss']:.4f}")
                if "constraint_loss" in recent_components:
                    loss_parts.append(f"Constraint={recent_components['constraint_loss']:.4f}")
                
                if loss_parts:
                    # NOTE: translated from Chinese
                    logs["loss_components"] = " | ".join(loss_parts)
                    # NOTE: translated from Chinese
                    if "loss" in logs:
                        logs["loss"] = f"{logs['loss']:.4f} ({' | '.join(loss_parts)})"

    trainer.add_callback(LossComponentsLoggingCallback(trainer))

    # NOTE: translated from Chinese
    class LossHistoryCallback(TrainerCallback):
        def __init__(self, trainer_ref, output_dir):
            self.trainer_ref = trainer_ref
            self.output_dir = output_dir
            self.loss_history_file = os.path.join(output_dir, "loss_history.csv")
            self.loss_records = []
            self.csv_initialized = False
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """在每次 logging 时记录 loss 组件"""
            if logs is None:
                return
            
            # NOTE: translated from Chinese
            trainer = self.trainer_ref
            if trainer is not None and hasattr(trainer, "_loss_components_history") and trainer._loss_components_history:
                recent_components = trainer._loss_components_history[-1]
                
                # NOTE: translated from Chinese
                record = {
                    "step": state.global_step,
                    "epoch": state.epoch if state.epoch is not None else 0.0,
                    "total_loss": logs.get("loss", 0.0),
                    "learning_rate": logs.get("learning_rate", 0.0),
                    "grad_norm": logs.get("grad_norm", 0.0),
                }
                
                # NOTE: translated from Chinese
                for loss_name, loss_value in recent_components.items():
                    record[loss_name] = loss_value
                
                self.loss_records.append(record)
        
        def on_save(self, args, state, control, **kwargs):
            """在保存 checkpoint 时将 loss 历史写入 CSV"""
            if not self.loss_records:
                return
            
            try:
                import csv
                
                # NOTE: translated from Chinese
                all_keys = set()
                for record in self.loss_records:
                    all_keys.update(record.keys())
                
                # NOTE: translated from Chinese
                base_columns = ["step", "epoch", "total_loss", "learning_rate", "grad_norm"]
                loss_columns = sorted([k for k in all_keys if k not in base_columns])
                fieldnames = base_columns + loss_columns
                
                # NOTE: translated from Chinese
                file_exists = os.path.exists(self.loss_history_file)
                
                with open(self.loss_history_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    # NOTE: translated from Chinese
                    if not file_exists or not self.csv_initialized:
                        writer.writeheader()
                        self.csv_initialized = True
                    
                    # NOTE: translated from Chinese
                    for record in self.loss_records:
                        # NOTE: translated from Chinese
                        row = {k: record.get(k, 0.0) for k in fieldnames}
                        writer.writerow(row)
                
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"✅ 已保存 {len(self.loss_records)} 条 loss 记录到 {self.loss_history_file}")
                
                # NOTE: translated from Chinese
                self.loss_records = []
                
            except Exception as e:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"⚠️ 保存 loss 历史失败: {e}")
        
        def on_train_end(self, args, state, control, **kwargs):
            """训练结束时确保所有记录都被保存"""
            self.on_save(args, state, control, **kwargs)

    trainer.add_callback(LossHistoryCallback(trainer, args.output_dir))

    # NOTE: translated from Chinese
    class SaveOnlyModelCallback(TrainerCallback):
        def __init__(self, save_only_model=False):
            self.save_only_model = save_only_model
            self.last_saved_step = -1
        
        def on_save(self, args, state, control, **kwargs):
            """在保存 checkpoint 后删除优化器状态文件"""
            if not self.save_only_model:
                return
            
            # NOTE: translated from Chinese
            import time
            time.sleep(0.5)
            
            # NOTE: translated from Chinese (args here is Trainer's TrainingArguments; output_dir is run_dir)
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            
            if not os.path.exists(checkpoint_dir):
                return
            
            # NOTE: translated from Chinese
            files_to_remove = [
                "optimizer.pt",
                "scheduler.pt",
            ]
            # NOTE: translated from Chinese
            import glob
            rng_files = glob.glob(os.path.join(checkpoint_dir, "rng_state_*.pth"))
            files_to_remove.extend([os.path.basename(f) for f in rng_files])
            
            removed_count = 0
            for filename in files_to_remove:
                filepath = os.path.join(checkpoint_dir, filename)
                if os.path.exists(filepath):
                    try:
                        file_size = os.path.getsize(filepath) / (1024**3)  # GB
                        os.remove(filepath)
                        removed_count += 1
                        silence, is_main, _, _ = _get_env_silence_and_rank()
                        if not silence and is_main:
                            print(f"已删除 {filename} ({file_size:.2f}GB) - checkpoint-{state.global_step}")
                    except Exception as e:
                        silence, is_main, _, _ = _get_env_silence_and_rank()
                        if not silence and is_main:
                            print(f"警告：无法删除 {filepath}: {e}")
            
            if removed_count > 0:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"已从 checkpoint-{state.global_step} 删除 {removed_count} 个优化器状态文件，节省空间")

    class EpochCheckpointAliasCallback(TrainerCallback):
        """Create epoch_N aliases that point to checkpoint-<step>."""

        def on_save(self, args, state, control, **kwargs):
            # Only rank0 manipulates alias links.
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return

            try:
                if state.epoch is None:
                    return
                epoch_tag = int(round(float(state.epoch)))
            except Exception:
                return

            if epoch_tag <= 0:
                return

            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt_dir):
                return

            alias_path = os.path.join(args.output_dir, f"epoch_{epoch_tag}")
            target_rel = os.path.basename(ckpt_dir)

            try:
                if os.path.lexists(alias_path):
                    if os.path.islink(alias_path) or os.path.isfile(alias_path):
                        os.remove(alias_path)
                    elif os.path.isdir(alias_path):
                        import shutil
                        shutil.rmtree(alias_path)

                # Use relative symlink so run directory can move as a whole.
                os.symlink(target_rel, alias_path)
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"已创建 epoch 别名: epoch_{epoch_tag} -> {target_rel}")
            except Exception as e:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"警告：创建 epoch ckpt 别名失败(epoch_{epoch_tag}): {e}")

    def _restore_onestep_metrics_columns(_run_dir: str) -> None:
        """Delegate to unified_metrics module."""
        restore_onestep_metrics_columns(_run_dir)

    def _mark_eval_shard_done(_rec_dir: str, _epoch_tag: str, _rank: int, _out_path: str) -> None:
        done_path = eval_shard_path(_rec_dir, _epoch_tag, _rank, "done")
        try:
            if not os.path.exists(_out_path):
                with open(_out_path, "a", encoding="utf-8") as _wf:
                    _wf.write("")
                    _wf.flush()
                    try:
                        os.fsync(_wf.fileno())
                    except Exception:
                        pass
            with open(done_path, "w", encoding="utf-8") as df:
                df.write("ok")
                df.flush()
                try:
                    os.fsync(df.fileno())
                except Exception:
                    pass
        except Exception:
            pass

    def _wait_for_eval_shards(_rec_dir: str, _epoch_tag: str, _rank: int, _world: int) -> None:
        if _rank != 0:
            return
        timeout_s = 30 * 60
        start_t = time.time()
        last_report_t = 0.0
        need_paths = []
        for r in range(_world):
            need_paths.append(eval_shard_path(_rec_dir, _epoch_tag, r, "jsonl"))
            need_paths.append(eval_shard_path(_rec_dir, _epoch_tag, r, "done"))
        while True:
            missing = [p for p in need_paths if not os.path.exists(p)]
            if not missing:
                return
            now_t = time.time()
            if (now_t - last_report_t) >= 10.0:
                missing_done_ranks = []
                for r in range(_world):
                    done_p = eval_shard_path(_rec_dir, _epoch_tag, r, "done")
                    if not os.path.exists(done_p):
                        missing_done_ranks.append(r)
                # Always print this heartbeat even when --silence is enabled,
                # otherwise users may think the process is stuck with no clue.
                print(
                    f"[Eval] waiting shards epoch={_epoch_tag}: "
                    f"missing_files={len(missing)}, missing_done_ranks={missing_done_ranks}",
                    flush=True,
                )
                last_report_t = now_t
            if (time.time() - start_t) > timeout_s:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(
                        f"[Eval] ERROR: 等待 eval shard 超时，epoch={_epoch_tag} "
                        f"missing={len(missing)} (示例: {missing[:2]})"
                    )
                raise RuntimeError(f"epoch {_epoch_tag} eval shards not complete; skip partial metrics merge")
            time.sleep(2.0)

    def _run_scheme_c_onestep_eval(
        _run_dir: str,
        _epoch_tag: str,
        _rank: int,
        _world: int,
        _args,
        _eval_ds,
    ) -> None:
        try:
            import csv as _csv
            import subprocess
            import time as _time

            repo_root = Path(__file__).resolve().parents[2]
            eval_script = repo_root / "test" / "onestep_planning" / "eval_onestep_end2end.py"
            agg_script = repo_root / "test" / "onestep_planning" / "aggregate_onestep_shards.py"
            if not eval_script.exists() or not agg_script.exists():
                raise FileNotFoundError("missing one-step eval/agg scripts")

            pred_dir = Path(_run_dir) / "eval_pred"
            out_dir = Path(_run_dir)
            one_dir = out_dir / "onestep_metrics"
            one_dir.mkdir(parents=True, exist_ok=True)

            onestep_world = max(int(getattr(_args, "onestep_workers", 16) or 16), _world)
            pred_input = Path(eval_shard_path(str(pred_dir), _epoch_tag, _rank, "jsonl"))
            if not pred_input.exists():
                raise FileNotFoundError(f"missing eval shard for one-step: {pred_input}")
            worker_ids = list(range(_rank, onestep_world, _world))

            def _fmt_progress(_done: int, _total: int, width: int = 20) -> str:
                if _total <= 0:
                    return "[no-workers]"
                filled = int(width * _done / _total)
                filled = max(0, min(width, filled))
                return "[" + "#" * filled + "." * (width - filled) + f"] {_done}/{_total}"

            l1_json = Path(str(getattr(_eval_ds, "_jsonl_path", "") or "")).resolve()
            if not l1_json.exists():
                cand = getattr(_args, "val_json", None)
                if cand:
                    l1_json = Path(str(cand)).resolve()
            anno = getattr(_args, "annotation", None) or getattr(_eval_ds, "_annotation_path", None)
            window_stride = int(getattr(_eval_ds, "window_stride", 3))
            horizon = int(getattr(_args, "predict_steps", 0) or 0)

            procs = []
            for wid in worker_ids:
                shard_agg = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, wid)).stem}.agg.json"
                shard_done = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, wid)).stem}.done"
                shard_log = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, wid)).stem}.log"
                shard_csv = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, wid)).stem}.csv"
                shard_actions = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, wid)).stem}.actions.jsonl"
                for stale_p in (shard_agg, shard_done, shard_log, shard_csv, shard_actions):
                    try:
                        if stale_p.exists():
                            stale_p.unlink()
                    except Exception:
                        pass
                cmd = [
                    sys.executable, str(eval_script),
                    "--pred_path", str(pred_input),
                    "--method_name", f"cot_sft_epoch{_epoch_tag}",
                    "--l1_json", str(l1_json),
                    "--annotation", str(anno) if anno else "",
                    "--window_stride", str(window_stride),
                    "--horizon", str(horizon if horizon > 0 else 5),
                    "--append_terminate",
                    "--human_mode", "hmin",
                    "--action_selector", "entropy",
                    "--entropy_candidate_mode", "future",
                    "--immediate_M", "1",
                    "--include_entropy_baseline",
                    "--shard_rank", str(wid),
                    "--shard_world", str(onestep_world),
                    "--tqdm_position", str(wid),
                    "--out_dir", str(out_dir),
                    "--metrics_subpath", onestep_metrics_relpath(_epoch_tag, wid),
                    "--dump_agg_json", str(shard_agg),
                    "--done_path", str(shard_done),
                    "--action_log", str(shard_actions),
                ]
                cmd = [c for c in cmd if c != ""]
                p = subprocess.Popen(cmd)
                procs.append((p, wid, shard_log))

            done_paths = {
                wid: one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, wid)).stem}.done"
                for wid in range(onestep_world)
            }
            if _rank == 0:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(
                        f"[Eval][one-step] 启动 worker: epoch={_epoch_tag}, "
                        f"world={onestep_world}, local_workers={worker_ids}, pred_input={pred_input}"
                    )

            closed_workers = set()
            last_done = -1
            while True:
                alive = 0
                for p, wid, shard_log in procs:
                    rc = p.poll()
                    if rc is None:
                        alive += 1
                        continue
                    if wid in closed_workers:
                        continue
                    closed_workers.add(wid)
                    if rc != 0:
                        raise RuntimeError(
                            f"one-step worker failed: rank={_rank}, wid={wid}, rc={rc}"
                        )
                if _rank == 0:
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    done_count = sum(1 for p in done_paths.values() if p.exists())
                    if done_count != last_done and not silence and is_main:
                        print(
                            f"[Eval][one-step] worker 进度 epoch={_epoch_tag}: "
                            f"{_fmt_progress(done_count, onestep_world)}"
                        )
                        last_done = done_count
                if alive == 0:
                    break
                _time.sleep(5.0)

            _safe_dist_barrier(f"onestep_workers_done_epoch{_epoch_tag}")

            if _rank == 0:
                timeout_s = 12 * 3600
                start_t = _time.time()
                silence, is_main, _, _ = _get_env_silence_and_rank()
                while True:
                    done_count = sum(1 for p in done_paths.values() if p.exists())
                    if done_count != last_done and not silence and is_main:
                        print(
                            f"[Eval][one-step] worker 进度 epoch={_epoch_tag}: "
                            f"{_fmt_progress(done_count, onestep_world)}"
                        )
                        last_done = done_count
                    if done_count == onestep_world:
                        break
                    if _time.time() - start_t > timeout_s:
                        missing = [str(p) for p in done_paths.values() if not p.exists()]
                        raise RuntimeError(
                            f"one-step done wait timeout: epoch={_epoch_tag}, missing={len(missing)}, sample={missing[:2]}"
                        )
                    _time.sleep(5.0)

                merged_action_log = out_dir / "planning_actions.jsonl"
                with merged_action_log.open("w", encoding="utf-8") as wf_actions:
                    for wid in range(onestep_world):
                        shard_actions = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, wid)).stem}.actions.jsonl"
                        if not shard_actions.exists():
                            continue
                        with shard_actions.open("r", encoding="utf-8") as rf_actions:
                            for line in rf_actions:
                                wf_actions.write(line)

                out_csv = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, 'agg')).stem}.csv"
                if not silence and is_main:
                    print(f"[Eval][one-step] 开始聚合 epoch={_epoch_tag}")
                cmd2 = [
                    sys.executable, str(agg_script),
                    "--epoch", str(_epoch_tag),
                    "--run_dir", str(out_dir),
                    "--world", str(onestep_world),
                    "--out_csv", str(out_csv),
                ]
                log_path = one_dir / f"{Path(onestep_metrics_relpath(_epoch_tag, 'agg')).stem}.log"
                with log_path.open("w", encoding="utf-8") as lf2:
                    subprocess.run(cmd2, stdout=lf2, stderr=lf2, check=True)

                if out_csv.exists():
                    with out_csv.open("r", encoding="utf-8") as rf:
                        rows = list(_csv.DictReader(rf))
                    if rows:
                        model_row = None
                        for r0 in rows:
                            if str(r0.get("method", "")).strip().lower() == "model":
                                model_row = r0
                                break
                        if model_row is None:
                            model_row = rows[-1]

                        agg_hist = one_dir / "onestep_metrics.csv"
                        header = ["epoch", "immediate_saved_rate", "effective_avg_entropy", "APA"]
                        write_header = not agg_hist.exists()
                        with agg_hist.open("a", encoding="utf-8", newline="") as wf:
                            w = _csv.DictWriter(wf, fieldnames=header)
                            if write_header:
                                w.writeheader()
                            w.writerow(
                                {
                                    "epoch": str(_epoch_tag),
                                    "immediate_saved_rate": model_row.get("immediate_saved_rate", ""),
                                    "effective_avg_entropy": model_row.get("effective_avg_entropy", ""),
                                    "APA": model_row.get("APA", ""),
                                }
                            )
                        if not silence and is_main:
                            print(
                                f"[Eval][one-step] 完成 epoch={_epoch_tag}: "
                                f"immediate_saved_rate={model_row.get('immediate_saved_rate', '')}, "
                                f"effective_avg_entropy={model_row.get('effective_avg_entropy', '')}, "
                                f"APA={model_row.get('APA', '')}"
                            )
        except Exception as e:
            try:
                import traceback as _traceback
                err_path = Path(_run_dir) / "onestep_metrics" / f"{_epoch_tag}.rank{_rank}.error.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(_traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"[Eval] scheme-C one-step eval skipped/failed: {e}")
            if bool(getattr(_args, "eval_only", False)):
                raise

    class EpochMetricsCallback(TrainerCallback):
        def __init__(self, trainer_ref, run_dir_ref, processor_ref, eval_ds_ref, record_prompts=False):
            self.trainer = trainer_ref
            self.run_dir = run_dir_ref
            self.processor = processor_ref
            self.eval_ds = eval_ds_ref
            self.record_prompts = record_prompts
            # NOTE: translated from Chinese
            try:
                self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            except Exception:
                self.local_rank = 0
            # NOTE: translated from Chinese (cache)
            self._task_managers: Dict[str, TaskGraphManager] = {}
            self._planners: Dict[str, EntropyPlanner] = {}
            # NOTE: translated from Chinese
            self._vocab_id_to_name = getattr(eval_ds_ref, "_vocab_id_to_name", {})
            self._vocab_name_to_id = {v: k for k, v in self._vocab_id_to_name.items()}
            # NOTE: translated from Chinese
            self._all_task_names = set()
            self._all_step_names = set(self._vocab_id_to_name.values())
            # NOTE: translated from Chinese
            if hasattr(eval_ds_ref, "video_task_names"):
                self._all_task_names.update(eval_ds_ref.video_task_names.values())
            if hasattr(eval_ds_ref, "task_lookup"):
                self._all_task_names.update(eval_ds_ref.task_lookup.values())
            self._belief_postprocessor = None
            if bool(getattr(self.trainer.args, "belief_state_eval", False)):
                self._belief_postprocessor = BeliefStatePostProcessor(
                    annotation_path=getattr(self.trainer.args, "annotation", None),
                    known_tasks=self._all_task_names,
                    known_steps=self._all_step_names,
                    task_window=int(getattr(self.trainer.args, "belief_task_history_window", 6) or 6),
                    step_window=int(getattr(self.trainer.args, "belief_step_history_window", 4) or 4),
                    future_topk=int(getattr(self.trainer.args, "belief_future_topk", 5) or 5),
                )
        
        def _normalize_text(self, text: str) -> str:
            """归一化文本用于匹配"""
            return "".join(text.lower().split())
        
        def _fuzzy_match(self, pred_text: str, candidate_set: Set[str], threshold: float = 0.9) -> Optional[str]:
            """
            使用相似度匹配预测文本和候选集合
            返回匹配度最高的候选（如果相似度 >= threshold），否则返回 None
            """
            if not pred_text or not candidate_set:
                return None
            
            import difflib
            pred_norm = self._normalize_text(pred_text)
            best_match = None
            best_ratio = 0.0
            
            for candidate in candidate_set:
                if not candidate:
                    continue
                candidate_norm = self._normalize_text(candidate)
                # NOTE: translated from Chinese
                ratio = difflib.SequenceMatcher(None, pred_norm, candidate_norm).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = candidate
            
            if best_ratio >= threshold:
                return best_match
            return None
        
        def _get_history_steps_names(self, sample_idx: int) -> List[str]:
            """
            从 dataset 中获取当前样本之前已完成的历史步骤名称列表
            
            Args:
                sample_idx: 当前样本索引
                
            Returns:
                List[str]: 历史步骤名称列表
            """
            history_steps = []
            if not hasattr(self.eval_ds, 'data') or sample_idx >= len(self.eval_ds.data):
                return history_steps
            
            # NOTE: translated from Chinese
            current_video_id = None
            try:
                current_item = self.eval_ds.data[sample_idx]
                if isinstance(current_item, dict):
                    current_video_id = current_item.get('video_id')
            except Exception:
                pass
            
            if not current_video_id:
                return history_steps
            
            # NOTE: translated from Chinese (step, iterate)
            vocab_id_to_name = getattr(self.eval_ds, '_vocab_id_to_name', {})
            
            for i in range(sample_idx):
                try:
                    item = self.eval_ds.data[i]
                    if not isinstance(item, dict):
                        continue
                    
                    # NOTE: translated from Chinese (check)
                    vid = item.get('video_id')
                    if vid != current_video_id:
                        continue
                    
                    # NOTE: translated from Chinese
                    # NOTE: translated from Chinese
                    labels = item.get('frame_labels', [])
                    if isinstance(labels, list):
                        # NOTE: translated from Chinese
                        for label_id in labels:
                            if isinstance(label_id, int) and label_id > 0:
                                step_name = vocab_id_to_name.get(label_id, '')
                                if step_name and step_name not in history_steps:
                                    history_steps.append(step_name)
                except Exception:
                    continue
            
            return history_steps

        def on_epoch_end(self, args, state, control, **kwargs):
            # NOTE: translated from Chinese
            try:
                n = int(getattr(self.trainer.args, "eval_every_n_epochs", 1))
            except Exception:
                n = 1
            # NOTE: sync an epoch_tag across ranks to avoid drift (state.epoch can differ slightly).
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world = dist.get_world_size()
            else:
                rank = 0
                world = 1
            epoch_tag_obj = [0]
            if rank == 0:
                try:
                    if state.epoch is not None:
                        epoch_tag_obj[0] = int(round(float(state.epoch)))
                    else:
                        epoch_tag_obj[0] = int(state.global_step)
                except Exception:
                    epoch_tag_obj[0] = int(state.global_step)
            if world > 1:
                dist.broadcast_object_list(epoch_tag_obj, src=0)
            epoch_tag = int(epoch_tag_obj[0])
            if n > 1 and (epoch_tag % n != 0):
                return
            # NOTE: translated from Chinese
            eval_out = self.trainer.evaluate()
            val_loss = float(eval_out.get("eval_loss", 0.0))
            # NOTE: get train_loss early so merge can include it (log_history updated after epoch).
            train_loss = None
            for log in reversed(self.trainer.state.log_history):
                if "loss" in log and "epoch" in log and int(log["epoch"]) == epoch_tag:
                    train_loss = float(log["loss"])
                    break
            # NOTE: translated from Chinese (stats)
            model = self.trainer.model
            model.eval()
            rec_dir = os.path.join(self.run_dir, "eval_pred")
            os.makedirs(rec_dir, exist_ok=True)
            out_path = os.path.join(rec_dir, f"epoch_{epoch_tag}.rank{rank}.jsonl")
            done_path = os.path.join(rec_dir, f"epoch_{epoch_tag}.rank{rank}.done")
            tp = fp = tn = fn = 0
            task_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            step_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            total_generation_time = 0.0  # NOTE: translated from Chinese
            generation_count = 0  # NOTE: translated from Chinese
            predict_steps_enabled = getattr(self.trainer.args, "predict_steps", 0) > 0
            future_edit_sum = 0.0
            future_edit_count = 0
            eval_round_k = max(0, int(getattr(self.trainer.args, "eval_round_k", 0) or 0))
            with open(out_path, "w", encoding="utf-8") as wf:
                # NOTE: translated from Chinese
                wf.flush()
                try:
                    os.fsync(wf.fileno())
                except Exception:
                    pass
                rows_written = 0
                from tqdm import tqdm
                # NOTE: translated from Chinese
                try:
                    local_indices = list(range(rank, len(self.eval_ds), world))
                    pbar = tqdm(
                        total=len(local_indices),
                        desc=f"Eval generate r{rank}",
                        ncols=0,
                        position=rank,
                        leave=True,
                        dynamic_ncols=True,
                    )

                    def _process_eval_index(i: int) -> None:
                        nonlocal tp, fp, tn, fn
                        nonlocal total_generation_time, generation_count
                        nonlocal future_edit_sum, future_edit_count
                        nonlocal rows_written
                        ex = self.eval_ds[i]
                        # NOTE: translated from Chinese
                        try:
                            if "prompt_input_ids" in ex and isinstance(ex["prompt_input_ids"], torch.Tensor):
                                prompt_ids_text = self.processor.batch_decode(
                                    ex["prompt_input_ids"].unsqueeze(0), skip_special_tokens=False
                                )[0]
                            else:
                                prompt_ids_text = ""
                        except Exception:
                            prompt_ids_text = ""

                        generation_start_time = time.time()
                        include_scores = bool(
                            getattr(self.eval_ds, "enable_scores", False)
                            or getattr(self.eval_ds, "if_score", False)
                        )
                        normalized_text, pred_is, pred_task, pred_step, pred_future_steps, _ = predict_trigger_with_logits(
                            model,
                            self.processor,
                            ex,
                            include_scores=include_scores,
                            priority_scores=getattr(self.eval_ds, "priority_scores", {}),
                            missing_priority_cache=getattr(self.eval_ds, "_missing_priority_tasks", None),
                        )
                        generation_time = time.time() - generation_start_time
                        total_generation_time += generation_time
                        generation_count += 1

                        gt = self.eval_ds.targets[i] if hasattr(self.eval_ds, "targets") and i < len(self.eval_ds.targets) else 0
                        gt_task = self.eval_ds.gt_tasks[i] if hasattr(self.eval_ds, "gt_tasks") and i < len(self.eval_ds.gt_tasks) else ""
                        gt_step = self.eval_ds.gt_steps[i] if hasattr(self.eval_ds, "gt_steps") and i < len(self.eval_ds.gt_steps) else ""
                        gt_future = self.eval_ds.gt_future_steps[i] if hasattr(self.eval_ds, "gt_future_steps") and i < len(self.eval_ds.gt_future_steps) else []
                        if pred_is == 1 and gt == 1: tp += 1
                        elif pred_is == 1 and gt == 0: fp += 1
                        elif pred_is == 0 and gt == 0: tn += 1
                        else: fn += 1

                        current_video_id = ""
                        try:
                            if hasattr(self.eval_ds, "data") and i < len(self.eval_ds.data):
                                current_item = self.eval_ds.data[i]
                                if isinstance(current_item, dict):
                                    current_video_id = str(current_item.get("video_id", "") or "")
                        except Exception:
                            current_video_id = ""

                        # NOTE: translated from Chinese (added, predict)
                        matched_pred_task = pred_task
                        matched_pred_step = pred_step
                        belief_completed_steps = []
                        
                        if pred_task and self._all_task_names:
                            matched = self._fuzzy_match(pred_task, self._all_task_names, threshold=0.9)
                            if matched:
                                matched_pred_task = matched
                        
                        if pred_step and self._all_step_names:
                            matched = self._fuzzy_match(pred_step, self._all_step_names, threshold=0.9)
                            if matched:
                                matched_pred_step = matched

                        if self._belief_postprocessor is not None:
                            belief_state = self._belief_postprocessor.refine(
                                video_id=current_video_id,
                                pred_is=bool(pred_is),
                                pred_task=matched_pred_task,
                                pred_step=matched_pred_step,
                                pred_future_steps=pred_future_steps,
                            )
                            matched_pred_task = belief_state.get("task", matched_pred_task)
                            matched_pred_step = belief_state.get("step", matched_pred_step)
                            pred_future_steps = belief_state.get("future_steps", pred_future_steps)
                            belief_completed_steps = self._belief_postprocessor.get_completed_steps(current_video_id)
                        
                        def _norm_label(val: str) -> str:
                            return (val or "").strip().lower()

                        # NOTE: translated from Chinese
                        p_task_str = _norm_label(matched_pred_task)
                        g_task_str = _norm_label(gt_task)
                        p_step_str = _norm_label(matched_pred_step)
                        g_step_str = _norm_label(gt_step)

                        if gt == 1:
                            if pred_is == 1:
                                if g_task_str:
                                    if p_task_str == g_task_str:
                                        task_stats[g_task_str]["tp"] += 1
                                    else:
                                        task_stats[g_task_str]["fn"] += 1
                                        if p_task_str:
                                            task_stats[p_task_str]["fp"] += 1
                                elif p_task_str:
                                    task_stats[p_task_str]["fp"] += 1
                            else:
                                if g_task_str:
                                    task_stats[g_task_str]["fn"] += 1
                        elif pred_is == 1 and p_task_str:
                            task_stats[p_task_str]["fp"] += 1

                        if gt == 1:
                            if pred_is == 1:
                                if g_step_str:
                                    if p_step_str == g_step_str:
                                        step_stats[g_step_str]["tp"] += 1
                                    else:
                                        step_stats[g_step_str]["fn"] += 1
                                        if p_step_str:
                                            step_stats[p_step_str]["fp"] += 1
                                elif p_step_str:
                                    step_stats[p_step_str]["fp"] += 1
                            else:
                                if g_step_str:
                                    step_stats[g_step_str]["fn"] += 1
                        elif pred_is == 1 and p_step_str:
                            step_stats[p_step_str]["fp"] += 1
                        
                        # NOTE: translated from Chinese (step)
                        if predict_steps_enabled:
                            future_edit_sum += compute_edit_distance(pred_future_steps, gt_future or [])
                            future_edit_count += 1
                        
                        # NOTE: translated from Chinese (added)
                        robot_decision = None
                        robot_decision_entropy = None
                        
                        # NOTE: translated from Chinese
                        anno_path = getattr(self.trainer.args, "annotation", None)
                        
                        # NOTE: translated from Chinese (task)
                        current_task_name = matched_pred_task or pred_task
                        
                        if current_task_name and anno_path and pred_is == 1:
                            try:
                                # NOTE: translated from Chinese (cache)
                                if current_task_name not in self._task_managers:
                                    self._task_managers[current_task_name] = TaskGraphManager(anno_path, current_task_name)
                                    self._planners[current_task_name] = EntropyPlanner(self._task_managers[current_task_name])
                                
                                graph_mgr = self._task_managers[current_task_name]
                                planner = self._planners[current_task_name]
                                
                                # NOTE: translated from Chinese (step)
                                history_steps = list(belief_completed_steps) if belief_completed_steps else self._get_history_steps_names(i)
                                
                                # NOTE: translated from Chinese (predict)
                                human_predicted_future = [x.strip() for x in (pred_future_steps or []) if str(x or '').strip()]
                                if not human_predicted_future and matched_pred_step and matched_pred_step.strip():
                                    human_predicted_future = [matched_pred_step.strip()]
                                
                                # NOTE: translated from Chinese
                                robot_history_mock = []
                                
                                # NOTE: translated from Chinese
                                best_robot_action, entropy_val = planner.decide(
                                    current_completed=history_steps,
                                    human_future=human_predicted_future,
                                    robot_history=robot_history_mock
                                )
                                
                                robot_decision = best_robot_action
                                robot_decision_entropy = float(entropy_val)
                                
                            except Exception as e:
                                # NOTE: translated from Chinese
                                if getattr(self.trainer.args, "user_debug", False):
                                    print(f"Planner Error for sample {i} (task={current_task_name}): {e}")
                        
                        # NOTE: translated from Chinese (stats)
                        record = {
                            "idx": i,
                            "video_id": current_video_id,
                            "pred_text": normalized_text,
                            "raw_pred_text": normalized_text,
                            "pred": pred_is,
                            "gt": gt,
                            "pred_task": pred_task,
                            "pred_task_matched": matched_pred_task,  # NOTE: translated from Chinese (added)
                            "gt_task": gt_task,
                            "pred_step": pred_step,
                            "pred_step_matched": matched_pred_step,  # NOTE: translated from Chinese (added)
                            "gt_step": gt_step,
                            "pred_future_steps": pred_future_steps,
                            "gt_future_steps": gt_future,
                            "generation_time": generation_time,
                            "belief_state_used": bool(self._belief_postprocessor is not None),
                            "belief_completed_steps": belief_completed_steps,
                        }
                        
                        # NOTE: translated from Chinese
                        if robot_decision is not None:
                            record["robot_decision"] = robot_decision
                            record["robot_decision_entropy"] = robot_decision_entropy
                        
                        # NOTE: translated from Chinese
                        if self.record_prompts:
                            pass
                        
                        wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        rows_written += 1
                        if rows_written % 50 == 0:
                            wf.flush()
                        pbar.update(1)

                    if eval_round_k > 0 and world > 1:
                        # Keep the number of sync rounds identical across ranks; local shard sizes can differ by 1.
                        max_local_count = (len(eval_dataset) + world - 1) // world
                        max_rounds = (max_local_count + eval_round_k - 1) // eval_round_k
                        for round_id in range(max_rounds):
                            s = round_id * eval_round_k
                            for i in local_indices[s : s + eval_round_k]:
                                try:
                                    _process_eval_index(i)
                                except Exception as e:
                                    print(
                                        f"[WARN] eval sample failed: epoch={epoch_tag}, rank={rank}, idx={i}, "
                                        f"round={round_id}, err={e}"
                                    )
                                    continue
                            _safe_dist_barrier(f"eval_round_sync_epoch{epoch_tag}_round{round_id}")
                    else:
                        for i in local_indices:
                            try:
                                _process_eval_index(i)
                            except Exception as e:
                                print(f"[WARN] eval sample failed: epoch={epoch_tag}, rank={rank}, idx={i}, err={e}")
                                continue
                    pbar.close()
                finally:
                    wf.flush()
                    try:
                        os.fsync(wf.fileno())
                    except Exception:
                        pass

            # Double check: the shard output must exist before we proceed.
            try:
                if not os.path.exists(out_path):
                    with open(out_path, "a", encoding="utf-8") as _wf:
                        _wf.write("")
                        _wf.flush()
                        try:
                            os.fsync(_wf.fileno())
                        except Exception:
                            pass
                with open(done_path, "w", encoding="utf-8") as df:
                    df.write("ok")
                    df.flush()
                    try:
                        os.fsync(df.fileno())
                    except Exception:
                        pass
            except Exception:
                pass

            # NOTE: translated from Chinese
            _safe_dist_barrier("eval_gen_end")
            # Rank0: strictly wait all shard files before any global merge.
            eval_shards_ready = True
            if rank == 0:
                timeout_s = 30 * 60
                start_t = time.time()
                need_paths = []
                for r in range(world):
                    need_paths.append(os.path.join(rec_dir, f"epoch_{epoch_tag}.rank{r}.jsonl"))
                    need_paths.append(os.path.join(rec_dir, f"epoch_{epoch_tag}.rank{r}.done"))
                while True:
                    missing = [p for p in need_paths if not os.path.exists(p)]
                    if not missing:
                        break
                    if (time.time() - start_t) > timeout_s:
                        eval_shards_ready = False
                        silence, is_main, _, _ = _get_env_silence_and_rank()
                        if not silence and is_main:
                            print(
                                f"[EpochMetricsCallback] ERROR: 等待 eval shard 超时，epoch={epoch_tag} "
                                f"missing={len(missing)} (示例: {missing[:2]})"
                            )
                        break
                    time.sleep(2.0)
            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor(1 if eval_shards_ready else 0, device=torch.device("cuda", rank) if torch.cuda.is_available() else "cpu")
                dist.broadcast(flag, src=0)
                eval_shards_ready = bool(flag.item())
            if not eval_shards_ready:
                raise RuntimeError(f"epoch {epoch_tag} eval shards not complete; skip partial metrics merge")
            # NOTE: translated from Chinese (debug)
                    # if getattr(self.trainer.args, "user_debug", False) and i < int(getattr(self.trainer.args, "user_debug_samples", 5)):
                    #     try:
                    #         print("==== EVAL DEBUG SAMPLE ====\n" \
                    #               f"idx={i}\n" \
                    #               f"prompt_text={prompt_ids_text[:2000]}\n" \
                    #               f"model_output={normalized_text[:2000]}\n" \
                    #               f"pred_is={pred_is}, gt_is={gt}, pred_task={pred_task}, gt_task={gt_task}, pred_step={pred_step}, gt_step={gt_step}")
                    #     except Exception:
                    #         pass
            pos_prec = tp / (tp + fp + 1e-9)
            pos_recall = tp / (tp + fn + 1e-9)
            neg_prec = tn / (tn + fn + 1e-9)
            neg_recall = tn / (tn + fp + 1e-9)
            val_acc = (tp + tn) / max(1, (tp + tn + fp + fn))
            f1_score = 2 * (pos_prec * pos_recall) / (pos_prec + pos_recall + 1e-9)

            def compute_macro_metrics(stats_dict):
                precs = []
                recs = []
                for cls_name, counts in stats_dict.items():
                    if not cls_name:
                        continue
                    tp_cls = counts["tp"]
                    fp_cls = counts["fp"]
                    fn_cls = counts["fn"]
                    if tp_cls + fp_cls > 0:
                        precs.append(tp_cls / (tp_cls + fp_cls))
                    if tp_cls + fn_cls > 0:
                        recs.append(tp_cls / (tp_cls + fn_cls))
                m_prec = sum(precs) / len(precs) if precs else 0.0
                m_rec = sum(recs) / len(recs) if recs else 0.0
                return m_prec, m_rec

            # NOTE: translated from Chinese (stats)
            # NOTE: translated from Chinese
            # NOTE: translated from Chinese (stats, predict)
            task_mprec, task_mrec = compute_macro_metrics(task_stats)
            step_mprec, step_mrec = compute_macro_metrics(step_stats)
            future_edit_avg = future_edit_sum / max(1, future_edit_count) if predict_steps_enabled else 0.0
            # NOTE: translated from Chinese
            avg_generation_time = total_generation_time / max(1, generation_count)
            # NOTE: translated from Chinese — only append local line when merge will NOT run,
            # so we avoid duplicate epoch rows (merge overwrites the whole file with global format).
            will_merge = merge_global_metrics is not None and (
                (not dist.is_available()) or (not dist.is_initialized()) or (rank == 0)
            )
            if self.local_rank == 0 and not will_merge:
                csv_path = os.path.join(self.run_dir, "metrics.csv")
                header = "epoch,train_loss,val_loss,val_acc,f1_score,pos_prec,pos_recall,neg_prec,neg_recall,task_mPrec,task_mRec,step_mPrec,step_mRec,avg_generation_time"
                if predict_steps_enabled:
                    header += ",future_edit_dist"
                header += "\n"
                line = (
                    f"{epoch_tag},{train_loss if train_loss is not None else ''},{val_loss:.6f},"
                    f"{val_acc:.6f},{f1_score:.6f},{pos_prec:.6f},{pos_recall:.6f},{neg_prec:.6f},"
                    f"{neg_recall:.6f},{task_mprec:.6f},{task_mrec:.6f},{step_mprec:.6f},{step_mrec:.6f},"
                    f"{avg_generation_time:.6f}"
                )
                if predict_steps_enabled:
                    line += f",{future_edit_avg:.6f}"
                line += "\n"
                if not os.path.exists(csv_path):
                    with open(csv_path, "w", encoding="utf-8") as wf:
                        wf.write(header)
                        wf.write(line)
                else:
                    with open(csv_path, "a", encoding="utf-8") as wf:
                        wf.write(line)

            # NOTE: translated from Chinese (stats)
            if self.local_rank == 0:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"\n[Rank {self.local_rank}] Epoch {epoch_tag} Eval (rank-0 shard; see metrics.csv for global):")
                    print(f"  Trigger: Acc={val_acc:.4f}, F1={f1_score:.4f}, Prec={pos_prec:.4f}, Rec={pos_recall:.4f}")
                    print(f"  Task:    mPrec={task_mprec:.4f}, mRec={task_mrec:.4f}")
                    print(f"  Step:    mPrec={step_mprec:.4f}, mRec={step_mrec:.4f}")
                    if predict_steps_enabled:
                        print(f"  Future:  edit_dist={future_edit_avg:.4f}")

            # NOTE: translated from Chinese
            _safe_dist_barrier("post_local_metrics")
            if merge_global_metrics is not None and ((not dist.is_available()) or (not dist.is_initialized()) or rank == 0):
                try:
                    ov = {
                        str(epoch_tag): (
                            "" if train_loss is None else f"{train_loss:.6f}",
                            f"{val_loss:.6f}",
                        )
                    }
                    merge_global_metrics(
                        self.run_dir,
                        getattr(self.trainer.args, "predict_steps", 0),
                        epoch_train_val_override=ov,
                    )
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    if not silence and is_main:
                        print(f"[EpochMetricsCallback] Epoch {epoch_tag} 全局指标已合并并写入 metrics.csv")
                except Exception as e:
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    if not silence and is_main:
                        print(f"[EpochMetricsCallback] merge_global_metrics 失败: {e}")

            # Scheme C (strict sync): run one-step eval shards, wait all shards, then aggregate on rank0.
            try:
                from pathlib import Path
                import subprocess
                import time as _time

                repo_root = Path(__file__).resolve().parents[2]
                eval_script = repo_root / "test" / "onestep_planning" / "eval_onestep_end2end.py"
                agg_script = repo_root / "test" / "onestep_planning" / "aggregate_onestep_shards.py"
                if not eval_script.exists() or not agg_script.exists():
                    raise FileNotFoundError("missing one-step eval/agg scripts")

                pred_dir = Path(self.run_dir) / "eval_pred"
                out_dir = Path(self.run_dir)
                one_dir = out_dir / "onestep_metrics"
                one_dir.mkdir(parents=True, exist_ok=True)

                # Rank-local input: use rank shard prediction file.
                shard_pred = pred_dir / f"epoch_{epoch_tag}.rank{rank}.jsonl"
                onestep_world = max(int(getattr(self.trainer.args, "onestep_workers", 16) or 16), world)
                worker_ids = list(range(rank, onestep_world, world))

                l1_json = Path(str(getattr(self.eval_ds, "_jsonl_path", "") or "")).resolve()
                if not l1_json.exists():
                    cand = getattr(self.trainer.args, "val_json", None)
                    if cand:
                        l1_json = Path(str(cand)).resolve()
                anno = getattr(self.trainer.args, "annotation", None) or getattr(self.eval_ds, "_annotation_path", None)
                window_stride = int(getattr(self.eval_ds, "window_stride", 3))
                horizon = int(getattr(self.trainer.args, "predict_steps", 0) or 0)

                procs = []
                for wid in worker_ids:
                    shard_agg = one_dir / f"epoch_{epoch_tag}.rank{wid}.agg.json"
                    shard_done = one_dir / f"epoch_{epoch_tag}.rank{wid}.done"
                    shard_log = one_dir / f"epoch_{epoch_tag}.rank{wid}.log"
                    shard_csv = one_dir / f"epoch_{epoch_tag}.rank{wid}.csv"
                    for stale_p in (shard_agg, shard_done, shard_log, shard_csv):
                        try:
                            if stale_p.exists():
                                stale_p.unlink()
                        except Exception:
                            pass
                    cmd = [
                        sys.executable, str(eval_script),
                        "--pred_path", str(shard_pred),
                        "--method_name", f"cot_sft_epoch{epoch_tag}",
                        "--l1_json", str(l1_json),
                        "--annotation", str(anno) if anno else "",
                        "--window_stride", str(window_stride),
                        "--horizon", str(horizon if horizon > 0 else 5),
                        "--append_terminate",
                        "--human_mode", "hmin",
                        "--action_selector", "entropy",
                        "--entropy_candidate_mode", "future",
                        "--immediate_M", "1",
                        "--include_entropy_baseline",
                        "--shard_rank", str(wid),
                        "--shard_world", str(onestep_world),
                        "--tqdm_position", str(wid),
                        "--out_dir", str(out_dir),
                        "--metrics_subpath", f"onestep_metrics/epoch_{epoch_tag}.rank{wid}.csv",
                        "--dump_agg_json", str(shard_agg),
                        "--done_path", str(shard_done),
                    ]
                    cmd = [c for c in cmd if c != ""]
                    p = subprocess.Popen(cmd)
                    procs.append((p, wid, shard_log))

                for p, wid, shard_log in procs:
                    rc = p.wait()
                    if rc != 0:
                        raise RuntimeError(f"one-step worker failed: rank={rank}, wid={wid}, rc={rc}")

                _safe_dist_barrier(f"onestep_workers_done_epoch{epoch_tag}")

                if rank == 0:
                    timeout_s = 12 * 3600
                    start_t = _time.time()
                    done_paths = [one_dir / f"epoch_{epoch_tag}.rank{wid}.done" for wid in range(onestep_world)]
                    while True:
                        if all(p.exists() for p in done_paths):
                            break
                        if _time.time() - start_t > timeout_s:
                            missing = [str(p) for p in done_paths if not p.exists()]
                            raise RuntimeError(
                                f"one-step done wait timeout: epoch={epoch_tag}, missing={len(missing)}, sample={missing[:2]}"
                            )
                        _time.sleep(5.0)

                    out_csv = one_dir / f"epoch_{epoch_tag}.agg.csv"
                    cmd2 = [
                        sys.executable, str(agg_script),
                        "--epoch", str(epoch_tag),
                        "--run_dir", str(out_dir),
                        "--world", str(onestep_world),
                        "--out_csv", str(out_csv),
                    ]
                    log_path = one_dir / f"epoch_{epoch_tag}.agg.log"
                    with log_path.open("w", encoding="utf-8") as lf2:
                        subprocess.run(cmd2, stdout=lf2, stderr=lf2, check=True)

                    import csv as _csv
                    import tempfile
                    import fcntl as _fcntl

                    if out_csv.exists():
                        with out_csv.open("r", encoding="utf-8") as rf:
                            rows = list(_csv.DictReader(rf))
                        if rows:
                            model_row = None
                            for r0 in rows:
                                if str(r0.get("method", "")).strip().lower() == "model":
                                    model_row = r0
                                    break
                            if model_row is None:
                                model_row = rows[-1]

                            imm = model_row.get("immediate_saved_rate", "")
                            ent = model_row.get("effective_avg_entropy", "")
                            apa = model_row.get("APA", "")

                            agg_hist = one_dir / "onestep_metrics.csv"
                            header = ["epoch", "immediate_saved_rate", "effective_avg_entropy", "APA"]
                            write_header = not agg_hist.exists()
                            with agg_hist.open("a", encoding="utf-8", newline="") as wf:
                                w = _csv.DictWriter(wf, fieldnames=header)
                                if write_header:
                                    w.writeheader()
                                w.writerow(
                                    {
                                        "epoch": str(epoch_tag),
                                        "immediate_saved_rate": imm,
                                        "effective_avg_entropy": ent,
                                        "APA": apa,
                                    }
                                )

                            metrics_path = out_dir / "metrics.csv"
                            if metrics_path.exists():
                                with metrics_path.open("r", encoding="utf-8") as rf:
                                    reader = _csv.DictReader(rf)
                                    m_rows = list(reader)
                                    m_fieldnames = list(reader.fieldnames or [])
                                if m_rows and "epoch" in m_fieldnames:
                                    add_cols = ["onestep_immediate_saved_rate", "onestep_effective_avg_entropy", "onestep_APA"]
                                    for c in add_cols:
                                        if c not in m_fieldnames:
                                            m_fieldnames.append(c)
                                    updated = False
                                    for rr in m_rows:
                                        if str(rr.get("epoch", "")).strip() == str(epoch_tag):
                                            rr["onestep_immediate_saved_rate"] = str(imm)
                                            rr["onestep_effective_avg_entropy"] = str(ent)
                                            rr["onestep_APA"] = str(apa)
                                            updated = True
                                    if updated:
                                        lock_path = str(metrics_path) + ".lock"
                                        with open(lock_path, "w", encoding="utf-8") as lf:
                                            try:
                                                _fcntl.flock(lf.fileno(), _fcntl.LOCK_EX)
                                            except Exception:
                                                pass
                                            tmp_fd, tmp_name = tempfile.mkstemp(prefix="metrics.", suffix=".csv", dir=str(out_dir))
                                            try:
                                                with open(tmp_fd, "w", encoding="utf-8", newline="") as wf:
                                                    w = _csv.DictWriter(wf, fieldnames=m_fieldnames)
                                                    w.writeheader()
                                                    for rr in m_rows:
                                                        w.writerow({k: rr.get(k, "") for k in m_fieldnames})
                                                os.replace(tmp_name, str(metrics_path))
                                            finally:
                                                try:
                                                    if os.path.exists(tmp_name):
                                                        os.remove(tmp_name)
                                                except Exception:
                                                    pass
                _safe_dist_barrier(f"onestep_agg_done_epoch{epoch_tag}")

            except Exception as e:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"[EpochMetricsCallback] scheme-C one-step eval skipped/failed: {e}")

    # NOTE: translated from Chinese
    if args.save_only_model:
        trainer.add_callback(SaveOnlyModelCallback(save_only_model=True))
    # Keep HF checkpoint-* naming for trainer internals, and add epoch_N aliases for convenience.
    trainer.add_callback(EpochCheckpointAliasCallback())
    
    trainer.add_callback(EpochMetricsCallback(trainer, run_dir, processor, eval_dataset, record_prompts=args.record_prompts))

    def _extract_epoch_from_path(path: str) -> Optional[int]:
        if not path:
            return None
        m = re.search(r"epoch_(\d+)", path)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _collect_epoch_checkpoint_map(base_dir: str) -> Dict[int, str]:
        epoch_map: Dict[int, str] = {}
        if not base_dir or (not os.path.isdir(base_dir)):
            return epoch_map
        for name in os.listdir(base_dir):
            m = re.fullmatch(r"epoch_(\d+)", name)
            if not m:
                continue
            ep = int(m.group(1))
            p = os.path.join(base_dir, name)
            if os.path.isdir(p):
                epoch_map[ep] = p
        return epoch_map

    def _load_lora_checkpoint_for_eval(checkpoint_path: str) -> None:
        """显式加载 base + LoRA adapter，确保评估用到对应 epoch 权重。"""
        adapter_safe = os.path.join(checkpoint_path, "adapter_model.safetensors")
        adapter_bin = os.path.join(checkpoint_path, "adapter_model.bin")
        if not (os.path.exists(adapter_safe) or os.path.exists(adapter_bin)):
            raise FileNotFoundError(
                f"checkpoint 缺少 LoRA adapter 权重: {checkpoint_path} "
                "(需要 adapter_model.safetensors 或 adapter_model.bin)"
            )
        # 关键：切换下一个 epoch checkpoint 之前，先释放旧模型并清空显存。
        old_model = getattr(trainer, "model", None)
        if old_model is not None:
            try:
                old_model.eval()
            except Exception:
                pass
            try:
                old_model.cpu()
            except Exception:
                pass
        try:
            trainer.model = None
        except Exception:
            pass
        try:
            del old_model
        except Exception:
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        _safe_dist_barrier("before_eval_checkpoint_reload")

        eval_load_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if bnb_config is not None:
            eval_load_kwargs["quantization_config"] = bnb_config
        if torch.cuda.is_available():
            if local_rank >= 0:
                eval_load_kwargs["device_map"] = {"": local_rank}
            else:
                eval_load_kwargs["device_map"] = "auto"

        base_model = model_cls.from_pretrained(
            args.model_name,
            **eval_load_kwargs,
        )
        loaded_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        loaded_model.eval()
        trainer.model = loaded_model

        _safe_dist_barrier("after_eval_checkpoint_reload")

    def _run_generation_eval(epoch_tag: str) -> None:
        """执行逐样本生成评估（复用 eval_only 逻辑）。"""
        # NOTE: translated from Chinese
        model = trainer.model
        model.eval()
        rec_dir = os.path.join(run_dir, "eval_pred")
        os.makedirs(rec_dir, exist_ok=True)
        # NOTE: translated from Chinese
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
        else:
            rank = 0
            world = 1
        out_path = eval_shard_path(rec_dir, epoch_tag, rank, "jsonl")
        tp = fp = tn = fn = 0
        task_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        step_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        total_generation_time = 0.0  # NOTE: translated from Chinese
        generation_count = 0  # NOTE: translated from Chinese
        predict_steps_enabled = getattr(args, "predict_steps", 0) > 0
        future_edit_sum = 0.0
        future_edit_count = 0
        eval_round_k = max(0, int(getattr(args, "eval_round_k", 0) or 0))

        # NOTE: translated from Chinese
        _vocab_id_to_name = getattr(eval_dataset, "_vocab_id_to_name", {})
        _all_task_names = set()
        _all_step_names = set(_vocab_id_to_name.values())
        # NOTE: translated from Chinese
        if hasattr(eval_dataset, "video_task_names"):
            _all_task_names.update(eval_dataset.video_task_names.values())
        if hasattr(eval_dataset, "task_lookup"):
            _all_task_names.update(eval_dataset.task_lookup.values())
        belief_postprocessor = None
        if bool(getattr(args, "belief_state_eval", False)):
            belief_postprocessor = BeliefStatePostProcessor(
                annotation_path=getattr(args, "annotation", None),
                known_tasks=_all_task_names,
                known_steps=_all_step_names,
                task_window=int(getattr(args, "belief_task_history_window", 6) or 6),
                step_window=int(getattr(args, "belief_step_history_window", 4) or 4),
                future_topk=int(getattr(args, "belief_future_topk", 5) or 5),
            )
        
        # NOTE: translated from Chinese (debug)
        if rank == 0:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"[Eval] 候选 task 数量: {len(_all_task_names)}, 候选 step 数量: {len(_all_step_names)}")
                if len(_all_task_names) > 0:
                    print(f"  示例 task: {list(_all_task_names)[:5]}")
                if len(_all_step_names) > 0:
                    print(f"  示例 step: {list(_all_step_names)[:5]}")

        def _normalize_text(text: str) -> str:
            """归一化文本用于匹配"""
            return "".join(text.lower().split())

        def _fuzzy_match(pred_text: str, candidate_set: Set[str], threshold: float = 0.9) -> Optional[str]:
            """
            使用相似度匹配预测文本和候选集合
            返回匹配度最高的候选（如果相似度 >= threshold），否则返回 None
            """
            if not pred_text or not candidate_set:
                return None
            
            import difflib
            pred_norm = _normalize_text(pred_text)
            best_match = None
            best_ratio = 0.0
            
            for candidate in candidate_set:
                if not candidate:
                    continue
                candidate_norm = _normalize_text(candidate)
                # NOTE: translated from Chinese
                ratio = difflib.SequenceMatcher(None, pred_norm, candidate_norm).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = candidate
            
            if best_ratio >= threshold:
                return best_match
            return None
        
        with open(out_path, "w", encoding="utf-8") as wf:
            # NOTE: translated from Chinese
            wf.flush()
            try:
                os.fsync(wf.fileno())
            except Exception:
                pass
            rows_written = 0
            from tqdm import tqdm
            # NOTE: translated from Chinese
            try:
                local_indices = list(range(rank, len(eval_dataset), world))
                pbar = tqdm(
                    total=len(local_indices),
                    desc=f"Eval generate r{rank}",
                    ncols=0,
                    position=rank,
                    leave=True,
                    dynamic_ncols=True,
                )

                def _process_eval_index(i: int) -> None:
                    nonlocal tp, fp, tn, fn
                    nonlocal total_generation_time, generation_count
                    nonlocal future_edit_sum, future_edit_count
                    nonlocal rows_written
                    ex = eval_dataset[i]
                    # NOTE: translated from Chinese
                    try:
                        if "prompt_input_ids" in ex and isinstance(ex["prompt_input_ids"], torch.Tensor):
                            prompt_ids_text = processor.batch_decode(ex["prompt_input_ids"].unsqueeze(0), skip_special_tokens=False)[0]
                        else:
                            prompt_ids_text = ""
                    except Exception:
                        prompt_ids_text = ""

                    generation_start_time = time.time()
                    include_scores = bool(getattr(eval_dataset, "enable_scores", False) or getattr(eval_dataset, "if_score", False))
                    normalized_text, pred_is, pred_task, pred_step, pred_future_steps, _ = predict_trigger_with_logits(
                        model,
                        processor,
                        ex,
                        include_scores=include_scores,
                        priority_scores=getattr(eval_dataset, "priority_scores", {}),
                        missing_priority_cache=getattr(eval_dataset, "_missing_priority_tasks", None),
                    )
                    generation_time = time.time() - generation_start_time
                    total_generation_time += generation_time
                    generation_count += 1

                    gt = eval_dataset.targets[i] if hasattr(eval_dataset, "targets") and i < len(eval_dataset.targets) else 0
                    gt_task = eval_dataset.gt_tasks[i] if hasattr(eval_dataset, "gt_tasks") and i < len(eval_dataset.gt_tasks) else ""
                    gt_step = eval_dataset.gt_steps[i] if hasattr(eval_dataset, "gt_steps") and i < len(eval_dataset.gt_steps) else ""
                    gt_future = eval_dataset.gt_future_steps[i] if hasattr(eval_dataset, "gt_future_steps") and i < len(eval_dataset.gt_future_steps) else []
                    if pred_is == 1 and gt == 1: tp += 1
                    elif pred_is == 1 and gt == 0: fp += 1
                    elif pred_is == 0 and gt == 0: tn += 1
                    else: fn += 1

                    current_video_id = ""
                    try:
                        if hasattr(eval_dataset, "data") and i < len(eval_dataset.data):
                            current_item = eval_dataset.data[i]
                            if isinstance(current_item, dict):
                                current_video_id = str(current_item.get("video_id", "") or "")
                    except Exception:
                        current_video_id = ""

                    # NOTE: translated from Chinese (predict)
                    matched_pred_task = pred_task
                    matched_pred_step = pred_step
                    belief_completed_steps = []
                    
                    if pred_task and _all_task_names:
                        matched = _fuzzy_match(pred_task, _all_task_names, threshold=0.9)
                        if matched:
                            matched_pred_task = matched
                    
                    if pred_step and _all_step_names:
                        matched = _fuzzy_match(pred_step, _all_step_names, threshold=0.9)
                        if matched:
                            matched_pred_step = matched

                    if belief_postprocessor is not None:
                        belief_state = belief_postprocessor.refine(
                            video_id=current_video_id,
                            pred_is=bool(pred_is),
                            pred_task=matched_pred_task,
                            pred_step=matched_pred_step,
                            pred_future_steps=pred_future_steps,
                        )
                        matched_pred_task = belief_state.get("task", matched_pred_task)
                        matched_pred_step = belief_state.get("step", matched_pred_step)
                        pred_future_steps = belief_state.get("future_steps", pred_future_steps)
                        belief_completed_steps = belief_postprocessor.get_completed_steps(current_video_id)
                    
                    def _norm_label(val: str) -> str:
                        return (val or "").strip().lower()

                    # NOTE: translated from Chinese
                    p_task_str = _norm_label(matched_pred_task)
                    g_task_str = _norm_label(gt_task)
                    p_step_str = _norm_label(matched_pred_step)
                    g_step_str = _norm_label(gt_step)

                    if gt == 1:
                        if pred_is == 1:
                            if g_task_str:
                                if p_task_str == g_task_str:
                                    task_stats[g_task_str]["tp"] += 1
                                else:
                                    task_stats[g_task_str]["fn"] += 1
                                    if p_task_str:
                                        task_stats[p_task_str]["fp"] += 1
                            elif p_task_str:
                                task_stats[p_task_str]["fp"] += 1
                        else:
                            if g_task_str:
                                task_stats[g_task_str]["fn"] += 1
                    elif pred_is == 1 and p_task_str:
                        task_stats[p_task_str]["fp"] += 1

                    if gt == 1:
                        if pred_is == 1:
                            if g_step_str:
                                if p_step_str == g_step_str:
                                    step_stats[g_step_str]["tp"] += 1
                                else:
                                    step_stats[g_step_str]["fn"] += 1
                                    if p_step_str:
                                        step_stats[p_step_str]["fp"] += 1
                            elif p_step_str:
                                step_stats[p_step_str]["fp"] += 1
                        else:
                            if g_step_str:
                                step_stats[g_step_str]["fn"] += 1
                    elif pred_is == 1 and p_step_str:
                        step_stats[p_step_str]["fp"] += 1
                    
                    if predict_steps_enabled:
                        future_edit_sum += compute_edit_distance(pred_future_steps, gt_future or [])
                        future_edit_count += 1
                    
                    # NOTE: translated from Chinese
                    record = {
                        "idx": i,
                        "video_id": current_video_id,
                        "pred_text": normalized_text,
                        "raw_pred_text": normalized_text,
                        "pred": pred_is,
                        "gt": gt,
                        "pred_task": pred_task,
                        "pred_task_matched": matched_pred_task,  # NOTE: translated from Chinese (added)
                        "gt_task": gt_task,
                        "pred_step": pred_step,
                        "pred_step_matched": matched_pred_step,  # NOTE: translated from Chinese (added)
                        "gt_step": gt_step,
                        "pred_future_steps": pred_future_steps,
                        "gt_future_steps": gt_future,
                        "generation_time": generation_time,
                        "belief_state_used": bool(belief_postprocessor is not None),
                        "belief_completed_steps": belief_completed_steps,
                    }
                    
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows_written += 1
                    if rows_written % 50 == 0:
                        wf.flush()
                    pbar.update(1)

                if eval_round_k > 0 and world > 1:
                    max_local_count = (len(eval_dataset) + world - 1) // world
                    max_rounds = (max_local_count + eval_round_k - 1) // eval_round_k
                    for round_id in range(max_rounds):
                        s = round_id * eval_round_k
                        for i in local_indices[s : s + eval_round_k]:
                            try:
                                _process_eval_index(i)
                            except Exception as e:
                                print(
                                    f"[WARN] eval_only sample failed: epoch={epoch_tag}, rank={rank}, idx={i}, "
                                    f"round={round_id}, err={e}"
                                )
                                continue
                        _safe_dist_barrier(f"eval_only_round_sync_{epoch_tag}_{round_id}")
                else:
                    for i in local_indices:
                        try:
                            _process_eval_index(i)
                        except Exception as e:
                            print(f"[WARN] eval_only sample failed: epoch={epoch_tag}, rank={rank}, idx={i}, err={e}")
                            continue
                pbar.close()
            finally:
                wf.flush()
                try:
                    os.fsync(wf.fileno())
                except Exception:
                    pass

        # Align eval_only with training-time generation eval flow:
        # rely on distributed barrier instead of filesystem polling.
        _safe_dist_barrier("eval_only_post_gen")

        pos_prec = tp / (tp + fp + 1e-9)
        pos_recall = tp / (tp + fn + 1e-9)
        neg_prec = tn / (tn + fn + 1e-9)
        neg_recall = tn / (tn + fp + 1e-9)
        val_acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        f1_score = 2 * (pos_prec * pos_recall) / (pos_prec + pos_recall + 1e-9)

        def compute_macro_metrics(stats_dict):
            precs = []
            recs = []
            for cls_name, counts in stats_dict.items():
                if not cls_name:
                    continue
                tp_cls = counts["tp"]
                fp_cls = counts["fp"]
                fn_cls = counts["fn"]
                if tp_cls + fp_cls > 0:
                    precs.append(tp_cls / (tp_cls + fp_cls))
                if tp_cls + fn_cls > 0:
                    recs.append(tp_cls / (tp_cls + fn_cls))
            m_prec = sum(precs) / len(precs) if precs else 0.0
            m_rec = sum(recs) / len(recs) if recs else 0.0
            return m_prec, m_rec

        task_mprec, task_mrec = compute_macro_metrics(task_stats)
        step_mprec, step_mrec = compute_macro_metrics(step_stats)
        future_edit_avg = future_edit_sum / max(1, future_edit_count) if predict_steps_enabled else 0.0
        # NOTE: translated from Chinese
        avg_generation_time = total_generation_time / max(1, generation_count)
        
        # NOTE: translated from Chinese
        if local_rank == 0:
            csv_path = os.path.join(run_dir, "metrics.csv")
            header = "epoch,train_loss,val_loss,val_acc,f1_score,pos_prec,pos_recall,neg_prec,neg_recall,task_mPrec,task_mRec,step_mPrec,step_mRec,avg_generation_time"
            if predict_steps_enabled:
                header += ",future_edit_dist"
            header += "\n"
            val_loss_str = ""
            line = (
                f"{epoch_tag},,{val_loss_str},{val_acc:.6f},{f1_score:.6f},{pos_prec:.6f},"
                f"{pos_recall:.6f},{neg_prec:.6f},{neg_recall:.6f},{task_mprec:.6f},{task_mrec:.6f},"
                f"{step_mprec:.6f},{step_mrec:.6f},{avg_generation_time:.6f}"
            )
            if predict_steps_enabled:
                line += f",{future_edit_avg:.6f}"
            line += "\n"
            if not os.path.exists(csv_path):
                with open(csv_path, "w", encoding="utf-8") as wf:
                    wf.write(header)
                    wf.write(line)
            else:
                with open(csv_path, "a", encoding="utf-8") as wf:
                    wf.write(line)
        
        _safe_dist_barrier("eval_only_merge")
        if merge_global_metrics is not None and ((not dist.is_available()) or (not dist.is_initialized()) or rank == 0):
            try:
                merge_global_metrics(run_dir, getattr(args, "predict_steps", 0))
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"[Eval] 全局指标已合并并写入 metrics.csv")
            except Exception as e:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"[Eval] merge_global_metrics 失败: {e}")

        _run_scheme_c_onestep_eval(run_dir, epoch_tag, rank, world, args, eval_dataset)
        if local_rank == 0:
            _restore_onestep_metrics_columns(run_dir)
        
        if rank == 0:
            print(f"评估结果已保存到: {out_path}")
            print(
                "指标: "
                f"val_acc={val_acc:.4f}, F1={f1_score:.4f}, pos_prec={pos_prec:.4f}, pos_recall={pos_recall:.4f}, "
                f"task_mPrec={task_mprec:.4f}, task_mRec={task_mrec:.4f}, "
                f"step_mPrec={step_mprec:.4f}, step_mRec={step_mrec:.4f}, avg_generation_time={avg_generation_time:.4f}s"
                + (f", future_edit_dist={future_edit_avg:.4f}" if predict_steps_enabled else "")
            )

    if not args.eval_only:
        # NOTE: translated from Chinese
        try:
            if args.train_jump:
                eval_every_n = max(1, int(getattr(args, "eval_every_n_epochs", 1) or 1))
                resume_epoch = _extract_epoch_from_path(args.resume_from_checkpoint) or 0
                epoch_ckpt_map = _collect_epoch_checkpoint_map(run_dir)
                jump_resume = args.resume_from_checkpoint
                jump_epoch = resume_epoch
                upper_epoch = int(args.num_train_epochs)
                while (jump_epoch + 1) <= upper_epoch and (jump_epoch + 1) in epoch_ckpt_map:
                    next_epoch = jump_epoch + 1
                    next_ckpt = epoch_ckpt_map[next_epoch]
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    if not silence and is_main:
                        print(f"[train_jump] 跳过 epoch_{next_epoch} 训练（已存在 checkpoint: {next_ckpt}）")
                    if next_epoch % eval_every_n == 0:
                        if not silence and is_main:
                            print(f"[train_jump] 使用 epoch_{next_epoch} checkpoint 执行评估（加载 LoRA）...")
                        _load_lora_checkpoint_for_eval(next_ckpt)
                        _run_generation_eval(f"epoch_{next_epoch}")
                    jump_resume = next_ckpt
                    jump_epoch = next_epoch

                if jump_epoch >= upper_epoch:
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    if not silence and is_main:
                        print("[train_jump] 目标 epoch 均已存在，训练阶段全部跳过。")
                else:
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    if not silence and is_main:
                        print(f"[train_jump] 从 {jump_resume} 继续训练（起点接近 epoch_{jump_epoch}）")
                    trainer.train(resume_from_checkpoint=jump_resume)
            elif args.resume_from_checkpoint:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"从 checkpoint 继续训练: {args.resume_from_checkpoint}")
                trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            else:
                trainer.train()
        except Exception as e:
            silence, is_main, world_size, local_rank = _get_env_silence_and_rank()
            if not (silence and not is_main):
                print("\n" + "=" * 96, flush=True)
                print(
                    f"[FATAL][trainer.train] rank={local_rank}/{world_size} "
                    f"exception={repr(e)}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                print("=" * 96 + "\n", flush=True)
            raise
        # NOTE: translated from Chinese
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print("训练完成！评估结果已通过 EpochMetricsCallback 在每个 epoch 结束时记录。")
        # NOTE: translated from Chinese
        if args.resume_from_checkpoint:
            eval_dir = os.path.join(run_dir, "eval_pred")
            if trainer.state.epoch is not None:
                expected_epoch = int(round(float(trainer.state.epoch)))
            else:
                expected_epoch = int(args.num_train_epochs)
            need_final_eval = False
            if local_rank == 0:
                has_eval = False
                if os.path.isdir(eval_dir):
                    prefix = f"epoch_{expected_epoch}"
                    has_eval = any(name.startswith(prefix) for name in os.listdir(eval_dir))
                need_final_eval = not has_eval
                if need_final_eval:
                    silence, is_main, _, _ = _get_env_silence_and_rank()
                    if not silence and is_main:
                        print(f"[Resume] 未发现 epoch_{expected_epoch} 评估文件，补做一次生成评估。")
            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor(1 if need_final_eval else 0, device=torch.device("cuda", local_rank) if torch.cuda.is_available() else "cpu")
                dist.broadcast(flag, src=0)
                need_final_eval = bool(flag.item())
            if need_final_eval:
                _run_generation_eval(str(expected_epoch))
        # NOTE: translated from Chinese
        if local_rank == 0 and merge_global_metrics is not None:
            try:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print("开始合并各 rank 的全局评估指标...")
                merge_global_metrics(run_dir, getattr(args, "predict_steps", 0))
                _restore_onestep_metrics_columns(run_dir)
            except Exception as e:
                silence, is_main, _, _ = _get_env_silence_and_rank()
                if not silence and is_main:
                    print(f"⚠️ 合并全局指标失败: {e}")
    else:
        # NOTE: translated from Chinese
        if args.checkpoint_path:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"正在加载 checkpoint: {args.checkpoint_path}")
            _load_lora_checkpoint_for_eval(args.checkpoint_path)
        else:
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print("警告：--eval_only 模式下未指定 --checkpoint_path，使用当前模型")
        
        # NOTE: translated from Chinese
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print("eval_only 模式：跳过 evaluate，直接开始生成评估...")

        _run_generation_eval("eval_only")
    # NOTE: translated from Chinese
    if not os.path.exists(os.path.join(run_dir, "adapter_model.safetensors")):
        trainer.model.save_pretrained(run_dir)
    if not os.path.exists(os.path.join(run_dir, "tokenizer_config.json")): # NOTE: translated from Chinese (config)
        processor.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "lora_rank": args.lora_rank,
            "bf16": args.bf16,
            "load_in_4bit": args.load_in_4bit,
            "trigger_loss_weight": args.trigger_loss_weight,
        }, f, ensure_ascii=False, indent=2)

    if local_rank == 0:
        try:
            if os.path.exists(marker_path):
                os.remove(marker_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
