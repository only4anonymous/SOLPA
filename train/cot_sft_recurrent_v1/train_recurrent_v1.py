from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from checkpoint_compat import remap_legacy_lora_key
from jump_resume_utils import (
    checkpoint_epoch,
    collect_checkpoint_map,
    discover_epochs_needing_metrics,
    resolve_resume_checkpoint,
)
from recurrent_chunking import build_video_chunks
from recurrent_memory import (
    RecurrentActionMemory,
    inject_memory_token_embeddings,
    pad_action_sequences,
)
from recurrent_pipeline_utils import (
    build_generation_eval_record,
    hydrate_samples_for_chunking,
)
from train.cot_sft_v2.train_cot_sft_two_stage import (
    LazySlidingWindowDataset,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    _get_env_silence_and_rank,
    build_history_memory_text,
    build_run_name_from_args,
    build_two_stage_conversation_bundle,
    collect_completed_steps_before_current_segment,
    collect_future_actions,
    make_collate_fn,
    teacher_next_action,
)
from train.l2.unified_metrics import compute_all_metrics
from two_stage_prompting import (
    format_decision_output_two_stage,
    format_state_output_two_stage,
)


def build_recurrent_run_name(args: argparse.Namespace) -> str:
    class _ArgsView:
        def __init__(self, ns: argparse.Namespace) -> None:
            self.model_name = ns.model_name
            self.window_size = ns.window_size
            self.window_stride = ns.window_stride
            self.num_train_epochs = ns.num_train_epochs
            self.learning_rate = ns.learning_rate
            self.per_device_train_batch_size = ns.per_device_train_batch_size
            self.per_device_eval_batch_size = ns.per_device_eval_batch_size
            self.gradient_accumulation_steps = ns.gradient_accumulation_steps
            self.lora_rank = ns.lora_rank
            self.if_score = False
            self.reasoning = False
            self.load_in_4bit = ns.load_in_4bit
            self.use_trigger_hints = False
            self.belief_state_eval = False
            self.history_memory_mode = "none"
            self.experiment_tag = ns.experiment_tag

    base = build_run_name_from_args(_ArgsView(args))
    name = (
        f"{base}_recurrent_chunk{args.chunk_size}_hw{args.history_window}"
        f"_mh{args.memory_hidden_size}_ms{args.memory_slots}"
    )
    cdim = getattr(args, "commitment_dim", 0)
    cmode = getattr(args, "commitment_mode", "learned")
    if cdim > 0:
        name += f"_cb{cdim}_{cmode}"
    elif cmode == "dropout":
        cdrop = getattr(args, "commitment_dropout", 0.5)
        name += f"_cbdrop{cdrop}"
    if getattr(args, "enable_action_memory", False):
        name += f"_actmem{getattr(args, 'action_recent_k', 4)}"
    return name


def build_memory_stub(memory_slot_tokens: Sequence[str]) -> str:
    return "Latent human-action memory:\n" + " ".join(memory_slot_tokens)


def _collect_recurrent_lazy_labels(base_dataset: LazySlidingWindowDataset) -> List[Dict[str, Any]]:
    silence, is_main, _, local_rank = _get_env_silence_and_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = local_rank
    total = len(base_dataset.samples_meta)
    ready_path = Path(f"{base_dataset._lazy_label_cache_prefix}.recurrent_ready.json")

    def _cache_missing_indices() -> List[int]:
        missing: List[int] = []
        for idx, meta in enumerate(base_dataset.samples_meta):
            cached = base_dataset._lazy_label_cache.get(idx)
            if cached is None or any(
                key not in cached for key in ("completed_steps", "future_steps", "next_action")
            ):
                missing.append(idx)
        return missing

    def _materialize_labels() -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for idx, meta in enumerate(base_dataset.samples_meta):
            cached_labels = base_dataset._lazy_label_cache.get(idx) or {}
            is_trigger = bool(meta.get("is_trigger", False))
            completed_steps = list(cached_labels.get("completed_steps", []) or [])
            future_steps_list = list(cached_labels.get("future_steps", []) or [])
            next_action = str(cached_labels.get("next_action", "") or "")
            if idx < len(base_dataset.gt_future_steps):
                base_dataset.gt_future_steps[idx] = future_steps_list if is_trigger else []
            if idx < len(base_dataset.gt_next_actions):
                base_dataset.gt_next_actions[idx] = next_action if is_trigger else ""
            labels.append(
                {
                    "completed_steps": completed_steps,
                    "future_steps": future_steps_list,
                    "next_action": next_action,
                }
            )
        return labels

    base_dataset._load_lazy_label_cache()
    missing_indices = _cache_missing_indices()
    if not missing_indices and ready_path.exists():
        if not silence and is_main:
            print(f"[recurrent-cache] lazy labels ready from cache: {total}/{total}")
        return _materialize_labels()

    # In multi-rank runs, let rank0 precompute once and let the other ranks reuse the cache.
    if world_size > 1 and rank != 0:
        wait_start = time.time()
        last_report = 0.0
        while True:
            base_dataset._load_lazy_label_cache()
            missing_indices = _cache_missing_indices()
            if not missing_indices and ready_path.exists():
                if not silence and is_main:
                    print(f"[recurrent-cache] cache loaded after wait: {total}/{total}")
                return _materialize_labels()
            elapsed = time.time() - wait_start
            if elapsed >= 7200:
                if not silence:
                    print(
                        f"[recurrent-cache] rank{rank} waited {elapsed:.0f}s; "
                        "cache still incomplete, falling back to local compute."
                    )
                break
            if (not silence) and elapsed - last_report >= 60:
                ready_flag = "yes" if ready_path.exists() else "no"
                print(
                    f"[recurrent-cache] rank{rank} waiting for rank0 cache: "
                    f"ready={ready_flag} filled={total - len(missing_indices)}/{total}"
                )
                last_report = elapsed
            time.sleep(5)

    if rank == 0 and ready_path.exists():
        try:
            ready_path.unlink()
        except OSError:
            pass

    if not silence and is_main:
        print(
            f"[recurrent-cache] building lazy labels locally: "
            f"missing={len(missing_indices)}/{total}"
        )

    for offset, idx in enumerate(missing_indices, start=1):
        meta = base_dataset.samples_meta[idx]
        is_trigger = bool(meta.get("is_trigger", False))
        frame_labels = meta.get("frame_labels", []) or []
        end = int(meta["end"])
        vid = str(meta["video_id"])
        task_name_output = str(meta.get("task_name", "") or "")
        step_name = str(meta.get("step_name", "") or "")
        completed_steps: List[str] = collect_completed_steps_before_current_segment(
            frame_labels,
            end,
            base_dataset._vocab_id_to_name,
        )
        future_steps_list: List[str] = []
        if is_trigger:
            future_steps_list = collect_future_actions(
                frame_labels=frame_labels,
                current_idx=end,
                predict_steps=base_dataset.predict_steps,
                vocab_map=base_dataset._vocab_id_to_name,
                missing_vocab_labels=base_dataset._missing_vocab_labels,
                video_id=vid,
            )
        next_action = ""
        if is_trigger and base_dataset.predict_next_action:
            next_action = teacher_next_action(
                task_name_output,
                completed_steps,
                step_name,
                future_steps_list,
                base_dataset._annotation_path,
                base_dataset._tg_cache,
            )
        if base_dataset._distill_label_map:
            _cd_distilled = base_dataset._distill_label_map.get(idx)
            if _cd_distilled:
                next_action = _cd_distilled
        cached_labels = {
            "completed_steps": completed_steps,
            "future_steps": future_steps_list if is_trigger else [],
            "next_action": next_action if is_trigger else "",
        }
        base_dataset._lazy_label_cache[idx] = dict(cached_labels)
        base_dataset._lazy_label_cache_updates[idx] = dict(cached_labels)
        if (not silence) and is_main and (offset == 1 or offset % 5000 == 0 or offset == len(missing_indices)):
            print(
                f"[recurrent-cache] built {offset}/{len(missing_indices)} "
                f"(filled={total - len(missing_indices) + offset}/{total})"
            )

    base_dataset._flush_lazy_label_cache(force=True)
    ready_path.write_text(json.dumps({"count": total, "updated_at": time.time()}), encoding="utf-8")
    return _materialize_labels()


def _safe_dist_barrier(tag: str = "") -> None:
    try:
        if not (dist.is_available() and dist.is_initialized()):
            return
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    except Exception as exc:
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"[WARN] barrier{f'({tag})' if tag else ''} failed: {exc}")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _extract_tagged_span(text: str, left: str, right: str) -> str:
    if not text:
        return ""
    pattern = re.escape(left) + r"(.*?)" + re.escape(right)
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else ""


def _parse_trigger_value(text: str) -> int:
    trigger_text = _extract_tagged_span(text, "<|trigger_start|>", "<|trigger_end|>").strip().lower()
    return 1 if trigger_text in {"true", "1", "yes"} else 0


def _parse_future_steps(text: str) -> List[str]:
    raw = _extract_tagged_span(text, "<|future_steps_start|>", "<|future_steps_end|>")
    if not raw:
        return []
    out: List[str] = []
    for seg in re.split(r"[;\n]+", raw):
        cleaned = seg.strip()
        if cleaned:
            out.append(cleaned)
    return out


def _normalize_text(text: str) -> str:
    """Lowercase + strip whitespace for fuzzy matching."""
    return "".join((text or "").lower().split())


def _fuzzy_match_step(
    pred_text: str,
    candidates: List[str],
    threshold: float = 0.8,
) -> Optional[str]:
    """Return the best-matching candidate if similarity >= threshold, else None."""
    if not pred_text or not candidates:
        return None
    import difflib
    pred_norm = _normalize_text(pred_text)
    best: Optional[str] = None
    best_ratio = 0.0
    for cand in candidates:
        c_norm = _normalize_text(cand)
        if not c_norm:
            continue
        r = difflib.SequenceMatcher(None, pred_norm, c_norm).ratio()
        if r > best_ratio:
            best_ratio = r
            best = cand
    return best if best_ratio >= threshold else None


def _load_task_to_canonical_steps(annotation_path: str) -> Dict[str, List[str]]:
    """
    Load task -> canonical step names from task_to_observed_steps.json
    (preferred) or fall back to taxonomy in all_annotations.json.
    """
    # 1) Try the curated observed-steps mapping
    observed_path = Path(__file__).resolve().parents[2] / "test" / "l2" / "task_mapping" / "task_to_observed_steps.json"
    if observed_path.exists():
        with observed_path.open("r", encoding="utf-8") as f:
            mapping: Dict[str, List[str]] = json.load(f)
        return mapping

    # 2) Fall back to annotation taxonomy
    if not annotation_path or not Path(annotation_path).exists():
        return {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    vocab = ann.get("vocabulary", {})
    taxonomy = ann.get("taxonomy", {})
    nodes = ann.get("nodes", [])
    midlevel_ids = set()
    terminate_ids = set()
    for n in nodes:
        nid = str(n.get("id", ""))
        if n.get("is_midlevel"):
            midlevel_ids.add(nid)
        if str(n.get("name", "")).lower().strip() in {"terminate", "end"}:
            terminate_ids.add(nid)
    result: Dict[str, List[str]] = {}
    for task_name, task_nodes in taxonomy.items():
        steps: List[str] = []
        for nid_str, node_info in task_nodes.items():
            if nid_str == "0":
                continue
            if nid_str in midlevel_ids or nid_str in terminate_ids:
                continue
            name = str(node_info.get("name", "")).strip()
            if name:
                steps.append(name)
        if steps:
            result[task_name] = steps
    return result


def _ground_predictions(
    pred_task: str,
    pred_step: str,
    pred_future_steps: List[str],
    pred_next_action: str,
    task_to_steps: Dict[str, List[str]],
    threshold: float = 0.8,
) -> Tuple[str, str, List[str], str]:
    """
    Ground model predictions to canonical step names via fuzzy matching.
    Returns (grounded_task, grounded_step, grounded_future_steps, grounded_next_action).
    """
    if not task_to_steps:
        return pred_task, pred_step, pred_future_steps, pred_next_action

    # 1) Ground task name
    all_task_names = list(task_to_steps.keys())
    grounded_task = _fuzzy_match_step(pred_task, all_task_names, threshold=0.85) or pred_task

    # 2) Get candidate steps for this task
    candidate_steps = task_to_steps.get(grounded_task, [])
    if not candidate_steps:
        # try original pred_task key
        candidate_steps = task_to_steps.get(pred_task, [])
    if not candidate_steps:
        # fall back: collect ALL step names
        all_steps_set: set[str] = set()
        for steps in task_to_steps.values():
            all_steps_set.update(steps)
        candidate_steps = list(all_steps_set)

    # 3) Ground step, future_steps, next_action
    grounded_step = _fuzzy_match_step(pred_step, candidate_steps, threshold) or pred_step

    grounded_future: List[str] = []
    for fs in pred_future_steps:
        matched = _fuzzy_match_step(fs, candidate_steps, threshold)
        grounded_future.append(matched if matched else fs)

    grounded_action = pred_next_action
    if pred_next_action and pred_next_action.strip().lower() not in {"wait / none", "wait/none", "none", "wait"}:
        matched_action = _fuzzy_match_step(pred_next_action, candidate_steps, threshold)
        if matched_action:
            grounded_action = matched_action

    return grounded_task, grounded_step, grounded_future, grounded_action


def _state_prompt_messages(system_prompt: str, state_user_text: str, num_images: int) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": ([{"type": "image"} for _ in range(num_images)] + [{"type": "text", "text": state_user_text}]),
        },
    ]


def _decision_prompt_messages(
    *,
    system_prompt: str,
    state_user_text: str,
    state_formatted: str,
    decision_user_text: str,
    num_images: int,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": ([{"type": "image"} for _ in range(num_images)] + [{"type": "text", "text": state_user_text}]),
        },
        {"role": "assistant", "content": [{"type": "text", "text": state_formatted}]},
        {"role": "user", "content": [{"type": "text", "text": decision_user_text}]},
    ]


def _prepare_generation_inputs(
    processor: Any,
    tokenizer: Any,
    images: List[Image.Image],
    messages_prompt: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[str, int, Dict[str, torch.Tensor]]:
    text_prompt = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = processor(text=[text_prompt], images=[images], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prepared: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "pixel_values" in enc:
        prepared["pixel_values"] = enc["pixel_values"].to(device=device, dtype=torch.bfloat16)
    if "image_grid_thw" in enc:
        prepared["image_grid_thw"] = enc["image_grid_thw"].to(device)
    return text_prompt, int(input_ids.shape[1]), prepared


def _generate_text(
    *,
    model,
    tokenizer: Any,
    prompt_inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    max_new_tokens: int,
    memory_token_ids: Optional[Sequence[int]] = None,
    memory_values: Optional[torch.Tensor] = None,
    prefix_allowed_tokens_fn: Optional[Any] = None,
) -> str:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    with torch.no_grad():
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        if memory_token_ids and memory_values is not None:
            core_model = _unwrap_model(model)
            memory_tensor = memory_values.to(core_model.get_input_embeddings().weight.dtype)
            with inject_memory_token_embeddings(
                core_model.get_input_embeddings(),
                memory_token_ids=list(memory_token_ids),
                memory_values=memory_tensor,
            ):
                with autocast_ctx:
                    outputs = model.generate(
                        **prompt_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    )
        else:
            with autocast_ctx:
                outputs = model.generate(
                    **prompt_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
    tokens = outputs[0, prompt_len:]
    return tokenizer.decode(tokens, skip_special_tokens=False).strip()


def build_branch_sample(
    *,
    processor: Any,
    tokenizer: Any,
    images: List[Image.Image],
    messages_prompt: List[Dict[str, Any]],
    messages_full: List[Dict[str, Any]],
    debug_target_text: str,
) -> Dict[str, Any]:
    text_prompt = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    text_full = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=False,
    )
    enc_full = processor(text=[text_full], images=[images], return_tensors="pt")
    enc_prompt = processor(text=[text_prompt], images=[images], return_tensors="pt")
    input_ids = enc_full["input_ids"].squeeze(0)
    prompt_len = int(enc_prompt["input_ids"].shape[1])
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    item: Dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        "debug_prompt_text": text_prompt,
        "debug_target_text": debug_target_text,
    }
    if "pixel_values" in enc_full:
        item["pixel_values"] = enc_full["pixel_values"].squeeze(0).to(torch.bfloat16)
    if "image_grid_thw" in enc_full:
        grid = enc_full["image_grid_thw"]
        if isinstance(grid, torch.Tensor):
            if grid.dim() >= 3:
                grid = grid.squeeze(0)
            grid = grid.view(-1, 3)
        item["image_grid_thw"] = grid
    return item


class RecurrentChunkDataset(Dataset):
    def __init__(
        self,
        *,
        jsonl_path: str,
        model_name: str,
        processor: Any,
        tokenizer: Any,
        frame_root: str,
        annotation_path: str,
        chunk_size: int,
        history_window: int,
        memory_slot_tokens: Sequence[str],
        window_size: int,
        window_stride: int,
        predict_steps: int,
        predict_next_action: bool,
        max_image_long_edge: int,
        enable_action_memory: bool = False,
        action_recent_k: int = 4,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.history_window = int(history_window)
        self.predict_steps = int(predict_steps)
        self.max_image_long_edge = int(max_image_long_edge)
        self.memory_slot_tokens = list(memory_slot_tokens)
        self.enable_action_memory = bool(enable_action_memory)
        self.action_recent_k = int(action_recent_k)
        self.base_dataset = LazySlidingWindowDataset(
            jsonl_path=jsonl_path,
            window_size=window_size,
            window_stride=window_stride,
            use_trigger_hints=False,
            trigger_json_path=None,
            tokenizer=tokenizer,
            if_score=False,
            frame_root=frame_root,
            annotation_path=annotation_path,
            processor=processor,
            record_path=None,
            preprocessed_data_dir=None,
            preprocessed_data_file=None,
            preprocessed_train_file=None,
            preprocessed_val_file=None,
            priority_score_path=None,
            enable_evidence_frames=False,
            enable_reasoning=False,
            enable_confidence=False,
            enable_scores=False,
            use_reasoning_tokens=False,
            random_seed=42,
            max_image_long_edge=max_image_long_edge,
            predict_steps=predict_steps,
            predict_next_action=predict_next_action,
            history_memory_mode="none",
            history_recent_k=4,
        )
        vocab = self.base_dataset._vocab_id_to_name
        self.action_name_to_id: Dict[str, int] = {"<pad>": 0}
        for _, name in sorted(vocab.items()):
            clean = str(name).strip()
            if clean and clean not in self.action_name_to_id:
                self.action_name_to_id[clean] = len(self.action_name_to_id)
        lazy_labels = _collect_recurrent_lazy_labels(self.base_dataset)
        samples_for_chunking = hydrate_samples_for_chunking(
            self.base_dataset.samples_meta,
            lazy_labels,
        )
        for idx, enriched in enumerate(samples_for_chunking):
            enriched["index"] = idx
        self.chunks = build_video_chunks(
            samples_for_chunking,
            chunk_size=self.chunk_size,
            history_window=self.history_window,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def _load_window(self, meta: Dict[str, Any]) -> tuple[List[Image.Image], List[str]]:
        images: List[Image.Image] = []
        frame_descs: List[str] = []
        window_files = meta["window_files"]
        if not window_files:
            raise RuntimeError(f"未找到任何帧文件: video_id={meta['video_id']}, idx={meta.get('index')}")
        idx0 = window_files[0][1]
        for j, (pth, idx_int) in enumerate(window_files):
            t = (idx_int - idx0) / 25.0
            frame_descs.append(f"F{j} [idx={idx_int} t={t:.2f}s]")
            images.append(self.base_dataset._load_image_cached(pth))
        return images, frame_descs

    def _action_ids(self, actions: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for action in actions:
            clean = str(action).strip()
            if not clean:
                continue
            ids.append(self.action_name_to_id.get(clean, 0))
        return ids

    def _build_step_bundle(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        images, frame_descs = self._load_window(meta)
        base_bundle = build_two_stage_conversation_bundle(
            video_id=meta["video_id"],
            frame_descs=frame_descs,
            images=images,
            completed_steps=[],
            is_trigger=bool(meta["is_trigger"]),
            task_name=str(meta.get("task_name", "") or ""),
            step_name=str(meta.get("step_name", "") or ""),
            future_steps=list(meta.get("future_steps", []) or []),
            next_action=str(meta.get("next_action", "") or ""),
            predict_steps=self.predict_steps,
            predict_next_action=True,
            history_mode="none",
            history_recent_k=4,
        )
        system_prompt = base_bundle["system_prompt_text"]
        state_messages_prompt = _state_prompt_messages(
            system_prompt=system_prompt,
            state_user_text=base_bundle["state_user_text"],
            num_images=len(images),
        )
        state_messages_full = state_messages_prompt + [
            {"role": "assistant", "content": [{"type": "text", "text": base_bundle["state_target"]}]},
        ]
        state_sample = build_branch_sample(
            processor=self.processor,
            tokenizer=self.tokenizer,
            images=images,
            messages_prompt=state_messages_prompt,
            messages_full=state_messages_full,
            debug_target_text=base_bundle["state_target"],
        )

        recurrent_decision_text = build_memory_stub(self.memory_slot_tokens)
        if self.enable_action_memory:
            completed = list(meta.get("completed_steps", []) or [])
            action_mem_text = build_history_memory_text(
                completed, mode="recent_only", recent_k=self.action_recent_k,
            )
            if action_mem_text:
                recurrent_decision_text += "\n\n" + action_mem_text
        recurrent_decision_text += "\n\n" + base_bundle["decision_user_text"]
        decision_messages_prompt = state_messages_full + [
            {"role": "user", "content": [{"type": "text", "text": recurrent_decision_text}]},
        ]
        decision_messages_full = decision_messages_prompt + [
            {"role": "assistant", "content": [{"type": "text", "text": base_bundle["decision_target"]}]},
        ]
        decision_sample = build_branch_sample(
            processor=self.processor,
            tokenizer=self.tokenizer,
            images=images,
            messages_prompt=decision_messages_prompt,
            messages_full=decision_messages_full,
            debug_target_text=base_bundle["decision_target"],
        )
        return {
            "state_sample": state_sample,
            "decision_sample": decision_sample,
            "update_actions_after_ids": self._action_ids(meta.get("update_actions_after", [])),
            "completed_steps": list(meta.get("completed_steps", []) or []),
            "meta": dict(meta),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        return {
            "video_id": chunk["video_id"],
            "initial_action_ids": self._action_ids(chunk["initial_completed_steps"]),
            "steps": [self._build_step_bundle(step_meta) for step_meta in chunk["steps"]],
        }


def chunk_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"chunks": batch}


class EveryNEpochCallback(TrainerCallback):
    def __init__(self, n: int) -> None:
        self.n = max(1, int(n))

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch or 0.0
        epoch_int = int(round(epoch))
        if epoch_int <= 0:
            return control
        should_run = (epoch_int % self.n) == 0
        control.should_evaluate = should_run
        control.should_save = should_run
        return control


class RecurrentChunkTrainer(Trainer):
    def __init__(
        self,
        *,
        base_collator,
        memory_token_ids: Sequence[int],
        state_loss_weight: float = 1.0,
        decision_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_collator = base_collator
        self.memory_token_ids = [int(x) for x in memory_token_ids]
        self.state_loss_weight = float(state_loss_weight)
        self.decision_loss_weight = float(decision_loss_weight)

    def _prepare_mm_batch(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device)
        return prepared

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "chunks" not in inputs:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        loss = loss.detach()
        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chunks = inputs["chunks"]
        device = next(model.parameters()).device
        core_model = _unwrap_model(model)
        initial_ids, initial_mask = pad_action_sequences(
            [chunk["initial_action_ids"] for chunk in chunks],
            device=device,
        )
        state = core_model.recurrent_memory.encode_history(initial_ids, initial_mask)
        total_loss = torch.zeros((), device=device)
        total_terms = 0
        max_steps = max((len(chunk["steps"]) for chunk in chunks), default=0)

        for step_idx in range(max_steps):
            active_indices = [i for i, chunk in enumerate(chunks) if step_idx < len(chunk["steps"])]
            if not active_indices:
                continue
            step_group = [chunks[i]["steps"][step_idx] for i in active_indices]

            state_batch = self.base_collator([entry["state_sample"] for entry in step_group])
            state_inputs = self._prepare_mm_batch(state_batch, device)
            state_outputs = model(**state_inputs)
            total_loss = total_loss + self.state_loss_weight * state_outputs.loss
            total_terms += 1

            decision_batch = self.base_collator([entry["decision_sample"] for entry in step_group])
            decision_inputs = self._prepare_mm_batch(decision_batch, device)
            active_state = state[active_indices]
            memory_values = core_model.recurrent_memory.project_memory_tokens(active_state)
            memory_values = memory_values.to(core_model.get_input_embeddings().weight.dtype)
            with inject_memory_token_embeddings(
                core_model.get_input_embeddings(),
                memory_token_ids=self.memory_token_ids,
                memory_values=memory_values,
            ):
                decision_outputs = model(**decision_inputs)
            total_loss = total_loss + self.decision_loss_weight * decision_outputs.loss
            total_terms += 1

            update_ids, update_mask = pad_action_sequences(
                [entry["update_actions_after_ids"] for entry in step_group],
                device=device,
            )
            next_state = core_model.recurrent_memory.update_state(active_state, update_ids, update_mask)
            state = state.clone()
            state[active_indices] = next_state

        if total_terms == 0:
            raise RuntimeError("空 chunk batch，无法计算 recurrent loss")
        total_loss = total_loss / float(total_terms)
        return (total_loss, {"loss": total_loss}) if return_outputs else total_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--frame_root", type=str, required=True)
    parser.add_argument("--annotation", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--window_stride", type=int, default=3)
    parser.add_argument("--predict_steps", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--history_window", type=int, default=4)
    parser.add_argument("--memory_hidden_size", type=int, default=512)
    parser.add_argument("--memory_slots", type=int, default=2)
    parser.add_argument("--max_image_long_edge", type=int, default=896)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_every_n_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--train_jump", action="store_true")
    parser.add_argument("--onestep_workers", type=int, default=0)
    parser.add_argument("--experiment_tag", type=str, default="recurrentv1")
    parser.add_argument("--disable_find_unused_parameters", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    # --- Commitment Bottleneck ---
    parser.add_argument("--commitment_dim", type=int, default=0,
                        help="Commitment bottleneck dim. 0=disabled (original).")
    parser.add_argument("--commitment_mode", type=str, default="learned",
                        choices=["learned", "random", "dropout"],
                        help="Bottleneck mode: learned | random (fixed proj) | dropout.")
    parser.add_argument("--commitment_dropout", type=float, default=0.5,
                        help="Dropout rate for commitment_mode=dropout control.")
    # --- Action Memory (text) injection ---
    parser.add_argument("--enable_action_memory", action="store_true",
                        help="Inject recent text action memory into decision prompt.")
    parser.add_argument("--action_recent_k", type=int, default=4,
                        help="Number of recent actions for text action memory.")
    # --- Diagnostics ---
    parser.add_argument("--dump_hidden_states", action="store_true",
                        help="Save GRU hidden states during eval for SVD/PCA analysis.")
    return parser.parse_args()


def resolve_ddp_find_unused_parameters(disable_find_unused_parameters: bool) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    return bool(world_size > 1 and not disable_find_unused_parameters)


def _save_recurrent_memory_state(model, checkpoint_dir: str | Path) -> None:
    core_model = _unwrap_model(model)
    target = Path(checkpoint_dir) / "recurrent_memory.safetensors"
    if not hasattr(core_model, "recurrent_memory"):
        return
    state = {
        key: value.detach().cpu()
        for key, value in core_model.recurrent_memory.state_dict().items()
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(target))


def _load_recurrent_memory_state(model, checkpoint_dir: str | Path, *, required: bool) -> bool:
    core_model = _unwrap_model(model)
    path = Path(checkpoint_dir) / "recurrent_memory.safetensors"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"未找到 recurrent_memory 权重: {path}")
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"[resume] recurrent_memory 权重缺失，将使用当前内存模块参数: {path}")
        return False
    load_result = core_model.recurrent_memory.load_state_dict(load_file(str(path)), strict=False)
    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main:
        print(
            f"[resume] loaded recurrent_memory from {path} "
            f"missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )
    return True


def _load_adapter_weights_into_model(model, checkpoint_dir: str | Path) -> None:
    adapter_path = Path(checkpoint_dir) / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"未找到 adapter 权重: {adapter_path}")

    adapter_state = load_file(str(adapter_path))
    load_result = model.load_state_dict(adapter_state, strict=False)
    if len(load_result.unexpected_keys) == len(adapter_state):
        remapped_state: Dict[str, torch.Tensor] = {}
        skipped = 0
        for key, value in adapter_state.items():
            mapped = remap_legacy_lora_key(key)
            if mapped is None:
                skipped += 1
                continue
            remapped_state[mapped] = value
        load_result = model.load_state_dict(remapped_state, strict=False)
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(
                f"[resume] loaded legacy-remapped adapter from {adapter_path} "
                f"missing={len(load_result.missing_keys)} "
                f"unexpected={len(load_result.unexpected_keys)} "
                f"skipped_non_lora={skipped}"
            )
        return

    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main:
        print(
            f"[resume] loaded adapter from {adapter_path} "
            f"missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )


def _load_recurrent_checkpoint_state(model, checkpoint_dir: str | Path, *, require_recurrent: bool) -> None:
    _load_adapter_weights_into_model(model, checkpoint_dir)
    _load_recurrent_memory_state(model, checkpoint_dir, required=require_recurrent)


def _latest_logged_train_loss(trainer_state) -> Optional[float]:
    for entry in reversed(getattr(trainer_state, "log_history", []) or []):
        if "loss" not in entry:
            continue
        try:
            return float(entry["loss"])
        except Exception:
            continue
    return None


def _checkpoint_logged_losses(checkpoint_dir: str | Path, epoch_tag: int) -> Tuple[Optional[float], Optional[float]]:
    trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not trainer_state_path.exists():
        return None, None
    try:
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    for entry in payload.get("log_history", []):
        try:
            entry_epoch = int(round(float(entry.get("epoch", 0.0) or 0.0)))
        except Exception:
            entry_epoch = 0
        if entry_epoch != int(epoch_tag):
            continue
        if "loss" in entry:
            try:
                train_loss = float(entry["loss"])
            except Exception:
                pass
        if "eval_loss" in entry:
            try:
                val_loss = float(entry["eval_loss"])
            except Exception:
                pass
    return train_loss, val_loss


def _eval_shard_jsonl(run_dir: str | Path, epoch_tag: int | str, rank: int) -> Path:
    return Path(run_dir) / "eval_pred" / f"epoch_{epoch_tag}.rank{rank}.jsonl"


def _eval_shard_done(run_dir: str | Path, epoch_tag: int | str, rank: int) -> Path:
    return Path(run_dir) / "eval_pred" / f"epoch_{epoch_tag}.rank{rank}.done"


def _wait_for_eval_shards(run_dir: str | Path, epoch_tag: int | str, world: int) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    start_time = time.time()
    timeout_s = 30 * 60
    while True:
        missing = []
        for shard_rank in range(world):
            if not _eval_shard_jsonl(run_dir, epoch_tag, shard_rank).exists():
                missing.append(str(_eval_shard_jsonl(run_dir, epoch_tag, shard_rank)))
            if not _eval_shard_done(run_dir, epoch_tag, shard_rank).exists():
                missing.append(str(_eval_shard_done(run_dir, epoch_tag, shard_rank)))
        if not missing:
            return
        if time.time() - start_time > timeout_s:
            raise RuntimeError(f"等待 eval shard 超时: epoch={epoch_tag}, sample={missing[:2]}")
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"[eval] waiting shards epoch={epoch_tag}: missing={len(missing)}")
        time.sleep(2.0)


class RecurrentGenerationMetricsRunner:
    def __init__(
        self,
        *,
        trainer: RecurrentChunkTrainer,
        run_dir: str,
        processor: Any,
        tokenizer: Any,
        eval_dataset: RecurrentChunkDataset,
        memory_slot_tokens: Sequence[str],
        memory_token_ids: Sequence[int],
        args: argparse.Namespace,
    ) -> None:
        self.trainer = trainer
        self.run_dir = run_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.memory_slot_tokens = list(memory_slot_tokens)
        self.memory_token_ids = [int(x) for x in memory_token_ids]
        self.args = args
        # Action grounding: load canonical step names for fuzzy matching
        self.task_to_steps = _load_task_to_canonical_steps(
            getattr(args, "annotation", ""),
        )
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            n_tasks = len(self.task_to_steps)
            n_steps = sum(len(v) for v in self.task_to_steps.values())
            print(f"[action-grounding] loaded {n_tasks} tasks, {n_steps} canonical steps")

    def _run_generation_eval(
        self,
        *,
        epoch_tag: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
    ) -> None:
        model = self.trainer.model
        core_model = _unwrap_model(model)
        device = next(model.parameters()).device
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        pred_dir = Path(self.run_dir) / "eval_pred"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_path = _eval_shard_jsonl(self.run_dir, epoch_tag, rank)
        done_path = _eval_shard_done(self.run_dir, epoch_tag, rank)
        for stale in (out_path, done_path):
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass

        local_chunk_indices = list(range(rank, len(self.eval_dataset), world))
        silence, is_main, _, local_rank = _get_env_silence_and_rank()
        if not silence and is_main:
            print(
                f"[recurrent-eval] epoch={epoch_tag} rank={rank}/{world} "
                f"chunks={len(local_chunk_indices)} out={out_path}"
            )

        with out_path.open("w", encoding="utf-8") as handle:
            written = 0
            for chunk_pos, chunk_idx in enumerate(local_chunk_indices, start=1):
                chunk = self.eval_dataset[chunk_idx]
                initial_ids, initial_mask = pad_action_sequences(
                    [chunk["initial_action_ids"]],
                    device=device,
                )
                state = core_model.recurrent_memory.encode_history(initial_ids, initial_mask)

                for step in chunk["steps"]:
                    meta = step["meta"]
                    images, frame_descs = self.eval_dataset._load_window(meta)
                    base_bundle = build_two_stage_conversation_bundle(
                        video_id=meta["video_id"],
                        frame_descs=frame_descs,
                        images=images,
                        completed_steps=[],
                        is_trigger=bool(meta["is_trigger"]),
                        task_name=str(meta.get("task_name", "") or ""),
                        step_name=str(meta.get("step_name", "") or ""),
                        future_steps=list(meta.get("future_steps", []) or []),
                        next_action=str(meta.get("next_action", "") or ""),
                        predict_steps=self.args.predict_steps,
                        predict_next_action=True,
                        history_mode="none",
                        history_recent_k=4,
                    )

                    t0 = time.time()
                    state_messages = _state_prompt_messages(
                        system_prompt=base_bundle["system_prompt_text"],
                        state_user_text=base_bundle["state_user_text"],
                        num_images=len(images),
                    )
                    _, state_prompt_len, state_inputs = _prepare_generation_inputs(
                        processor=self.processor,
                        tokenizer=self.tokenizer,
                        images=images,
                        messages_prompt=state_messages,
                        device=device,
                    )
                    state_text = _generate_text(
                        model=model,
                        tokenizer=self.tokenizer,
                        prompt_inputs=state_inputs,
                        prompt_len=state_prompt_len,
                        max_new_tokens=128,
                    )

                    pred_is = _parse_trigger_value(state_text)
                    pred_task = _extract_tagged_span(state_text, "<|task_start|>", "<|task_end|>") if pred_is else ""
                    pred_step = _extract_tagged_span(state_text, "<|step_start|>", "<|step_end|>") if pred_is else ""
                    state_formatted = format_state_output_two_stage(bool(pred_is), pred_task, pred_step)

                    pred_future_steps: List[str] = []
                    pred_next_action = ""
                    decision_formatted = format_decision_output_two_stage(False, [], "")
                    if pred_is:
                        recurrent_decision_text = build_memory_stub(self.memory_slot_tokens)
                        if getattr(self.args, "enable_action_memory", False):
                            completed = list(meta.get("completed_steps", []) or [])
                            action_mem_text = build_history_memory_text(
                                completed, mode="recent_only",
                                recent_k=getattr(self.args, "action_recent_k", 4),
                            )
                            if action_mem_text:
                                recurrent_decision_text += "\n\n" + action_mem_text
                        recurrent_decision_text += "\n\n" + base_bundle["decision_user_text"]
                        decision_messages = _decision_prompt_messages(
                            system_prompt=base_bundle["system_prompt_text"],
                            state_user_text=base_bundle["state_user_text"],
                            state_formatted=state_formatted,
                            decision_user_text=recurrent_decision_text,
                            num_images=len(images),
                        )
                        _, decision_prompt_len, decision_inputs = _prepare_generation_inputs(
                            processor=self.processor,
                            tokenizer=self.tokenizer,
                            images=images,
                            messages_prompt=decision_messages,
                            device=device,
                        )
                        memory_values = core_model.recurrent_memory.project_memory_tokens(state)
                        decision_text = _generate_text(
                            model=model,
                            tokenizer=self.tokenizer,
                            prompt_inputs=decision_inputs,
                            prompt_len=decision_prompt_len,
                            max_new_tokens=192,
                            memory_token_ids=self.memory_token_ids,
                            memory_values=memory_values,
                        )
                        pred_future_steps = _parse_future_steps(decision_text)
                        pred_next_action = _extract_tagged_span(
                            decision_text,
                            "<|next_action_start|>",
                            "<|next_action_end|>",
                        )
                        decision_formatted = decision_text or format_decision_output_two_stage(
                            True,
                            pred_future_steps,
                            pred_next_action,
                        )

                    generation_time = time.time() - t0
                    # --- Action Grounding: fuzzy match to canonical step names ---
                    grounded_task, grounded_step, grounded_future, grounded_action = _ground_predictions(
                        pred_task=pred_task,
                        pred_step=pred_step,
                        pred_future_steps=pred_future_steps,
                        pred_next_action=pred_next_action,
                        task_to_steps=self.task_to_steps,
                        threshold=0.8,
                    )
                    normalized_text = state_formatted + "\n" + decision_formatted
                    record = build_generation_eval_record(
                        meta=meta,
                        pred_is=bool(pred_is),
                        pred_task=pred_task,
                        pred_task_matched=grounded_task,
                        pred_step=pred_step,
                        pred_step_matched=grounded_step,
                        pred_future_steps=grounded_future,
                        pred_next_action=grounded_action,
                        normalized_text=normalized_text,
                        generation_time=float(generation_time),
                    )
                    # --- Diagnostics: hidden state dump ---
                    if getattr(self.args, "dump_hidden_states", False):
                        hs = state[0].detach().cpu().float()
                        record["hidden_state"] = hs.tolist()
                        record["hidden_state_norm"] = float(hs.norm().item())
                        cv = core_model.recurrent_memory.get_commitment_vector(state)
                        if cv is not None:
                            record["commitment_vector"] = cv[0].detach().cpu().float().tolist()
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

                    update_ids, update_mask = pad_action_sequences(
                        [step["update_actions_after_ids"]],
                        device=device,
                    )
                    state = core_model.recurrent_memory.update_state(state, update_ids, update_mask)

                if not silence and is_main and chunk_pos % 10 == 0:
                    print(
                        f"[recurrent-eval] epoch={epoch_tag} rank={rank}/{world} "
                        f"processed_chunks={chunk_pos}/{len(local_chunk_indices)}"
                    )

            handle.flush()
            os.fsync(handle.fileno())

        done_path.write_text("ok", encoding="utf-8")
        _safe_dist_barrier(f"recurrent_eval_gen_done_{epoch_tag}")
        _wait_for_eval_shards(self.run_dir, epoch_tag, world)
        _safe_dist_barrier(f"recurrent_eval_shards_ready_{epoch_tag}")

        # ---- Unified metrics: classification + onestep planning (APA/Saved/Ent) ----
        compute_all_metrics(
            run_dir=self.run_dir,
            epoch_tag=epoch_tag,
            args=self.args,
            train_loss=train_loss,
            val_loss=val_loss,
            method_name="recurrent",
            barrier_prefix="recurrent",
        )


class RecurrentMetricsCallback(TrainerCallback):
    def __init__(
        self,
        *,
        trainer: RecurrentChunkTrainer,
        run_dir: str,
        processor: Any,
        tokenizer: Any,
        eval_dataset: RecurrentChunkDataset,
        memory_slot_tokens: Sequence[str],
        memory_token_ids: Sequence[int],
        args: argparse.Namespace,
    ) -> None:
        self.trainer = trainer
        self.run_dir = run_dir
        self.args = args
        self.runner = RecurrentGenerationMetricsRunner(
            trainer=trainer,
            run_dir=run_dir,
            processor=processor,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            memory_slot_tokens=memory_slot_tokens,
            memory_token_ids=memory_token_ids,
            args=args,
        )
        self._completed_epochs: set[int] = set()

    def on_save(self, args, state, control, **kwargs):
        silence, is_main, _, local_rank = _get_env_silence_and_rank()
        if local_rank != 0:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            _save_recurrent_memory_state(self.trainer.model, checkpoint_dir)
            epoch_int = int(round(float(state.epoch or 0.0)))
            if epoch_int > 0:
                alias_path = Path(args.output_dir) / f"epoch_{epoch_int}"
                try:
                    if alias_path.is_symlink() or alias_path.exists():
                        alias_path.unlink()
                    target_rel = os.path.relpath(str(checkpoint_dir), str(alias_path.parent))
                    os.symlink(target_rel, alias_path)
                except Exception as exc:
                    if not silence and is_main:
                        print(f"[save] failed to create epoch alias for {checkpoint_dir}: {exc}")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch_int = int(round(float(state.epoch or 0.0)))
        if epoch_int <= 0 or epoch_int in self._completed_epochs:
            return control
        self._completed_epochs.add(epoch_int)
        train_loss = _latest_logged_train_loss(state)
        val_loss = None
        if isinstance(metrics, dict) and "eval_loss" in metrics:
            try:
                val_loss = float(metrics["eval_loss"])
            except Exception:
                val_loss = None
        self.runner._run_generation_eval(
            epoch_tag=epoch_int,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        return control


def build_model_and_processor(args: argparse.Namespace):
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", "")).lower()
    if "qwen2_5_vl" in model_type or "qwen2.5" in model_type or "qwen2_5" in model_type:
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif "qwen3_vl" in model_type:
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("当前环境缺少 Qwen3VLForConditionalGeneration")
        model_cls = Qwen3VLForConditionalGeneration
    else:
        model_cls = AutoModelForVision2Seq

    quant_cfg = getattr(config, "quantization_config", None)
    use_bnb = bool(args.load_in_4bit and quant_cfg is None)
    bnb_config = None
    if use_bnb:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    tokenizer = processor.tokenizer
    memory_slot_tokens = [f"<|memory_{idx}|>" for idx in range(args.memory_slots)]
    special_tokens = [
        "<|trigger_start|>",
        "<|trigger_end|>",
        "<|task_start|>",
        "<|task_end|>",
        "<|step_start|>",
        "<|step_end|>",
        "<|future_steps_start|>",
        "<|future_steps_end|>",
        "<|next_action_start|>",
        "<|next_action_end|>",
        *memory_slot_tokens,
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    from_pretrained_kwargs = {"torch_dtype": torch.bfloat16}
    if bnb_config is not None:
        from_pretrained_kwargs["quantization_config"] = bnb_config
    local_rank = int(os.environ.get("LOCAL_RANK", "-1") or "-1")
    if local_rank >= 0 and torch.cuda.is_available():
        from_pretrained_kwargs["device_map"] = {"": local_rank}

    model = model_cls.from_pretrained(args.model_name, **from_pretrained_kwargs)
    if use_bnb:
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except TypeError:
            model = prepare_model_for_kbit_training(model)
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
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
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)

    memory_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in memory_slot_tokens]
    model_hidden_size = int(model.get_input_embeddings().weight.shape[1])
    action_vocab = LazySlidingWindowDataset(
        jsonl_path=args.train_json,
        window_size=args.window_size,
        window_stride=args.window_stride,
        use_trigger_hints=False,
        trigger_json_path=None,
        tokenizer=tokenizer,
        if_score=False,
        frame_root=args.frame_root,
        annotation_path=args.annotation,
        processor=processor,
        record_path=None,
        preprocessed_data_dir=None,
        preprocessed_data_file=None,
        preprocessed_train_file=None,
        preprocessed_val_file=None,
        priority_score_path=None,
        enable_evidence_frames=False,
        enable_reasoning=False,
        enable_confidence=False,
        enable_scores=False,
        use_reasoning_tokens=False,
        random_seed=args.seed,
        max_image_long_edge=args.max_image_long_edge,
        predict_steps=args.predict_steps,
        predict_next_action=True,
        history_memory_mode="none",
        history_recent_k=4,
    )._vocab_id_to_name
    num_actions = 1 + len({str(v).strip() for v in action_vocab.values() if str(v).strip()})
    model.recurrent_memory = RecurrentActionMemory(
        num_actions=num_actions,
        hidden_size=args.memory_hidden_size,
        memory_slots=args.memory_slots,
        memory_token_dim=model_hidden_size,
        commitment_dim=getattr(args, "commitment_dim", 0),
        commitment_mode=getattr(args, "commitment_mode", "learned"),
        commitment_dropout=getattr(args, "commitment_dropout", 0.5),
    )
    if args.load_from_checkpoint:
        _load_recurrent_checkpoint_state(model, args.load_from_checkpoint, require_recurrent=False)
    return model, processor, tokenizer, memory_slot_tokens, memory_token_ids


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    resolved_resume = ""
    if args.resume_from_checkpoint:
        resolved_resume = resolve_resume_checkpoint(args.resume_from_checkpoint) or ""
        if not resolved_resume:
            raise FileNotFoundError(f"无法解析 resume 路径: {args.resume_from_checkpoint}")
        args.resume_from_checkpoint = resolved_resume

    if args.resume_from_checkpoint and args.load_from_checkpoint:
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print("[resume] 已设置 resume_from_checkpoint，忽略 load_from_checkpoint")
        args.load_from_checkpoint = ""

    run_name = build_recurrent_run_name(args)
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model, processor, tokenizer, memory_slot_tokens, memory_token_ids = build_model_and_processor(args)
    train_dataset = RecurrentChunkDataset(
        jsonl_path=args.train_json,
        model_name=args.model_name,
        processor=processor,
        tokenizer=tokenizer,
        frame_root=args.frame_root,
        annotation_path=args.annotation,
        chunk_size=args.chunk_size,
        history_window=args.history_window,
        memory_slot_tokens=memory_slot_tokens,
        window_size=args.window_size,
        window_stride=args.window_stride,
        predict_steps=args.predict_steps,
        predict_next_action=True,
        max_image_long_edge=args.max_image_long_edge,
        enable_action_memory=getattr(args, "enable_action_memory", False),
        action_recent_k=getattr(args, "action_recent_k", 4),
    )
    eval_dataset = RecurrentChunkDataset(
        jsonl_path=args.val_json,
        model_name=args.model_name,
        processor=processor,
        tokenizer=tokenizer,
        frame_root=args.frame_root,
        annotation_path=args.annotation,
        chunk_size=args.chunk_size,
        history_window=args.history_window,
        memory_slot_tokens=memory_slot_tokens,
        window_size=args.window_size,
        window_stride=args.window_stride,
        predict_steps=args.predict_steps,
        predict_next_action=True,
        max_image_long_edge=args.max_image_long_edge,
        enable_action_memory=getattr(args, "enable_action_memory", False),
        action_recent_k=getattr(args, "action_recent_k", 4),
    )

    if args.smoke:
        train_dataset.chunks = train_dataset.chunks[:1]
        eval_dataset.chunks = eval_dataset.chunks[:1]
        args.num_train_epochs = 1
        args.eval_every_n_epochs = 1

    train_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=resolve_ddp_find_unused_parameters(
            args.disable_find_unused_parameters
        ),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
        seed=args.seed,
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass

    trainer = RecurrentChunkTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=chunk_collate_fn,
        base_collator=make_collate_fn(tokenizer.pad_token_id or tokenizer.eos_token_id),
        memory_token_ids=memory_token_ids,
        callbacks=[EveryNEpochCallback(args.eval_every_n_epochs)],
    )
    trainer.add_callback(
        RecurrentMetricsCallback(
            trainer=trainer,
            run_dir=run_dir,
            processor=processor,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            memory_slot_tokens=memory_slot_tokens,
            memory_token_ids=memory_token_ids,
            args=args,
        )
    )

    try:
        if hasattr(trainer.model, "_set_static_graph"):
            trainer.model._set_static_graph()
    except Exception:
        pass

    if args.train_jump and not args.resume_from_checkpoint:
        raise ValueError("train_jump 模式必须提供 resume_from_checkpoint（可传 run 目录或 checkpoint 目录）")

    if args.train_jump:
        expected_world = int(os.environ.get("WORLD_SIZE", "1") or "1")
        checkpoint_map = collect_checkpoint_map(run_dir)
        resume_epoch = checkpoint_epoch(args.resume_from_checkpoint) or 0
        pending_epochs = discover_epochs_needing_metrics(
            run_dir=run_dir,
            checkpoint_map=checkpoint_map,
            resume_epoch=resume_epoch,
            expected_world=expected_world,
        )
        latest_resume = args.resume_from_checkpoint
        for epoch in sorted(checkpoint_map):
            if epoch < resume_epoch:
                continue
            checkpoint_path = checkpoint_map[epoch]
            latest_resume = checkpoint_path
            if epoch not in pending_epochs:
                continue
            train_loss, val_loss = _checkpoint_logged_losses(checkpoint_path, epoch)
            _load_recurrent_checkpoint_state(trainer.model, checkpoint_path, require_recurrent=False)
            trainer.callback_handler.model = trainer.model
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"[train_jump] 补做 epoch_{epoch} 生成评测: {checkpoint_path}")
            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, RecurrentMetricsCallback):
                    callback.runner._run_generation_eval(
                        epoch_tag=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )
                    break
        latest_epoch = checkpoint_epoch(latest_resume) or 0
        if latest_epoch >= int(args.num_train_epochs):
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print("[train_jump] 所有已有 epoch 已补齐 metrics，训练阶段跳过。")
        else:
            _load_recurrent_checkpoint_state(trainer.model, latest_resume, require_recurrent=False)
            silence, is_main, _, _ = _get_env_silence_and_rank()
            if not silence and is_main:
                print(f"[train_jump] 从 {latest_resume} 继续训练")
            trainer.train(resume_from_checkpoint=latest_resume)
    elif args.resume_from_checkpoint:
        _load_recurrent_checkpoint_state(trainer.model, args.resume_from_checkpoint, require_recurrent=False)
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"[resume] 从 {args.resume_from_checkpoint} 继续训练")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(run_dir)
    if int(os.environ.get("LOCAL_RANK", "0") or "0") == 0:
        _save_recurrent_memory_state(trainer.model, run_dir)


if __name__ == "__main__":
    main()
