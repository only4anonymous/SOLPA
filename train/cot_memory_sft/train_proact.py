"""
ProAct 2.0 E2E Training + Inference

Architecture:
  Prompt:  [SYS] [USER: ep_stub IMAGES str_stub observation]
  Target:  trigger task step  PROC [GRAPH]  future action

  - ep/str tokens in user prompt prefix  -> perception (trigger/task/step)
  - proc/graph tokens in assistant output -> decision (future/action)
  - proc/graph positions masked from CE loss
  - Graph dropout (50%) during training

Inference (two-phase):
  Phase 1: generate trigger+task+step  (with ep+str injection)
  Phase 2: build full context up to proc/graph, generate future+action
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from safetensors import safe_open
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

import random as _random

from proact_memory import (
    ALL_MEMORY_TOKENS,
    EP_SLOT_TOKENS,
    GRAPH_SLOT_TOKENS,
    PROC_SLOT_TOKENS,
    STR_SLOT_TOKENS,
    ProActMemory,
    build_ep_stub,
    build_graph_stub,
    build_proc_stub,
    build_str_stub,
    build_task_graph_registry,
    inject_memory_token_embeddings,
    pad_action_sequences,
    prepare_graph_tensors,
    truncate_to_recent,
)
# Use the node-aware compressed renderer from train/graph_add/tool/subgraph.py
# (autocrop + non-node-band compression). This is the same one used by
# the QA dataset and the CPU preflight, so all paths share one impl.
from train.graph_add.tool.subgraph import build_subgraph_for_step
from train.graph_add.tool.graph_video_compat import is_graph_usable_for_state
from train.graph_add.tool.apa_next_action import (
    DEFAULT_WAIT_ACTION,
    resolve_next_action_label,
    resolve_future_steps_label,
    _get_mgr as _apa_get_mgr,
    _legal_now as _apa_legal_now,
    _is_cross_thread as _apa_is_cross_thread,
)

_SG_STAT = {"built": 0, "total": 0}

# -- Lever B: in-graph (task-graph-masked) constrained decoding for next_action --
# Pure inference-time. Gated by env GRAPH_MASKED_DECODE (default OFF). When ON, the
# next_action span (<|next_action_start|>...<|next_action_end|>) is restricted to a
# trie of legal in-graph node-name strings of the current task graph (plus the Wait
# token), eliminating the "action_not_in_task_graph" forced-Wait failure mode.
# Does NOT change any label/teacher behaviour or training; only the decode mask at eval.
_INGUARD_TRIE_CACHE: Dict[str, Dict[Any, Any]] = {}
_INGUARD_TOK_CACHE: Dict[int, Dict[str, Any]] = {}


def _inguard_real_action_names(mgr) -> List[str]:
    """Real action node names of a task graph (exclude midlevel/parallel/Terminate)."""
    out: List[str] = []
    n2n = getattr(mgr, "name2node", {}) or {}
    for name, node in n2n.items():
        s = str(name or "").strip()
        if not s or s.lower() == "terminate":
            continue
        try:
            if bool(node.get("is_midlevel", False)):
                continue
            if str(node.get("midlevel_type") or "").lower() == "parallel":
                continue
        except Exception:
            pass
        out.append(s)
    return out


def _inguard_tok_consts(tokenizer) -> Dict[str, Any]:
    key = id(tokenizer)
    c = _INGUARD_TOK_CACHE.get(key)
    if c is None:
        na_start = tokenizer.encode("<|next_action_start|>", add_special_tokens=False)
        na_end = tokenizer.encode("<|next_action_end|>", add_special_tokens=False)
        na_end_first = na_end[0] if na_end else (tokenizer.eos_token_id or 0)
        c = {
            "na_start": list(na_start),
            "na_end": list(na_end),
            "na_end_first": int(na_end_first),
            "all_tokens": list(range(len(tokenizer))),
        }
        _INGUARD_TOK_CACHE[key] = c
    return c


def _build_inguard_prefix_fn(tokenizer, mgr, task_pred: str, wait_token: str, prompt_len: int):
    """Build a HF prefix_allowed_tokens_fn that masks the next_action span to in-graph names."""
    consts = _inguard_tok_consts(tokenizer)
    na_start = consts["na_start"]
    na_end = consts["na_end"]
    na_end_first = consts["na_end_first"]
    all_tokens = consts["all_tokens"]

    cache = _INGUARD_TRIE_CACHE.get(task_pred)
    if cache is None:
        names = _inguard_real_action_names(mgr)
        if wait_token and wait_token not in names:
            names = names + [wait_token]
        root: Dict[Any, Any] = {}
        for nm in names:
            ids = tokenizer.encode(nm, add_special_tokens=False)
            if not ids:
                continue
            node = root
            for t in ids:
                node = node.setdefault(int(t), {})
            node["__end__"] = True
        cache = root
        _INGUARD_TRIE_CACHE[task_pred] = cache
    root = cache
    if not root:
        return None

    ns = len(na_start)
    ne = len(na_end)

    def _rfind(seq: List[int], pat: List[int]) -> int:
        n = len(pat)
        if n == 0:
            return -1
        for i in range(len(seq) - n, -1, -1):
            if seq[i:i + n] == pat:
                return i
        return -1

    def prefix_fn(batch_id, input_ids):
        gen = input_ids[prompt_len:].tolist()
        pos = _rfind(gen, na_start)
        if pos < 0:
            return all_tokens
        partial = gen[pos + ns:]
        if ne and _rfind(partial, na_end) >= 0:
            return all_tokens
        node = root
        ok = True
        for t in partial:
            nxt = node.get(int(t))
            if isinstance(nxt, dict):
                node = nxt
            else:
                ok = False
                break
        if not ok:
            return all_tokens
        allowed = [k for k in node.keys() if isinstance(k, int)]
        if node.get("__end__"):
            allowed.append(na_end_first)
        if not allowed:
            return all_tokens
        return allowed

    return prefix_fn


from recurrent_chunking import build_video_chunks
from recurrent_pipeline_utils import (
    build_generation_eval_record,
    hydrate_samples_for_chunking,
)
from train.cot_sft_recurrent_v1.jump_resume_utils import (
    checkpoint_epoch,
    collect_checkpoint_map,
    discover_epochs_needing_metrics,
    resolve_resume_checkpoint,
)
from train.cot_sft_recurrent_v1.train_recurrent_v1 import (
    _checkpoint_logged_losses,
    _collect_recurrent_lazy_labels,
    _extract_tagged_span,
    _generate_text,
    _ground_predictions,
    _latest_logged_train_loss,
    _load_task_to_canonical_steps,
    _parse_future_steps,
    _parse_trigger_value,
    _prepare_generation_inputs,
    _safe_dist_barrier,
)
from train.cot_sft_v2.train_cot_sft_two_stage import (
    LazySlidingWindowDataset,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    _get_env_silence_and_rank,
    build_history_memory_text,
    build_run_name_from_args,
    build_two_stage_conversation_bundle,
    make_collate_fn,
)
from train.l2.unified_metrics import compute_all_metrics
from eval_loader import (
    load_checkpoint_for_eval,
    load_stage_adapter_weights as _load_stage_adapter_weights,
    unwrap_model as _unwrap_model,
)

# QA integration imports (joint training)
_QA_IMPORTS_READY = False
try:
    from train.graph_add.qa_sampler import sample_qa_for_epoch, build_fixed_eval_qa
    from train.graph_add.qa_dataset import GraphQAMiniDataset, qa_collate_fn
    _QA_IMPORTS_READY = True
except ImportError:
    pass


_REQUIRED_TASK_PLANNER = (
    Path(__file__).resolve().parents[2] / "test" / "onestep_planning" / "task_planner.py"
)
if not _REQUIRED_TASK_PLANNER.exists():
    raise FileNotFoundError(
        f"[train_proact] task_planner.py not found: {_REQUIRED_TASK_PLANNER}"
    )


# ============================================================================
# E2E Prompt construction
# ============================================================================

_PROMPT_VARIANT_SYSTEM_SUFFIX: Dict[str, str] = {
    "baseline": "",
    "D1": (
        "Prefer robot actions that save the human a future step when that "
        "action is legally feasible."
    ),
    "C1": (
        "When the task allows parallel execution, prefer a robot action on a "
        "thread that the human is not currently working on."
    ),
    "C2": (
        "Prefer a robot action on a different branch than the human's current "
        "step to support parallel task completion."
    ),
    "D3": (
        "Minimize unnecessary future steps; prefer actions that reduce total "
        "remaining work."
    ),
    "C1D1": (
        "When the task allows parallel execution, prefer a robot action on a "
        "thread that the human is not currently working on. "
        "Prefer robot actions that save the human a future step when that "
        "action is legally feasible."
    ),
    "C1C2": (
        "When the task allows parallel execution, prefer a robot action on a "
        "thread that the human is not currently working on. "
        "Prefer a robot action on a different branch than the human's current "
        "step to support parallel task completion."
    ),
    "E2": (
        "Prefer robot actions that save the human a future step when that "
        "action is legally feasible."
    ),
}


def _normalize_prompt_variant(variant: str) -> str:
    key = str(variant or "baseline").strip().lower()
    return key if key in _PROMPT_VARIANT_SYSTEM_SUFFIX else "baseline"


def build_e2e_system_prompt(
    predict_steps: int = 5,
    history_mode: str = "none",
    thread_hint_mode: str = "none",
    prompt_variant: str = "baseline",
    future_steps_target: str = "horizon",
) -> str:
    lines = [
        "You are a proactive video assistant.",
        "Observe the scene, decide whether to trigger a proactive response,",
        "and if so, predict the current task and step, upcoming human steps,",
        "and the next robot action.",
        "Output the following tags in order: trigger, task, step, future steps, next action.",
        "Use none inside tags when information is unavailable.",
    ]
    if str(history_mode or "none").strip().lower() not in {"", "none"}:
        lines.append("Use the provided completed action history to avoid predicting already-done steps.")
    if str(thread_hint_mode or "none").strip().lower() == "oracle_cross_thread":
        lines.append(
            "The task has parallel execution threads. "
            "The robot should predict its next action on a different thread "
            "than the human's current step."
        )
    fst = str(future_steps_target or "horizon").strip().lower()
    if fst == "remaining":
        lines.append(
            "Predict all remaining human steps until task completion "
            "(include Terminate when the human is done)."
        )
    elif predict_steps > 0:
        lines.append(f"Predict up to {predict_steps} future human steps.")
    lines.append("Predict the next robot action.")
    suffix = _PROMPT_VARIANT_SYSTEM_SUFFIX.get(_normalize_prompt_variant(prompt_variant), "")
    if suffix:
        lines.append(suffix)
    lines.append("End your response with <|im_end|>.")
    return "\n".join(lines)


def _format_thread_display(thread_id: str) -> str:
    tid = str(thread_id or "serial_main").strip()
    if tid in {"", "serial_main"}:
        return "Main thread"
    return tid.replace("_", " ")


def _build_cross_thread_legal_pool_text(
    *,
    task_name: str,
    human_step: str,
    completed_steps: Optional[List[str]],
    future_steps: Optional[List[str]],
    annotation_path: str,
    tg_cache: Optional[Dict[str, Any]],
    wait_token: str = DEFAULT_WAIT_ACTION,
) -> str:
    """E1 cross-thread legal-pool surfacing.

    Lists the cross-thread *legal-now* action pool (the same set apa_parallel
    selects its label from) as an explicit text block, so the model can switch
    threads instead of defaulting to the same-thread autoregressive prior.
    Targets the legal-pool execution gap (gold cross-thread legal 100% but in
    rendered frontier only ~3%; CROSS_THREAD_MISS_RCA.md).

    The block is loss-masked context at train (like subgraph/thread-hint) and is
    reproduced verbatim in the eval prefix → train/eval input parity preserved.
    Returns "" on any failure so it degrades to the plain recipe.
    """
    step = str(human_step or "").strip()
    task = str(task_name or "").strip()
    if not step or not task:
        return ""
    try:
        mgr = _apa_get_mgr(task, annotation_path, tg_cache if tg_cache is not None else {})
        if mgr is None:
            return ""
        legal = _apa_legal_now(mgr, list(completed_steps or []), list(future_steps or []))
        legal.discard(step)
        cross = sorted(a for a in legal if _apa_is_cross_thread(mgr, a, step))
    except Exception:
        return ""
    if not cross:
        return (
            "[Executable now on OTHER threads]: (none)\n"
            f"No cross-thread action is legal now; the robot should answer \"{wait_token}\"."
        )
    listed = "; ".join(cross)
    return (
        f"[Executable now on OTHER threads]: {listed}\n"
        "These are the cross-thread actions the human is NOT doing but the robot "
        f"may start now. Pick exactly ONE of them as next_action (or \"{wait_token}\" "
        "if none truly helps); do NOT continue the human's current thread."
    )


def build_thread_hint_text(
    *,
    mode: str,
    task_name: str,
    human_step: str,
    taxonomy: Dict[str, Dict],
    completed_steps: Optional[List[str]] = None,
    future_steps: Optional[List[str]] = None,
    annotation_path: str = "",
    tg_cache: Optional[Dict[str, Any]] = None,
    wait_token: str = DEFAULT_WAIT_ACTION,
) -> str:
    """Oracle cross-thread hint (W2h) / cross-thread legal-pool surfacing (E1).

    Used in both train (format_e2e_target) and e_w2 eval (partial_assistant);
    callers MUST pass identical args in both paths to preserve input parity.
    """
    m = str(mode or "none").strip().lower()
    if m == "none":
        return ""

    step = str(human_step or "").strip()
    task = str(task_name or "").strip()

    if m == "cross_thread_legal_pool":
        return _build_cross_thread_legal_pool_text(
            task_name=task,
            human_step=step,
            completed_steps=completed_steps,
            future_steps=future_steps,
            annotation_path=annotation_path,
            tg_cache=tg_cache,
            wait_token=wait_token,
        )

    if m != "oracle_cross_thread":
        return ""
    if not step or not task or task not in taxonomy:
        return ""
    try:
        from train.graph_add.tool.thread_map import get_thread_map, thread_id_of_name
    except ImportError:
        return ""
    graph = {str(k): v for k, v in taxonomy[task].items()}
    thread_map = get_thread_map(graph, cache_key=task)
    tid = thread_id_of_name(step, thread_map)
    label = _format_thread_display(tid)
    return (
        f"Human current thread: {label}\n"
        "Robot must predict an action from a DIFFERENT parallel thread."
    )


def format_e2e_target(
    is_trigger: bool,
    task_name: str,
    step_name: str,
    future_steps: List[str],
    next_action: str,
    include_proc: bool = True,
    include_graph: bool = False,
    subgraph_text: str = "",
    thread_hint_text: str = "",
) -> str:
    trig = "true" if is_trigger else "false"
    task = (task_name or "").strip() if is_trigger else ""
    step = (step_name or "").strip() if is_trigger else ""
    task = task or "none"
    step = step or "none"

    future_steps = (
        [str(x).strip() for x in (future_steps or []) if str(x).strip()]
        if is_trigger else []
    )
    next_action = (next_action or "").strip() if is_trigger else ""
    future_text = "; ".join(future_steps) if future_steps else "Terminate"
    next_action = next_action or "Terminate"

    parts = [
        f"<|trigger_start|>{trig}<|trigger_end|>",
        f"<|task_start|>{task}<|task_end|>",
        f"<|step_start|>{step}<|step_end|>",
    ]

    # Subgraph text (ego-centric task graph) goes right after step_end,
    # before proc/graph tokens
    if subgraph_text:
        parts.append(subgraph_text)

    inject_parts = []
    if include_proc:
        inject_parts.append(build_proc_stub())
    if include_graph:
        inject_parts.append(build_graph_stub())
    if inject_parts:
        parts.append(" ".join(inject_parts))

    if thread_hint_text:
        parts.append(thread_hint_text)

    parts.append(f"<|future_steps_start|>{future_text}<|future_steps_end|>")
    parts.append(f"<|next_action_start|>{next_action}<|next_action_end|>")
    return "\n".join(parts)


_SUBGRAPH_SCOPE_NOTE = (
    "\n"
    "The graph shows legal/reachable steps for next robot action guidance only "
    "(frontier highlights proactive opportunities). "
    "Future human-step predictions are not restricted to nodes visible here; "
    "humans may take blocked, off-graph, or not-yet-legal steps."
)


def _e2e_user_content(
    images: List[Image.Image],
    frame_descs: List[str],
    video_id: str,
    subgraph_image: Optional[Image.Image] = None,
    history_text: str = "",
    subgraph_render_mode: str = "default",
) -> List[Dict[str, str]]:
    """Build user content list for E2E prompt."""
    content: List[Dict[str, str]] = []
    content.append({"type": "text", "text": build_ep_stub() + "\n\n"})
    for _ in images:
        content.append({"type": "image"})
    idx_text = ", ".join(frame_descs)
    obs_text = (
        "\n\n" + build_str_stub() + "\n\n"
        f"Video: {video_id}\n"
        f"Frames: {idx_text}\n"
    )
    if history_text:
        obs_text += f"{history_text}\n"
    obs_text += "Analyze the current scene and respond proactively."
    if subgraph_image is not None:
        # Embed the same legend the QA training uses so the model
        # interprets the colored / dashed boxes consistently across
        # E2E prediction and QA evaluation.
        _sg_mode = str(subgraph_render_mode or "default").strip().lower()
        if _sg_mode == "cross_thread_frontier":
            obs_text += (
                "\n\nTask progress graph (image follows). Legend:\n"
                "  Green box      ([done])     -- already completed step.\n"
                "  Yellow box     ([current])  -- human's current step.\n"
                "  Light-blue box ([frontier]) -- cross-thread legal action\n"
                "    (legal now AND on a different thread than current).\n"
                "  Gray box       ([future])   -- blocked or same-thread legal;\n"
                "    not highlighted as robot frontier.\n"
                "  Dashed grey box (no fill, small font) -- structural phase\n"
                "    container, not an action step (do NOT predict it).\n"
                "  Box labelled 'N done prereqs' -- N collapsed already-done\n"
                "    prerequisites converging on the same next step.\n"
                "  Edge label AND -- child needs every parent done.\n"
                "  Edge label OR  -- child needs any one parent done.\n"
                "  Subtle background tint -- execution thread grouping."
                + _SUBGRAPH_SCOPE_NOTE
            )
        else:
            obs_text += (
                "\n\nTask progress graph (image follows). Legend:\n"
                "  Green box      ([done])     -- already completed step.\n"
                "  Yellow box     ([current])  -- step currently in progress.\n"
                "  Light-blue box ([frontier]) -- currently legal next action\n"
                "    (all prerequisites already satisfied; can be started now).\n"
                "  Gray box       ([future])   -- still blocked: at least one\n"
                "    prerequisite is not yet done, so it cannot be started now.\n"
                "  Dashed grey box (no fill, small font) -- structural phase\n"
                "    container, not an action step (do NOT predict it).\n"
                "  Box labelled 'N done prereqs' -- N collapsed already-done\n"
                "    prerequisites converging on the same next step.\n"
                "  Edge label AND -- child needs every parent done.\n"
                "  Edge label OR  -- child needs any one parent done."
                + _SUBGRAPH_SCOPE_NOTE
            )
        content.append({"type": "text", "text": obs_text})
        content.append({"type": "image"})
    else:
        content.append({"type": "text", "text": obs_text})
    return content


def _e2e_prompt_messages(
    system_prompt: str,
    images: List[Image.Image],
    frame_descs: List[str],
    video_id: str,
    subgraph_image: Optional[Image.Image] = None,
    history_text: str = "",
    subgraph_render_mode: str = "default",
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": _e2e_user_content(
            images, frame_descs, video_id, subgraph_image=subgraph_image,
            history_text=history_text,
            subgraph_render_mode=subgraph_render_mode,
        )},
    ]


# ============================================================================
# E2E Sample builder
# ============================================================================

def _find_subsequence(seq: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    last = len(seq) - n
    for i in range(max(0, start), last + 1):
        if list(seq[i:i + n]) == list(needle):
            return i
    return -1


def _make_perception_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
    prompt_len: int,
) -> torch.Tensor:
    """Keep CE labels only for trigger/task/step output fields.

    Stage-2 graph tuning can improve decision tokens while drifting the earlier
    trigger/task/step format. This mask lets us add a small auxiliary CE on only
    those perception fields, without regularizing future/action decision tokens.
    """
    out = labels.new_full(labels.shape, -100)
    ids = input_ids.tolist()

    # Qwen's BPE may merge the leading whitespace/newline with "<|" and may
    # merge the closing ">" with a following newline. Match stable marker cores
    # instead of requiring the whole marker tokenization to be byte-identical.
    trigger_pat = tokenizer.encode("<|trigger_start|>", add_special_tokens=False)
    step_end_pat = tokenizer.encode("<|step_end|>", add_special_tokens=False)
    trigger_core = trigger_pat[2:] if len(trigger_pat) > 2 else trigger_pat
    step_end_core = step_end_pat[2:-1] if len(step_end_pat) > 3 else step_end_pat

    core_start = _find_subsequence(ids, trigger_core, start=prompt_len)
    if core_start < 0:
        return out
    start = core_start
    if core_start - 1 >= prompt_len and "<|" in tokenizer.decode([ids[core_start - 1]]):
        start = core_start - 1
    elif core_start - 2 >= prompt_len and tokenizer.decode(ids[core_start - 2:core_start]) == "<|":
        start = core_start - 2

    end_core_start = _find_subsequence(ids, step_end_core, start=core_start)
    if end_core_start < 0:
        return out
    end = end_core_start + len(step_end_core)
    for j in range(end, min(len(ids), end + 3)):
        end = j + 1
        if ">" in tokenizer.decode(ids[end_core_start:end]):
            break

    out[start:end] = labels[start:end]
    out[labels == -100] = -100
    return out


def _make_decision_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
    prompt_len: int,
    include_future_steps: bool = True,
) -> torch.Tensor:
    """CE on decision fields; next_action-only when include_future_steps=False."""
    out = labels.new_full(labels.shape, -100)
    ids = input_ids.tolist()

    if include_future_steps:
        field_start_pat = tokenizer.encode("<|future_steps_start|>", add_special_tokens=False)
        field_end_pat = tokenizer.encode("<|next_action_end|>", add_special_tokens=False)
    else:
        field_start_pat = tokenizer.encode("<|next_action_start|>", add_special_tokens=False)
        field_end_pat = tokenizer.encode("<|next_action_end|>", add_special_tokens=False)

    start_core = field_start_pat[2:] if len(field_start_pat) > 2 else field_start_pat
    end_core = field_end_pat[2:-1] if len(field_end_pat) > 3 else field_end_pat

    core_start = _find_subsequence(ids, start_core, start=prompt_len)
    if core_start < 0:
        return out
    start = core_start
    if core_start - 1 >= prompt_len and "<|" in tokenizer.decode([ids[core_start - 1]]):
        start = core_start - 1
    elif core_start - 2 >= prompt_len and tokenizer.decode(ids[core_start - 2:core_start]) == "<|":
        start = core_start - 2

    end_core_start = _find_subsequence(ids, end_core, start=core_start)
    if end_core_start < 0:
        return out
    end = end_core_start + len(end_core)
    for j in range(end, min(len(ids), end + 3)):
        end = j + 1
        if ">" in tokenizer.decode(ids[end_core_start:end]):
            break

    out[start:end] = labels[start:end]
    out[labels == -100] = -100
    return out



def build_e2e_sample(
    *,
    processor: Any,
    tokenizer: Any,
    images: List[Image.Image],
    system_prompt: str,
    frame_descs: List[str],
    video_id: str,
    target_text: str,
    proc_token_ids: List[int],
    graph_token_ids: List[int],
    subgraph_image: Optional[Image.Image] = None,
    subgraph_text: str = "",
    thread_hint_text: str = "",
    lora_stage: str = "",
    future_actions: bool = False,
    history_text: str = "",
    subgraph_render_mode: str = "default",
) -> Dict[str, Any]:
    """Build a single E2E training sample with proper loss masking.

    When subgraph_text is provided, its tokens in the target are masked from CE
    loss: the model reads the subgraph as context but is not trained to generate it.
    """
    messages_prompt = _e2e_prompt_messages(
        system_prompt, images, frame_descs, video_id,
        subgraph_image=subgraph_image,
        history_text=history_text,
        subgraph_render_mode=subgraph_render_mode,
    )
    messages_full = messages_prompt + [
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]

    all_images = list(images)
    if subgraph_image is not None:
        all_images.append(subgraph_image)

    text_full = processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False,
    )
    text_prompt = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True,
    )

    enc_full = processor(
        text=[text_full], images=all_images, return_tensors="pt", padding=True,
    )
    enc_prompt = processor(
        text=[text_prompt], images=all_images, return_tensors="pt", padding=True,
    )

    input_ids = enc_full["input_ids"][0]
    prompt_len = int(enc_prompt["input_ids"].shape[1])

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    # Mask injected proc/graph token positions in the target region
    all_inject_ids = proc_token_ids + graph_token_ids
    for tid in all_inject_ids:
        labels[input_ids == tid] = -100

    # Mask subgraph / thread-hint context regions from CE loss.
    mask_context_text = subgraph_text
    if thread_hint_text:
        mask_context_text = (mask_context_text + "\n" + thread_hint_text).strip()
    if mask_context_text:
        target_no_ctx = target_text
        if subgraph_text:
            target_no_ctx = target_no_ctx.replace(subgraph_text, "", 1)
        if thread_hint_text:
            target_no_ctx = target_no_ctx.replace(thread_hint_text, "", 1)
        msg_no_ctx = messages_prompt + [
            {"role": "assistant", "content": [{"type": "text", "text": target_no_ctx}]},
        ]
        text_no_ctx = processor.apply_chat_template(
            msg_no_ctx, tokenize=False, add_generation_prompt=False,
        )
        enc_no_ctx = processor(
            text=[text_no_ctx], images=all_images, return_tensors="pt", padding=True,
        )
        len_with = int(enc_full["input_ids"].shape[1])
        len_without = int(enc_no_ctx["input_ids"].shape[1])
        ctx_token_len = len_with - len_without

        if ctx_token_len > 0:
            ids_with = input_ids.tolist()
            ids_without = enc_no_ctx["input_ids"][0].tolist()
            ctx_start = prompt_len
            for i in range(prompt_len, min(len(ids_with), len(ids_without))):
                if ids_with[i] != ids_without[i]:
                    ctx_start = i
                    break
            labels[ctx_start:ctx_start + ctx_token_len] = -100

    perception_labels = _make_perception_labels(
        input_ids=input_ids,
        labels=labels,
        tokenizer=tokenizer,
        prompt_len=prompt_len,
    )
    stage = str(lora_stage or "").strip().lower()
    # W1/W2 always supervise future_steps (honest ED requires model generation at eval).
    include_future = stage in ("w1", "w2") or bool(future_actions)
    decision_labels = _make_decision_labels(
        input_ids=input_ids,
        labels=labels,
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        include_future_steps=include_future,
    )
    if stage == "w1":
        labels = perception_labels.clone()
        dec = decision_labels.clone()
        labels = torch.where(dec != -100, dec, labels)
    elif stage == "w2":
        labels = decision_labels.clone()

    sample: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": enc_full["attention_mask"][0],
        "labels": labels,
        "perception_labels": perception_labels,
        "decision_labels": decision_labels,
    }
    pv = enc_full.get("pixel_values")
    if pv is not None:
        sample["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv
    grid = enc_full.get("image_grid_thw")
    if grid is not None:
        if grid.dim() >= 3:
            grid = grid.squeeze(0)
        sample["image_grid_thw"] = grid.view(-1, 3)

    return sample


# ============================================================================
# Annotation / taxonomy helpers
# ============================================================================

def _load_annotation_taxonomy(annotation_path: Optional[str]) -> Dict[str, Dict]:
    """Load raw task taxonomy used by subgraph rendering.

    LazySlidingWindowDataset only keeps task/step mappings and does not expose
    ``_taxonomy``. Subgraph image/text construction needs the raw
    ``all_annotations.json["taxonomy"]`` graph, so we load it explicitly here.
    """
    if not annotation_path:
        return {}
    path = Path(annotation_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        taxonomy = data.get("taxonomy", {}) if isinstance(data, dict) else {}
        return taxonomy if isinstance(taxonomy, dict) else {}
    except Exception as e:
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"[taxonomy] failed to load {path}: {e}")
        return {}


def _dataset_taxonomy(ds: Any) -> Dict[str, Dict]:
    """Best-effort taxonomy getter for ProAct datasets and wrappers."""
    taxonomy = getattr(ds, "taxonomy", None)
    if taxonomy:
        return taxonomy
    taxonomy = getattr(ds, "annotation_taxonomy", None)
    if taxonomy:
        return taxonomy
    base = getattr(ds, "base_dataset", None)
    taxonomy = getattr(base, "_taxonomy", None) if base is not None else None
    return taxonomy if taxonomy else {}


# ============================================================================
# Dataset
# ============================================================================

class ProActE2EDataset(Dataset):
    """
    ProAct 2.0 E2E chunk dataset.

    Each sample is a chunk of consecutive time steps from one video.
    Each step produces a single E2E training sequence with memory stubs.
    """

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
        short_window: int,
        predict_steps: int,
        max_image_long_edge: int,
        use_graph: bool,
        window_size: int,
        window_stride: int,
        subgraph_mode: str = "none",
        subgraph_dropout: float = 0.5,
        subgraph_render_mode: str = "default",
        subgraph_text_format: str = "legacy",
        subgraph_text_omit_legal_pool: bool = False,
        lora_stage: str = "",
        future_actions: bool = False,
        history_memory_mode: str = "none",
        history_recent_k: int = 12,
        thread_hint_mode: str = "none",
        prompt_variant: str = "baseline",
        next_action_label_mode: str = "teacher",
        future_steps_target: str = "horizon",
        apa_wait_token: str = DEFAULT_WAIT_ACTION,
        annotation_path_for_labels: str = "",
        distill_label_map_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.short_window = int(short_window)
        self.predict_steps = int(predict_steps)
        self.max_image_long_edge = int(max_image_long_edge)
        self.use_graph = use_graph
        self.subgraph_mode = subgraph_mode
        self.subgraph_dropout = subgraph_dropout
        self.subgraph_render_mode = str(subgraph_render_mode or "default").strip().lower()
        self.subgraph_text_format = str(subgraph_text_format or "legacy").strip().lower()
        self.subgraph_text_omit_legal_pool = bool(subgraph_text_omit_legal_pool)
        self.lora_stage = str(lora_stage or "").strip().lower()
        self.future_actions = bool(future_actions)
        self.history_memory_mode = str(history_memory_mode or "none").strip().lower()
        self.history_recent_k = max(1, int(history_recent_k or 1))
        self.thread_hint_mode = str(thread_hint_mode or "none").strip().lower()
        self.prompt_variant = _normalize_prompt_variant(prompt_variant)
        self.next_action_label_mode = str(next_action_label_mode or "teacher").strip().lower()
        self.future_steps_target = str(future_steps_target or "horizon").strip().lower()
        self.apa_wait_token = str(apa_wait_token or DEFAULT_WAIT_ACTION).strip()
        self._annotation_path = str(
            annotation_path_for_labels or annotation_path or ""
        ).strip()
        self._tg_cache: Dict[str, Any] = {}

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
            predict_next_action=True,
            history_memory_mode=self.history_memory_mode,
            history_recent_k=self.history_recent_k,
            distill_label_map_path=distill_label_map_path,
        )

        # Raw task graph taxonomy for subgraph rendering. LazySlidingWindowDataset
        # does not expose this, so load it directly from annotation_path.
        self.annotation_taxonomy: Dict[str, Dict] = _load_annotation_taxonomy(annotation_path)

        # Build action vocabulary
        vocab = self.base_dataset._vocab_id_to_name
        self.action_name_to_id: Dict[str, int] = {"<pad>": 0}
        for _, name in sorted(vocab.items()):
            clean = str(name).strip()
            if clean and clean not in self.action_name_to_id:
                self.action_name_to_id[clean] = len(self.action_name_to_id)

        # Build graph registry from taxonomy
        self.graph_registry: Dict[str, Dict] = {}
        taxonomy_for_graph = (
            getattr(self.base_dataset, "_taxonomy", None)
            or self.annotation_taxonomy
        )
        if use_graph:
            if taxonomy_for_graph:
                self.graph_registry = build_task_graph_registry(
                    taxonomy_for_graph, self.action_name_to_id,
                )
            else:
                raise RuntimeError(
                    "use_graph=True but no task taxonomy was loaded from annotation_path"
                )

        # Load raw taxonomy for subgraph extraction
        self.taxonomy: Dict[str, Dict] = {}
        if subgraph_mode in ("text", "image"):
            self.taxonomy = taxonomy_for_graph or {}
            if not self.taxonomy:
                raise RuntimeError(
                    f"subgraph_mode={subgraph_mode!r} requires taxonomy in annotation_path"
                )

        # Collect lazy labels and build chunks
        lazy_labels = _collect_recurrent_lazy_labels(self.base_dataset)
        samples_for_chunking = hydrate_samples_for_chunking(
            self.base_dataset.samples_meta, lazy_labels,
        )
        for idx, enriched in enumerate(samples_for_chunking):
            enriched["index"] = idx

        self.chunks = build_video_chunks(
            samples_for_chunking,
            chunk_size=self.chunk_size,
            history_window=self.short_window,
        )

        # System prompt
        self.system_prompt = build_e2e_system_prompt(
            predict_steps,
            history_mode=self.history_memory_mode,
            thread_hint_mode=self.thread_hint_mode,
            prompt_variant=self.prompt_variant,
            future_steps_target=self.future_steps_target,
        )

        # Token IDs for injection
        self.ep_token_ids = [
            tokenizer.convert_tokens_to_ids(t) for t in EP_SLOT_TOKENS
        ]
        self.str_token_ids = [
            tokenizer.convert_tokens_to_ids(t) for t in STR_SLOT_TOKENS
        ]
        self.proc_token_ids = [
            tokenizer.convert_tokens_to_ids(t) for t in PROC_SLOT_TOKENS
        ]
        self.graph_token_ids = [
            tokenizer.convert_tokens_to_ids(t) for t in GRAPH_SLOT_TOKENS
        ] if use_graph else []

    def __len__(self) -> int:
        return len(self.chunks)

    def _load_window(self, meta: Dict[str, Any]) -> Tuple[List[Image.Image], List[str]]:
        images: List[Image.Image] = []
        frame_descs: List[str] = []
        window_files = meta["window_files"]
        if not window_files:
            raise RuntimeError(f"No frames: video_id={meta['video_id']}")
        idx0 = window_files[0][1]
        for j, (pth, idx_int) in enumerate(window_files):
            t = (idx_int - idx0) / 25.0
            frame_descs.append(f"F{j} [idx={idx_int} t={t:.2f}s]")
            images.append(self.base_dataset._load_image_cached(pth))
        return images, frame_descs

    def _action_ids(self, actions: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for a in actions:
            clean = str(a).strip()
            if clean:
                ids.append(self.action_name_to_id.get(clean, 0))
        return ids

    def _build_step_bundle(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        images, frame_descs = self._load_window(meta)
        all_completed = list(meta.get("completed_steps", []) or [])
        short_actions = truncate_to_recent(all_completed, self.short_window)

        is_trigger = bool(meta["is_trigger"])
        task_name = str(meta.get("task_name", "") or "")
        step_name = str(meta.get("step_name", "") or "")
        horizon_future = list(meta.get("future_steps", []) or [])
        future_steps = resolve_future_steps_label(
            self.future_steps_target,
            task_name=task_name,
            completed_steps=all_completed,
            horizon_future_steps=horizon_future,
            annotation_path=self._annotation_path,
            tg_cache=self._tg_cache,
            is_trigger=is_trigger,
        )
        teacher_next_action = str(meta.get("next_action", "") or "")
        next_action = resolve_next_action_label(
            self.next_action_label_mode,
            task_name=task_name,
            completed_steps=all_completed,
            current_step=step_name,
            future_steps=future_steps,
            teacher_action=teacher_next_action,
            annotation_path=self._annotation_path,
            tg_cache=self._tg_cache,
            wait_token=self.apa_wait_token,
            is_trigger=is_trigger,
        )

        # Decide whether this sample includes graph tokens
        has_graph = (
            self.use_graph
            and task_name in self.graph_registry
        )

        # Compute subgraph (text or image) with dropout
        subgraph_text = ""
        subgraph_image = None
        has_subgraph = False
        if (
            self.subgraph_mode in ("text", "image")
            and task_name
            and task_name in self.taxonomy
            and is_trigger
        ):
            use_subgraph = _random.random() >= self.subgraph_dropout
            if use_subgraph:
                _vid_ok, _vid_reason = is_graph_usable_for_state(
                    self.taxonomy, task_name, all_completed, step_name
                )
                if not _vid_ok:
                    use_subgraph = False
            if use_subgraph:
                _sg_kw = dict(
                    render_mode=self.subgraph_render_mode,
                    future_steps=future_steps,
                    annotation_path=self._annotation_path,
                    tg_cache=self._tg_cache,
                )
                if self.subgraph_mode == "text":
                    result = build_subgraph_for_step(
                        self.taxonomy, task_name, all_completed,
                        step_name, mode="text",
                        text_format=self.subgraph_text_format,
                        text_omit_legal_pool=self.subgraph_text_omit_legal_pool,
                        **_sg_kw,
                    )
                    if result:
                        subgraph_text = result
                        has_subgraph = True
                elif self.subgraph_mode == "image":
                    try:
                        result = build_subgraph_for_step(
                            self.taxonomy, task_name, all_completed,
                            step_name, mode="image",
                            **_sg_kw,
                        )
                        if result is not None:
                            subgraph_image = result
                            has_subgraph = True
                    except Exception:
                        pass

        # Proc tokens (short-term memory) always coexist with subgraph
        use_proc = True
        use_graph_tokens = has_graph

        thread_hint_text = ""
        if self.thread_hint_mode != "none" and is_trigger:
            thread_hint_text = build_thread_hint_text(
                mode=self.thread_hint_mode,
                task_name=task_name,
                human_step=step_name,
                taxonomy=self.taxonomy,
                completed_steps=all_completed,
                future_steps=future_steps,
                annotation_path=self._annotation_path,
                tg_cache=self._tg_cache,
                wait_token=self.apa_wait_token,
            )

        target_text = format_e2e_target(
            is_trigger=is_trigger,
            task_name=task_name,
            step_name=step_name,
            future_steps=future_steps,
            next_action=next_action,
            include_proc=use_proc,
            include_graph=use_graph_tokens,
            subgraph_text=subgraph_text,
            thread_hint_text=thread_hint_text,
        )

        proc_ids_for_mask = self.proc_token_ids if use_proc else []
        graph_ids_for_mask = self.graph_token_ids if use_graph_tokens else []

        train_history_text = ""
        if self.history_memory_mode != "none":
            train_history_text = build_history_memory_text(
                all_completed,
                mode=self.history_memory_mode,
                recent_k=self.history_recent_k,
            )

        sample = build_e2e_sample(
            processor=self.processor,
            tokenizer=self.tokenizer,
            images=images,
            system_prompt=self.system_prompt,
            frame_descs=frame_descs,
            video_id=meta["video_id"],
            target_text=target_text,
            proc_token_ids=proc_ids_for_mask,
            graph_token_ids=graph_ids_for_mask,
            subgraph_image=subgraph_image,
            subgraph_text=subgraph_text,
            thread_hint_text=thread_hint_text,
            lora_stage=self.lora_stage,
            future_actions=self.future_actions,
            history_text=train_history_text,
            subgraph_render_mode=self.subgraph_render_mode,
        )

        return {
            "sample": sample,
            "short_action_ids": self._action_ids(short_actions),
            "all_completed_ids": self._action_ids(all_completed),
            "future_ids": self._action_ids(future_steps),
            "update_actions_after_ids": self._action_ids(
                meta.get("update_actions_after", [])
            ),
            "completed_steps": all_completed,
            "has_graph": has_graph,
            "has_subgraph": has_subgraph,
            "task_name": task_name,
            "step_name": step_name,
            "meta": dict(meta),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        initial_all = chunk["initial_completed_steps"]
        initial_short = truncate_to_recent(initial_all, self.short_window)
        return {
            "video_id": chunk["video_id"],
            "initial_short_ids": self._action_ids(initial_short),
            "initial_all_ids": self._action_ids(initial_all),
            "steps": [self._build_step_bundle(m) for m in chunk["steps"]],
        }


def chunk_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"chunks": batch}


# ============================================================================
# Generation / Evaluation
# ============================================================================

def _eval_shard_path(run_dir, epoch_tag, rank):
    return Path(run_dir) / "eval_pred" / f"epoch_{epoch_tag}.rank{rank}.jsonl"


def _eval_done_path(run_dir, epoch_tag, rank):
    return Path(run_dir) / "eval_pred" / f"epoch_{epoch_tag}.rank{rank}.done"


def _wait_for_shards(run_dir, epoch_tag, world):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    t0 = time.time()
    while True:
        missing = []
        for r in range(world):
            if not _eval_shard_path(run_dir, epoch_tag, r).exists():
                missing.append(r)
            if not _eval_done_path(run_dir, epoch_tag, r).exists():
                missing.append(r)
        if not missing:
            return
        if time.time() - t0 > 1800:
            raise RuntimeError(f"Eval shard timeout: epoch={epoch_tag}")
        time.sleep(2.0)


class ProActGenerationRunner:
    """
    Two-phase inference for ProAct 2.0 E2E model.

    Phase 1: ep+str in prompt -> generate trigger + task + step
    Phase 2: build full context with proc [+graph], generate future + action
    """

    def __init__(
        self,
        *,
        trainer: "ProActTrainer",
        run_dir: str,
        processor: Any,
        tokenizer: Any,
        eval_dataset: ProActE2EDataset,
        args: argparse.Namespace,
    ) -> None:
        self.trainer = trainer
        self.run_dir = run_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.args = args
        self.system_prompt = eval_dataset.system_prompt
        self.task_to_steps = _load_task_to_canonical_steps(
            getattr(args, "annotation", ""),
        )

    def _phase1_generate(
        self,
        model: Any,
        memory: ProActMemory,
        images: List[Image.Image],
        frame_descs: List[str],
        video_id: str,
        epi_state: torch.Tensor,
        proc_state: torch.Tensor,
        device: torch.device,
    ) -> str:
        """Phase 1: generate trigger + task + step with ep+str tokens."""
        z, str_tokens = memory.compute_str_tokens(epi_state, proc_state)
        ep_tokens = memory.project_ep_tokens(epi_state)
        ep_tokens = ep_tokens.to(dtype=next(model.parameters()).dtype)
        str_tokens = str_tokens.to(dtype=next(model.parameters()).dtype)

        messages = _e2e_prompt_messages(
            self.system_prompt, images, frame_descs, video_id,
        )
        _, prompt_len, inputs = _prepare_generation_inputs(
            processor=self.processor,
            tokenizer=self.tokenizer,
            images=images,
            messages_prompt=messages,
            device=device,
        )

        phase1_inject_ids = (
            self.eval_dataset.ep_token_ids + self.eval_dataset.str_token_ids
        )
        phase1_inject_values = torch.cat([ep_tokens, str_tokens], dim=1)

        text = _generate_text(
            model=model,
            tokenizer=self.tokenizer,
            prompt_inputs=inputs,
            prompt_len=prompt_len,
            max_new_tokens=96,
            memory_token_ids=phase1_inject_ids,
            memory_values=phase1_inject_values,
        )
        return text

    def _phase2_generate(
        self,
        model: Any,
        memory: ProActMemory,
        images: List[Image.Image],
        frame_descs: List[str],
        video_id: str,
        epi_state: torch.Tensor,
        proc_state: torch.Tensor,
        state_formatted: str,
        task_pred: str,
        device: torch.device,
        completed_steps: Optional[List[str]] = None,
        step_pred: str = "",
        future_steps: Optional[List[str]] = None,
    ) -> str:
        """Phase 2: full re-encode, generate future+action.

        When subgraph is active, the subgraph replaces proc/graph tokens.
        Only ep+str tokens are injected for perception.
        """
        z, str_tokens = memory.compute_str_tokens(epi_state, proc_state)
        ep_tokens = memory.project_ep_tokens(epi_state)

        model_dtype = next(model.parameters()).dtype
        ep_tokens = ep_tokens.to(dtype=model_dtype)
        str_tokens = str_tokens.to(dtype=model_dtype)

        # Will be zeroed after subgraph construction if graph_mask_str is active
        _should_mask_str = bool(os.environ.get("PROACT_EVAL_GRAPH_MASK_STR"))

        # Compute subgraph for inference (always include, no dropout)
        subgraph_text_insert = ""
        subgraph_img = None
        has_subgraph = False
        sg_mode = self.eval_dataset.subgraph_mode
        sg_render_mode = getattr(self.eval_dataset, "subgraph_render_mode", "default")
        sg_text_format = getattr(self.eval_dataset, "subgraph_text_format", "legacy")
        taxonomy = self.eval_dataset.taxonomy
        _sg_kw = dict(
            render_mode=sg_render_mode,
            future_steps=list(future_steps or []),
            annotation_path=getattr(self.eval_dataset, "_annotation_path", ""),
            tg_cache=getattr(self.eval_dataset, "_tg_cache", None),
        )
        if sg_mode in ("text", "image") and task_pred and taxonomy:
            comp = completed_steps or []
            cur_step = step_pred or ""
            if sg_mode == "text":
                result = build_subgraph_for_step(
                    taxonomy, task_pred, comp, cur_step, mode="text",
                    text_format=sg_text_format,
                    **_sg_kw,
                )
                if result:
                    subgraph_text_insert = result
                    has_subgraph = True
            elif sg_mode == "image":
                try:
                    result = build_subgraph_for_step(
                        taxonomy, task_pred, comp, cur_step, mode="image",
                        **_sg_kw,
                    )
                    if result is not None:
                        subgraph_img = result
                        has_subgraph = True
                except Exception:
                    pass

        # Eval-time graph ablation probe (research only; training is unaffected
        # because this runner is only used during generation eval). Controlled
        # by env PROACT_EVAL_GRAPH_ABLATE: none(default) | blank | drop.
        #   blank: keep the injection path identical (image still present, proc
        #          tokens still suppressed) but erase all graph *content* -> tests
        #          whether the decision leans on graph information.
        #   drop : remove the graph entirely and fall back to proc/graph tokens.
        _abl = os.environ.get("PROACT_EVAL_GRAPH_ABLATE", "none").strip().lower()
        if _abl == "blank" and subgraph_img is not None:
            subgraph_img = Image.new("RGB", subgraph_img.size, (255, 255, 255))
        elif _abl == "shuffle" and subgraph_img is not None and taxonomy and task_pred:
            alt_tasks = [t for t in taxonomy if t != task_pred]
            if alt_tasks:
                wrong_task = alt_tasks[hash(str(video_id)) % len(alt_tasks)]
                try:
                    wrong_img = build_subgraph_for_step(
                        taxonomy, wrong_task, comp, cur_step, mode="image",
                    )
                    if wrong_img is not None:
                        subgraph_img = wrong_img
                except Exception:
                    pass
        elif _abl == "drop":
            subgraph_img = None
            subgraph_text_insert = ""
            has_subgraph = False

        if os.environ.get("PROACT_LOG_SUBGRAPH"):
            _SG_STAT["total"] += 1
            if has_subgraph:
                _SG_STAT["built"] += 1
            if _SG_STAT["total"] % 50 == 0:
                print(f"[sg-stat] has_subgraph built={_SG_STAT['built']}/{_SG_STAT['total']} "
                      f"({100.0*_SG_STAT['built']/max(1,_SG_STAT['total']):.1f}%)", flush=True)

        # Build decision-side inject stubs (proc/graph only when no subgraph)
        decision_inject_ids: List[int] = []
        decision_inject_vals: List[torch.Tensor] = []
        inject_stub = ""

        # Zero str tokens when graph is present and mask_str is active
        if _should_mask_str and has_subgraph:
            str_tokens = torch.zeros_like(str_tokens)

        # Match training: --no_proact_memory skips proc/graph injection (W2d).
        if not getattr(self.trainer, "_no_proact_memory", False) and not has_subgraph:
            proc_tokens = memory.project_proc_tokens(proc_state)
            proc_tokens = proc_tokens.to(dtype=model_dtype)
            inject_stub = build_proc_stub()
            decision_inject_ids = list(self.eval_dataset.proc_token_ids)
            decision_inject_vals = [proc_tokens]

            has_graph = (
                memory.use_graph
                and task_pred in self.eval_dataset.graph_registry
            )
            if has_graph:
                graph_data = prepare_graph_tensors(
                    self.eval_dataset.graph_registry[task_pred], device=device,
                )
                g_tokens = memory.compute_graph_tokens(
                    node_ids=graph_data["node_ids"],
                    node_mask=graph_data["node_mask"],
                    adjacency_mask=graph_data.get("adjacency_mask"),
                )
                if g_tokens is not None:
                    g_tokens = g_tokens.to(dtype=model_dtype)
                    inject_stub += " " + build_graph_stub()
                    decision_inject_ids.extend(self.eval_dataset.graph_token_ids)
                    decision_inject_vals.append(g_tokens)

        # Build partial assistant text
        partial_parts = [state_formatted]
        if subgraph_text_insert:
            partial_parts.append(subgraph_text_insert)
            # Match training format_e2e_target(include_proc=True): proc stub text
            # is present in the target even when subgraph replaces proc injection.
            partial_parts.append(build_proc_stub())
        if inject_stub:
            partial_parts.append(inject_stub)
        _hint_mode = getattr(self.eval_dataset, "thread_hint_mode", "none")
        if _hint_mode != "none" and task_pred:
            thread_hint = build_thread_hint_text(
                mode=_hint_mode,
                task_name=task_pred,
                human_step=step_pred or "",
                taxonomy=taxonomy,
                completed_steps=list(completed_steps or []),
                future_steps=list(future_steps or []),
                annotation_path=getattr(self.eval_dataset, "_annotation_path", ""),
                tg_cache=getattr(self.eval_dataset, "_tg_cache", None),
                wait_token=getattr(self.eval_dataset, "apa_wait_token", DEFAULT_WAIT_ACTION),
            )
            if thread_hint:
                partial_parts.append(thread_hint)
        partial_assistant = "\n".join(partial_parts)
        # Eval contract: format_e2e_target joins fields with "\n". Trailing newline before
        # <|future_steps_start|> generation matches teacher-forced training boundary.
        if not os.environ.get("ZERO_APA_DISABLE_NL", "") and partial_assistant:
            if not partial_assistant.endswith("\n"):
                partial_assistant += "\n"

        # Build history text for eval — must match training format exactly.
        eval_history_text = ""
        _hist_mode = getattr(self.eval_dataset, "history_memory_mode", "none")
        if _hist_mode != "none":
            _hist_k = getattr(self.eval_dataset, "history_recent_k", 12)
            eval_history_text = build_history_memory_text(
                completed_steps or [],
                mode=_hist_mode,
                recent_k=_hist_k,
            )

        # Build user content (with optional subgraph image)
        user_content = _e2e_user_content(
            images, frame_descs, video_id,
            subgraph_image=subgraph_img,
            history_text=eval_history_text,
            subgraph_render_mode=sg_render_mode,
        )
        all_images = list(images)
        if subgraph_img is not None:
            all_images.append(subgraph_img)

        base_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        base_text = self.processor.apply_chat_template(
            base_messages, tokenize=False, add_generation_prompt=True,
        )
        full_context = base_text + partial_assistant

        enc = self.processor(
            text=[full_context], images=all_images,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        prompt_len = inputs["input_ids"].shape[1]

        # Inject perception tokens (ep+str) + decision (proc/graph); skip all when no_proact_memory.
        if getattr(self.trainer, "_no_proact_memory", False):
            full_inject_ids = None
            full_inject_values = None
        else:
            full_inject_ids = (
                list(self.eval_dataset.ep_token_ids)
                + list(self.eval_dataset.str_token_ids)
                + decision_inject_ids
            )
            full_inject_vals = [ep_tokens, str_tokens] + decision_inject_vals
            full_inject_values = torch.cat(full_inject_vals, dim=1)

        _inguard_fn = None
        if os.environ.get("GRAPH_MASKED_DECODE") and task_pred:
            try:
                _ann = getattr(self.eval_dataset, "_annotation_path", "")
                _tgc = getattr(self.eval_dataset, "_tg_cache", None)
                _mgr = _apa_get_mgr(task_pred, _ann, _tgc if _tgc is not None else {})
                if _mgr is not None:
                    _wait = getattr(self.eval_dataset, "apa_wait_token", DEFAULT_WAIT_ACTION)
                    _inguard_fn = _build_inguard_prefix_fn(
                        self.tokenizer, _mgr, task_pred, _wait, prompt_len,
                    )
            except Exception:
                _inguard_fn = None
        text = _generate_text(
            model=model,
            tokenizer=self.tokenizer,
            prompt_inputs=inputs,
            prompt_len=prompt_len,
            max_new_tokens=192,
            memory_token_ids=full_inject_ids,
            memory_values=full_inject_values,
            prefix_allowed_tokens_fn=_inguard_fn,
        )
        if os.environ.get("ZERO_APA_DUMP", ""):
            try:
                import json as _json
                _dump_path = os.environ.get(
                    "ZERO_APA_DUMP_PATH",
                    "<REPO_ROOT>/refine-logs/zero_apa_debug/phase2_dump.jsonl",
                )
                _ids = inputs["input_ids"][0].tolist()
                _gen_ids = _ids[prompt_len:] if False else None
                _na_ids = self.tokenizer.encode("<|next_action_start|>", add_special_tokens=False)
                _fse_ids = self.tokenizer.encode("<|future_steps_end|>", add_special_tokens=False)
                _rec = {
                    "video_id": video_id,
                    "no_proact_memory": bool(getattr(self.trainer, "_no_proact_memory", False)),
                    "lora_stage": str(getattr(self.args, "lora_stage", "")),
                    "prompt_len": int(prompt_len),
                    "partial_assistant_tail": partial_assistant[-800:],
                    "full_context_tail": full_context[-400:],
                    "prompt_last_ids": _ids[-40:],
                    "next_action_start_ids": _na_ids,
                    "future_steps_end_ids": _fse_ids,
                    "raw_gen_text": text,
                    "raw_gen_repr": repr(text)[:600],
                }
                with open(_dump_path, "a", encoding="utf-8") as _fh:
                    _fh.write(_json.dumps(_rec, ensure_ascii=False) + "\n")
            except Exception as _e:
                print(f"[ZERO_APA_DUMP] failed: {_e}", flush=True)
        return text

    def run(
        self,
        epoch_tag: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
    ) -> None:
        model = self.trainer.model
        memory: ProActMemory = self.trainer.proact_memory
        device = next(model.parameters()).device
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        pred_dir = Path(self.run_dir) / "eval_pred"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_path = _eval_shard_path(self.run_dir, epoch_tag, rank)
        done_path = _eval_done_path(self.run_dir, epoch_tag, rank)
        for p in (out_path, done_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        local_indices = list(range(rank, len(self.eval_dataset), world))
        silence, is_main, _, _ = _get_env_silence_and_rank()
        _eval_max = int(os.environ.get("EVAL_MAX_CHUNKS", "0") or 0)
        if _eval_max > 0:
            capped = list(range(min(_eval_max, len(self.eval_dataset))))
            local_indices = [i for i in capped if i % world == rank]
            if not silence and is_main:
                print(
                    f"[proact-eval] EVAL_MAX_CHUNKS={_eval_max} "
                    f"total_chunks={len(capped)} world={world}",
                    flush=True,
                )

        model.eval()
        with out_path.open("w", encoding="utf-8") as handle:
            for chunk_pos, chunk_idx in enumerate(local_indices, 1):
                chunk = self.eval_dataset[chunk_idx]

                # Init GRU states
                epi_ids, epi_mask = pad_action_sequences(
                    [chunk["initial_all_ids"]], device=device,
                )
                epi_state = memory.encode_epi(epi_ids, epi_mask)
                proc_ids, proc_mask = pad_action_sequences(
                    [chunk["initial_short_ids"]], device=device,
                )
                proc_state = memory.encode_proc(proc_ids, proc_mask)

                for step in chunk["steps"]:
                    meta = step["meta"]
                    images, frame_descs = self.eval_dataset._load_window(meta)
                    t0 = time.time()

                    # === DLV2 eval-protocol patch ===
                    _eval_proto = str(getattr(self.args, "eval_protocol", "e2e") or "e2e").strip().lower()
                    _lora_stage = str(getattr(self.args, "lora_stage", "") or "").strip().lower()
                    if _eval_proto == "e_w2":
                        pred_is = bool(meta.get("is_trigger"))
                        pred_task = str(step.get("task_name") or meta.get("task_name") or "")
                        pred_step = str(step.get("step_name") or meta.get("step_name") or "")
                        if not pred_is:
                            pred_task = ""
                            pred_step = ""
                        phase1_text = "\n".join([
                            f"<|trigger_start|>{'true' if pred_is else 'false'}<|trigger_end|>",
                            f"<|task_start|>{pred_task or 'none'}<|task_end|>",
                            f"<|step_start|>{pred_step or 'none'}<|step_end|>",
                        ])
                    else:
                        # === DLV2 combined-adapter switch patch ===
                        if _lora_stage == "combined":
                            _peft = _unwrap_model(model)
                            try:
                                _peft.set_adapter(["default", "w1"])
                            except Exception:
                                _peft.set_adapter("w1")
                        phase1_text = self._phase1_generate(
                            model, memory, images, frame_descs,
                            meta["video_id"], epi_state, proc_state, device,
                        )
                        pred_is = _parse_trigger_value(phase1_text)
                        pred_task = (
                            _extract_tagged_span(
                                phase1_text, "<|task_start|>", "<|task_end|>"
                            ) if pred_is else ""
                        )
                        pred_step = (
                            _extract_tagged_span(
                                phase1_text, "<|step_start|>", "<|step_end|>"
                            ) if pred_is else ""
                        )

                    state_formatted = "\n".join([
                        f"<|trigger_start|>{'true' if pred_is else 'false'}<|trigger_end|>",
                        f"<|task_start|>{pred_task or 'none'}<|task_end|>",
                        f"<|step_start|>{pred_step or 'none'}<|step_end|>",
                    ])

                    pred_future: List[str] = []
                    pred_next_action = ""
                    if pred_is and _eval_proto != "e_w1":
                        # combined-only: switch adapters before phase-2. w1/w2 stay on
                        # init-time active adapters (inline + dedicated disk-reload paths).
                        if _lora_stage == "combined":
                            _peft = _unwrap_model(model)
                            try:
                                _peft.set_adapter(["default", "w1", "w2"])
                            except Exception:
                                _peft.set_adapter("w2")
                        phase2_text = self._phase2_generate(
                            model, memory, images, frame_descs,
                            meta["video_id"], epi_state, proc_state,
                            state_formatted, pred_task or "", device,
                            completed_steps=step.get("completed_steps"),
                            step_pred=pred_step or "",
                            future_steps=list(meta.get("future_steps") or []),
                        )
                        pred_future = _parse_future_steps(phase2_text)
                        pred_next_action = _extract_tagged_span(
                            phase2_text,
                            "<|next_action_start|>", "<|next_action_end|>",
                        )

                    gen_time = time.time() - t0

                    # Grounding
                    gt, gs, gf, ga = _ground_predictions(
                        pred_task=pred_task,
                        pred_step=pred_step,
                        pred_future_steps=pred_future,
                        pred_next_action=pred_next_action,
                        task_to_steps=self.task_to_steps,
                        threshold=0.8,
                    )

                    record = build_generation_eval_record(
                        meta=meta,
                        pred_is=bool(pred_is),
                        pred_task=pred_task,
                        pred_task_matched=gt,
                        pred_step=pred_step,
                        pred_step_matched=gs,
                        pred_future_steps=gf,
                        pred_next_action=ga,
                        normalized_text=phase1_text,
                        generation_time=float(gen_time),
                        future_steps_target=self.eval_dataset.future_steps_target,
                        annotation_path=self.eval_dataset._annotation_path,
                        tg_cache=self.eval_dataset._tg_cache,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Update GRU states
                    upd_ids, upd_mask = pad_action_sequences(
                        [step["update_actions_after_ids"]], device=device,
                    )
                    epi_state = memory.update_epi(epi_state, upd_ids, upd_mask)
                    proc_state = memory.update_proc(proc_state, upd_ids, upd_mask)

                if not silence and is_main and chunk_pos % 10 == 0:
                    print(
                        f"[proact-eval] epoch={epoch_tag} rank={rank} "
                        f"chunks={chunk_pos}/{len(local_indices)}"
                    )

            handle.flush()
            os.fsync(handle.fileno())

        done_path.write_text("ok", encoding="utf-8")
        _safe_dist_barrier(f"proact_eval_done_{epoch_tag}")
        _wait_for_shards(self.run_dir, epoch_tag, world)
        _safe_dist_barrier(f"proact_shards_ready_{epoch_tag}")

        _eval_proto = str(getattr(self.args, "eval_protocol", "e2e") or "e2e").strip().lower()
        compute_all_metrics(
            run_dir=self.run_dir,
            epoch_tag=epoch_tag,
            args=self.args,
            train_loss=train_loss,
            val_loss=val_loss,
            method_name="proact",
            barrier_prefix="proact",
            skip_onestep=(_eval_proto == "e_w1"),
        )

        # QA accuracy evaluation (rank0 only, after metrics.csv is written)
        _, _, _, local_rank = _get_env_silence_and_rank()
        qa_eval_enabled = (
            hasattr(self, "_qa_eval_set")
            and bool(getattr(self, "_qa_eval_set", None))
            and _QA_IMPORTS_READY
        )
        if qa_eval_enabled and local_rank == 0:
            try:
                qa_acc = self._eval_qa_accuracy(model, device)
                print(f"[QA] Epoch {epoch_tag} graph_qa_acc: {qa_acc:.4f}")

                metrics_csv = Path(self.run_dir) / "metrics.csv"
                import pandas as pd
                if metrics_csv.exists():
                    df = pd.read_csv(metrics_csv)
                else:
                    df = pd.DataFrame(columns=["epoch"])

                if "graph_qa_acc" not in df.columns:
                    df["graph_qa_acc"] = pd.NA

                ep = int(epoch_tag)
                if "epoch" not in df.columns:
                    df["epoch"] = pd.Series(dtype="Int64")
                try:
                    epoch_col = df["epoch"].astype("Int64")
                except Exception:
                    epoch_col = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
                mask = epoch_col == ep
                if mask.any():
                    df.loc[mask, "graph_qa_acc"] = float(qa_acc)
                else:
                    new_row = {c: pd.NA for c in df.columns}
                    new_row["epoch"] = ep
                    new_row["graph_qa_acc"] = float(qa_acc)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                df.to_csv(metrics_csv, index=False)
            except Exception as e:
                print(f"[QA] Eval failed at epoch {epoch_tag}: {e}")

        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass

        model.train()


    def _eval_qa_accuracy(self, model, device):
        """Evaluate QA accuracy on the fixed eval set."""
        model.eval()
        taxonomy = _dataset_taxonomy(self.eval_dataset)
        if not taxonomy:
            raise RuntimeError("[QA] Eval taxonomy is empty; cannot render graph QA samples")

        qa_ds = GraphQAMiniDataset(
            self._qa_eval_set, taxonomy, self.processor, self.tokenizer,
            reasoning_mode=getattr(self, "_qa_reasoning_mode", "ignore"),
        )
        from functools import partial
        pad_id = self.tokenizer.pad_token_id or 0
        qa_dl = torch.utils.data.DataLoader(
            qa_ds, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=partial(qa_collate_fn, pad_token_id=pad_id),
        )

        # Pre-compute the letter token ids once so we can also report a
        # constrained-letter accuracy (argmax restricted to A/B/C/D) -- this
        # answers "is the model's distribution over the 4 valid options at
        # least leaning the right way" even if it would prefer some other
        # token in unconstrained decoding.
        letter_ids = []
        for letter in ("A", "B", "C", "D"):
            for variant in (letter, " " + letter):
                tid = self.tokenizer.convert_tokens_to_ids(variant)
                if tid is not None and tid != self.tokenizer.unk_token_id:
                    letter_ids.append(tid)
                    break
            else:
                letter_ids.append(-1)
        letter_id_set = set(i for i in letter_ids if i >= 0)
        letter_id_list = [i for i in letter_ids if i >= 0]

        # Match model param dtype for pixel_values to avoid silent dtype-cast
        # corruption on the vision branch.
        model_dtype = next(model.parameters()).dtype

        correct = 0
        constrained_correct = 0
        total = 0
        debug_n = 0
        with torch.no_grad():
            for batch in qa_dl:
                if batch is None:
                    continue
                mm = {}
                for k in ("input_ids", "attention_mask", "labels",
                           "pixel_values", "image_grid_thw"):
                    v = batch.get(k)
                    if isinstance(v, torch.Tensor):
                        if k == "pixel_values":
                            v = v.to(device=device, dtype=model_dtype)
                        else:
                            v = v.to(device)
                        mm[k] = v
                try:
                    outputs = model(**mm)
                except Exception as e:
                    if debug_n < 3:
                        print(f"[QA][debug] forward raised: {e!r}")
                        debug_n += 1
                    continue
                logits = outputs.logits
                labels = mm["labels"]
                for i in range(labels.size(0)):
                    target_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if len(target_positions) == 0:
                        continue
                    # For assistant-side rationale ablations there may be
                    # multiple supervised tokens (e.g. "Answer: B").  Measure
                    # QA accuracy at the actual option-letter token when it is
                    # present, otherwise fall back to the first supervised token.
                    letter_positions = []
                    if letter_id_set:
                        for p0 in target_positions.tolist():
                            try:
                                if int(labels[i, int(p0)].item()) in letter_id_set:
                                    letter_positions.append(int(p0))
                            except Exception:
                                pass
                    pos = int(letter_positions[-1] if letter_positions else target_positions[0].item())
                    if pos - 1 < 0 or pos - 1 >= logits.size(1):
                        if debug_n < 3:
                            print(f"[QA][debug] pos={pos} but logits len={logits.size(1)} labels len={labels.size(1)}")
                            debug_n += 1
                        continue
                    pred_id = int(logits[i, pos - 1].argmax(-1).item())
                    true_id = int(labels[i, pos].item())
                    if pred_id == true_id:
                        correct += 1
                    if letter_id_list:
                        sub = logits[i, pos - 1, letter_id_list]
                        constrained_pick = int(letter_id_list[int(sub.argmax().item())])
                        if constrained_pick == true_id:
                            constrained_correct += 1
                    if debug_n < 5:
                        try:
                            true_tok = self.tokenizer.convert_ids_to_tokens(true_id)
                            pred_tok = self.tokenizer.convert_ids_to_tokens(pred_id)
                        except Exception:
                            true_tok = pred_tok = "?"
                        in_letters = pred_id in letter_id_set
                        print(
                            f"[QA][debug] sample={debug_n} "
                            f"logits.shape={tuple(logits.shape)} labels.shape={tuple(labels.shape)} "
                            f"pos={pos} true_id={true_id}({true_tok!r}) "
                            f"pred_id={pred_id}({pred_tok!r}) pred_in_ABCD={in_letters}"
                        )
                        debug_n += 1
                    total += 1
                if total >= len(self._qa_eval_set):
                    break

        if total == 0:
            raise RuntimeError("[QA] Eval produced 0 valid samples; refusing to write graph_qa_acc=0")
        if letter_id_list:
            print(
                f"[QA][debug] summary total={total} unconstrained_acc={correct/total:.4f} "
                f"constrained_letter_acc={constrained_correct/total:.4f} "
                f"letter_ids={letter_id_list}"
            )
        return correct / total


# ============================================================================
# Trainer
# ============================================================================

class ProActTrainer(Trainer):
    """
    ProAct 2.0 E2E Trainer.

    Single forward per step (not two-stage).
    Memory tokens injected via embedding hook.
    Graph dropout and structural supervision handled here.
    """

    def __init__(
        self,
        *,
        base_collator,
        proact_memory: ProActMemory,
        ep_token_ids: List[int],
        str_token_ids: List[int],
        proc_token_ids: List[int],
        graph_token_ids: List[int],
        short_window: int,
        use_graph: bool,
        struct_loss_weight: float = 0.1,
        bind_trigger_task: bool = False,
        bind_task_step: bool = False,
        bind_tt_weight: float = 0.3,
        bind_ts_weight: float = 0.5,
        bind_projection_dim: int = 0,
        perception_anchor_weight: float = 0.0,
        no_proact_memory: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proact_memory = proact_memory
        self.base_collator = base_collator
        self.ep_token_ids = ep_token_ids
        self.str_token_ids = str_token_ids
        self.proc_token_ids = proc_token_ids
        self.graph_token_ids = graph_token_ids
        self.short_window = short_window
        self.use_graph = use_graph
        self.struct_loss_weight = struct_loss_weight
        self._perception_anchor_weight = float(perception_anchor_weight or 0.0)
        self._graph_conditional_struct = False
        self._graph_mask_str = False
        self._no_proact_memory = bool(no_proact_memory)

        self._bind_trigger_task = bool(bind_trigger_task)
        self._bind_task_step = bool(bind_task_step)
        self._bind_tt_weight = float(bind_tt_weight)
        self._bind_ts_weight = float(bind_ts_weight)
        self._bind_projection_dim = int(bind_projection_dim)
        self._bind_proj_head = None
        self._bind_temperature = 0.07

        # QA joint training state
        self._qa_dataloader = None
        self._qa_iterator = None
        self._qa_loss_weight = 0.5
        self._qa_collate_fn = None
        self._qa_reasoning_mode = "ignore"

    def _get_or_create_bind_proj_head(self, hidden_dim, device):
        if self._bind_proj_head is None:
            dim = self._bind_projection_dim
            self._bind_proj_head = nn.Sequential(
                nn.Linear(hidden_dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            ).to(device)
        return self._bind_proj_head

    def create_optimizer(self):
        super().create_optimizer()
        mem_params = [
            p for p in self.proact_memory.parameters() if p.requires_grad
        ]
        if mem_params:
            self.optimizer.add_param_group({
                "params": mem_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            })
        if self._bind_proj_head is not None:
            proj_params = [
                p for p in self._bind_proj_head.parameters() if p.requires_grad
            ]
            if proj_params:
                self.optimizer.add_param_group({
                    "params": proj_params,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                })
        return self.optimizer

    def set_qa_dataloader(self, dataloader, loss_weight=0.5, collate_fn=None):
        self._qa_dataloader = dataloader
        self._qa_loss_weight = loss_weight
        self._qa_collate_fn = collate_fn
        self._qa_iterator = None

    def _get_qa_batch(self, device):
        if self._qa_dataloader is None:
            return None
        if self._qa_iterator is None:
            self._qa_iterator = iter(self._qa_dataloader)
        try:
            batch = next(self._qa_iterator)
        except StopIteration:
            self._qa_iterator = iter(self._qa_dataloader)
            batch = next(self._qa_iterator)
        if batch is None:
            return None
        prepared = {}
        for key in ("input_ids", "attention_mask", "labels",
                     "pixel_values", "image_grid_thw"):
            v = batch.get(key)
            if isinstance(v, torch.Tensor):
                prepared[key] = v.to(device)
        return prepared

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size()
            for p in self.proact_memory.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                p.grad.div_(ws)
            if self._bind_proj_head is not None:
                for p in self._bind_proj_head.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    torch.distributed.all_reduce(
                        p.grad, op=torch.distributed.ReduceOp.SUM
                    )
                    p.grad.div_(ws)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "chunks" not in inputs:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys,
            )
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach(), None, None)

    def _prepare_mm_batch(self, batch, device):
        prepared = {}
        for key in ("input_ids", "attention_mask", "labels",
                     "pixel_values", "image_grid_thw"):
            v = batch.get(key)
            if isinstance(v, torch.Tensor):
                prepared[key] = v.to(device)
        return prepared

    @staticmethod
    def _pad_label_list(label_list, target_len: int, device) -> Optional[torch.Tensor]:
        padded = []
        for labels in label_list:
            if not isinstance(labels, torch.Tensor):
                labels = torch.full((target_len,), -100, dtype=torch.long)
            if labels.size(0) < target_len:
                labels = torch.cat([
                    labels,
                    labels.new_full((target_len - labels.size(0),), -100),
                ], dim=0)
            elif labels.size(0) > target_len:
                labels = labels[:target_len]
            padded.append(labels)
        if not padded:
            return None
        out = torch.stack(padded, dim=0).to(device)
        return out if (out != -100).any() else None

    @staticmethod
    def _causal_lm_loss_for_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chunks = inputs["chunks"]
        device = next(model.parameters()).device
        core_model = _unwrap_model(model)
        memory = self.proact_memory

        # Init GRU states
        epi_ids, epi_mask = pad_action_sequences(
            [c["initial_all_ids"] for c in chunks], device=device,
        )
        epi_state = memory.encode_epi(epi_ids, epi_mask)

        proc_ids, proc_mask = pad_action_sequences(
            [c["initial_short_ids"] for c in chunks], device=device,
        )
        proc_state = memory.encode_proc(proc_ids, proc_mask)

        total_loss = torch.zeros((), device=device)
        total_terms = 0
        max_steps = max((len(c["steps"]) for c in chunks), default=0)

        for step_idx in range(max_steps):
            active = [i for i, c in enumerate(chunks) if step_idx < len(c["steps"])]
            if not active:
                continue
            step_group = [chunks[i]["steps"][step_idx] for i in active]

            if self._no_proact_memory:
                step_samples = [dict(e["sample"]) for e in step_group]
                for s in step_samples:
                    s.pop("perception_labels", None)
                    s.pop("decision_labels", None)
                batch = self.base_collator(step_samples)
                mm_inputs = self._prepare_mm_batch(batch, device)
                try:
                    outputs = model(**mm_inputs)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if not _get_env_silence_and_rank()[0]:
                        print("[OOM-skip] step in chunk too large, skipping", flush=True)
                    continue
                total_loss = total_loss + outputs.loss
                total_terms += 1
                continue

            a_epi = epi_state[active]
            a_proc = proc_state[active]

            # Check if this step group uses subgraph (no proc/graph injection)
            any_has_subgraph = any(e.get("has_subgraph", False) for e in step_group)

            # Compute all tokens
            graph_data_list = []
            for entry in step_group:
                if entry["has_graph"] and memory.use_graph and not any_has_subgraph:
                    task_name = entry["task_name"]
                    ds = self.train_dataset if hasattr(self, 'train_dataset') else None
                    if ds and task_name in ds.graph_registry:
                        gd = prepare_graph_tensors(
                            ds.graph_registry[task_name], device=device,
                        )
                        graph_data_list.append(gd)
                    else:
                        graph_data_list.append(None)
                else:
                    graph_data_list.append(None)

            all_tokens = memory.compute_all_tokens(
                a_epi, a_proc,
                graph_data=graph_data_list[0] if (
                    len(graph_data_list) == 1 and graph_data_list[0] is not None
                ) else None,
                training=model.training,
            )

            # Build inject IDs and values
            # ep + str always injected (perception tokens in user prompt)
            _str_tokens = all_tokens["str_tokens"]
            if self._graph_mask_str and any_has_subgraph:
                # Zero out str tokens so the model cannot use latent-state
                # shortcuts; structural info must come from the subgraph image.
                _str_tokens = torch.zeros_like(_str_tokens)
            inject_ids = list(self.ep_token_ids + self.str_token_ids)
            inject_vals = [
                all_tokens["ep_tokens"],
                _str_tokens,
            ]

            # Always inject proc tokens; graph_tokens stay gated on use_graph
            inject_ids.extend(self.proc_token_ids)
            inject_vals.append(all_tokens["proc_tokens"])
            if all_tokens["use_graph"] and all_tokens["graph_tokens"] is not None:
                inject_ids.extend(self.graph_token_ids)
                inject_vals.append(all_tokens["graph_tokens"])

            inject_values = torch.cat(inject_vals, dim=1)
            inject_values = inject_values.to(
                core_model.get_input_embeddings().weight.dtype
            )

            # Forward
            step_samples = [dict(e["sample"]) for e in step_group]
            perception_label_list = [s.pop("perception_labels", None) for s in step_samples]
            batch = self.base_collator(step_samples)
            mm_inputs = self._prepare_mm_batch(batch, device)

            try:
                with inject_memory_token_embeddings(
                    core_model.get_input_embeddings(),
                    memory_token_ids=inject_ids,
                    memory_values=inject_values,
                ):
                    outputs = model(**mm_inputs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if not _get_env_silence_and_rank()[0]:
                    print(f"[OOM-skip] step in chunk too large, skipping",
                          flush=True)
                continue

            total_loss = total_loss + outputs.loss
            if model.training and self._perception_anchor_weight > 0:
                perception_labels = self._pad_label_list(
                    perception_label_list,
                    target_len=mm_inputs["input_ids"].shape[1],
                    device=device,
                )
                if perception_labels is not None:
                    anchor_loss = self._causal_lm_loss_for_labels(
                        outputs.logits, perception_labels,
                    )
                    total_loss = total_loss + self._perception_anchor_weight * anchor_loss
            total_terms += 1

            # Structural supervision (L_prog + L_front)
            if self.struct_loss_weight > 0:
                comp_ids = [e["all_completed_ids"] for e in step_group]
                fut_ids = [e["future_ids"] for e in step_group]
                try:
                    l_prog, l_front = memory.latent_state.compute_supervision(
                        all_tokens["z"], comp_ids, fut_ids,
                    )
                    if self._graph_conditional_struct and any_has_subgraph:
                        # Graph is present: only supervise completion (L_prog).
                        # Frontier info must come from the graph image, not
                        # from str tokens. This prevents the memory shortcut.
                        total_loss = total_loss + self.struct_loss_weight * l_prog
                    else:
                        # No graph: full structural supervision so str tokens
                        # encode frontier as fallback.
                        total_loss = total_loss + self.struct_loss_weight * (
                            l_prog + l_front
                        )
                except Exception:
                    pass

            # Update GRU states
            upd_ids, upd_mask = pad_action_sequences(
                [e["update_actions_after_ids"] for e in step_group],
                device=device,
            )
            new_epi = memory.update_epi(a_epi, upd_ids, upd_mask)
            epi_state = epi_state.clone()
            epi_state[active] = new_epi

            new_proc = memory.update_proc(a_proc, upd_ids, upd_mask)
            proc_state = proc_state.clone()
            proc_state[active] = new_proc

        if total_terms == 0:
            raise RuntimeError("Empty chunk batch")
        total_loss = total_loss / float(total_terms)

        # QA auxiliary loss with annealing and step-skipping
        if model.training and self._qa_dataloader is not None:
            cur_step = getattr(self, "_qa_global_step", 0)
            self._qa_global_step = cur_step + 1
            every_n = getattr(self, "_qa_every_n_steps", 1) or 1
            do_qa = (cur_step % every_n == 0)
            qa_w = self._qa_loss_weight
            if do_qa and hasattr(self, "_qa_anneal_schedule") and self._qa_anneal_schedule:
                start_ep, end_ep = self._qa_anneal_schedule
                cur_epoch = float(getattr(self.state, "epoch", 0) or 0)
                if cur_epoch >= end_ep:
                    do_qa = False
                    qa_w = 0.0
                elif cur_epoch > start_ep:
                    frac = (cur_epoch - start_ep) / max(end_ep - start_ep, 1)
                    qa_w = self._qa_loss_weight * (1.0 - frac)
            if do_qa and qa_w > 0:
                qa_batch = self._get_qa_batch(device)
                if qa_batch is not None:
                    try:
                        qa_outputs = model(**qa_batch)
                        total_loss = total_loss + qa_w * qa_outputs.loss
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()

        return (total_loss, {"loss": total_loss}) if return_outputs else total_loss


# ============================================================================
# Callbacks
# ============================================================================



class QAEpochResampleCallback(TrainerCallback):
    """Resample QA data at the start of each epoch with a new seed."""

    def __init__(self, qa_data, e2e_task_counts, total_qa_budget,
                 taxonomy, processor, tokenizer, min_per_task=4,
                 base_seed=42, qa_loss_weight=0.5, cached_image_dir="",
                 reasoning_mode="ignore", sampling_mode="parallel_policy_only",
                 sample_manifest="", qa_source_path=""):
        self.qa_data = qa_data
        self.e2e_task_counts = e2e_task_counts
        self.total_qa_budget = total_qa_budget
        self.taxonomy = taxonomy
        self.processor = processor
        self.tokenizer = tokenizer
        self.min_per_task = min_per_task
        self.base_seed = base_seed
        self.qa_loss_weight = qa_loss_weight
        self.cached_image_dir = cached_image_dir or None
        self.reasoning_mode = str(reasoning_mode or "ignore")
        self.sampling_mode = str(sampling_mode or "parallel_policy_only")
        self.sample_manifest = str(sample_manifest or "")
        self.qa_source_path = str(qa_source_path or "")
        self.trainer_ref = None

    def set_trainer(self, t):
        self.trainer_ref = t

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.trainer_ref is None or not _QA_IMPORTS_READY:
            return control
        epoch = int(round(float(state.epoch or 0)))
        _, _, _, local_rank = _get_env_silence_and_rank()

        qa_samples = sample_qa_for_epoch(
            self.qa_data, self.e2e_task_counts, self.total_qa_budget,
            min_per_task=self.min_per_task,
            base_seed=self.base_seed, epoch=epoch,
            sampling_mode=self.sampling_mode,
            sample_manifest=self.sample_manifest,
            qa_source_path=self.qa_source_path,
        )
        if local_rank == 0:
            print(f"[QA] Epoch {epoch}: sampled {len(qa_samples)} QA pairs")
        if self.sample_manifest:
            actual_tokens = sum(
                len(
                    self.tokenizer.encode(
                        str(sample.get("answer_text") or ""),
                        add_special_tokens=False,
                    )
                )
                for sample in qa_samples
            )
            expected_tokens = sum(
                int(sample.get("manifest_answer_tokens", -1))
                for sample in qa_samples
            )
            if actual_tokens != expected_tokens:
                raise RuntimeError(
                    f"[QA] manifest supervised-token mismatch: "
                    f"actual={actual_tokens} expected={expected_tokens}"
                )
            tuple_sequence = "\n".join(
                str(sample.get("tuple_id") or "") for sample in qa_samples
            )
            tuple_hash = hashlib.sha256(tuple_sequence.encode("utf-8")).hexdigest()
            if local_rank == 0:
                print(
                    f"[QA] frozen manifest={qa_samples[0]['manifest_sha256']} "
                    f"tuple_sequence_sha256={tuple_hash} "
                    f"supervised_answer_tokens={actual_tokens}"
                )

        qa_ds = GraphQAMiniDataset(
            qa_samples, self.taxonomy, self.processor, self.tokenizer,
            cached_image_dir=self.cached_image_dir,
            reasoning_mode=self.reasoning_mode,
        )
        valid_probe = sum(
            1 for i in range(min(32, len(qa_ds)))
            if qa_ds[i] is not None
        )
        if valid_probe == 0:
            raise RuntimeError(
                "[QA] sampled QA data produced 0 valid graph-image samples; "
                "check taxonomy/task/current_step matching"
            )
        from torch.utils.data import DataLoader
        from functools import partial
        pad_id = self.tokenizer.pad_token_id or 0
        qa_dl = DataLoader(
            qa_ds, batch_size=1, shuffle=True, num_workers=4,
            collate_fn=partial(qa_collate_fn, pad_token_id=pad_id),
            pin_memory=True, prefetch_factor=4, drop_last=True,
        )
        self.trainer_ref.set_qa_dataloader(
            qa_dl, loss_weight=self.qa_loss_weight,
        )
        return control



def _safe_update_symlink(alias: Path, target: Path) -> None:
    try:
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        alias.symlink_to(target.name)
    except OSError:
        pass


def _update_best_checkpoint_aliases(run_dir: Path, ckpt: Path, epoch_int: int) -> None:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists() or not ckpt.exists():
        return
    try:
        import pandas as pd
        df = pd.read_csv(metrics_csv)
    except Exception:
        return
    if "epoch" not in df.columns or df.empty:
        return
    try:
        epoch_col = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
        row_df = df.loc[epoch_col == int(epoch_int)]
    except Exception:
        return
    if row_df.empty:
        return
    row = row_df.iloc[-1]

    def val(name, default=0.0):
        try:
            x = row.get(name, default)
            if pd.isna(x):
                return default
            return float(x)
        except Exception:
            return default

    # Higher is better.  Decision score prioritizes future rollout and saved-step
    # behavior while still accounting for APA.  Perception score tracks the
    # upstream trigger/task/step recognition surface.
    decision_score = (
        -val("future_edit_dist")
        + val("onestep_immediate_saved_rate")
        + val("onestep_APA")
    )
    perception_score = (
        val("trig_mAcc") + val("trig_mF1") + val("trig_Acc") + val("trig_F1")
        + val("task_mAcc") + val("task_mF1") + val("task_Acc") + val("task_F1")
        + val("step_mAcc") + val("step_mF1") + val("step_Acc") + val("step_F1")
    )

    state_path = run_dir / "best_checkpoints.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except Exception:
        state = {}

    updates = {
        "best_decision": decision_score,
        "best_perception": perception_score,
    }
    changed = False
    for name, score in updates.items():
        prev = state.get(name, {})
        prev_score = prev.get("score")
        if prev_score is None or float(score) > float(prev_score):
            alias = run_dir / name
            _safe_update_symlink(alias, ckpt)
            state[name] = {
                "epoch": int(epoch_int),
                "global_step": int(ckpt.name.split("-")[-1]) if "-" in ckpt.name else None,
                "checkpoint": ckpt.name,
                "score": float(score),
            }
            changed = True
            print(f"[best] updated {name}: epoch={epoch_int} score={score:.6f} -> {ckpt.name}")
    if changed:
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")




class DiskAwareSaveCallback(TrainerCallback):
    """Switch save root to redacted_platform if <REDACTED_PATH> lacks headroom before checkpoint write."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        self._relocated = False

    def on_save(self, args, state, control, **kwargs):
        _, is_main, _, local_rank = _get_env_silence_and_rank()
        if local_rank != 0 or not is_main:
            return control
        try:
            from utils.checkpoint_dir import ensure_checkpoint_space, used_fallback, active_save_root
            parent = ensure_checkpoint_space(args.output_dir)
            if used_fallback() and not self._relocated:
                new_out = parent / self.run_name
                new_out.mkdir(parents=True, exist_ok=True)
                args.output_dir = str(new_out)
                self._relocated = True
                print(f"[DLV2][disk] relocated output_dir -> {args.output_dir} (fallback={used_fallback()})")
            elif used_fallback():
                print(f"[DLV2][disk] using fallback root {active_save_root()}")
        except Exception as exc:
            print(f"[DLV2][disk] check skipped: {exc}")
        return control

class EveryNEpochCallback(TrainerCallback):
    """
    HF Trainer flow: on_epoch_end -> evaluate -> save -> on_save.
    Generation eval is triggered in on_save so that the checkpoint
    is already persisted before the (slow) generation starts.
    If generation crashes, the checkpoint is safe.
    """

    def __init__(self, n: int, epochs: Optional[set] = None,
                 resume_every: int = 0) -> None:
        self.n = max(1, int(n))
        self.epochs = epochs
        self.resume_every = max(0, int(resume_every))
        self._resume_only_save = False
        self.trainer_ref = None
        self.runner: Optional[ProActGenerationRunner] = None
        self._completed: set = set()
        self._pending_eval_loss: Optional[float] = None

    def set_trainer(self, t):
        self.trainer_ref = t

    def set_runner(self, r):
        self.runner = r

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(float(state.epoch or 0)))
        if epoch <= 0:
            return control
        if self.epochs is not None:
            should = epoch in self.epochs
        else:
            should = epoch % self.n == 0
        if should:
            # DLV2 resume train-only: NO_TRAIN_EVAL=1 skips training-time forward eval
            # (checkpoint still saved). Avoids epoch-boundary distributed eval / NCCL watchdog.
            if os.environ.get("NO_TRAIN_EVAL", "0") != "1":
                control.should_evaluate = True
            control.should_save = True
            self._resume_only_save = False
        elif self.resume_every > 0 and epoch % self.resume_every == 0:
            # Crash resilience only: a mid-run Xid then costs at most
            # resume_every epochs instead of everything since the last milestone.
            control.should_save = True
            self._resume_only_save = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if isinstance(metrics, dict) and "eval_loss" in metrics:
            try:
                self._pending_eval_loss = float(metrics["eval_loss"])
            except Exception:
                self._pending_eval_loss = None
        return control

    def on_save(self, args, state, control, **kwargs):
        _, _, _, local_rank = _get_env_silence_and_rank()
        epoch_int = int(round(float(state.epoch or 0)))

        if getattr(self, "_resume_only_save", False):
            import shutil as _sh

            self._resume_only_save = False
            if local_rank == 0:
                ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if ckpt.exists():
                    if self.trainer_ref is not None:
                        _save_memory_state(self.trainer_ref.proact_memory, ckpt)
                    # Hide it from resolve_ckpt_by_position, which maps the Nth
                    # `checkpoint-*` under the run dir onto the Nth eval_epoch.
                    hold = Path(args.output_dir) / "_resume"
                    try:
                        hold.mkdir(exist_ok=True)
                        dest = hold / ckpt.name
                        if dest.exists():
                            _sh.rmtree(dest, ignore_errors=True)
                        _sh.move(str(ckpt), str(dest))
                        # Keep only the newest; optimizer state is 2.7GB a copy.
                        keep = sorted(hold.glob("checkpoint-*"),
                                      key=lambda p: int(p.name.split("-")[-1]))
                        for old in keep[:-1]:
                            _sh.rmtree(old, ignore_errors=True)
                        print(f"[resume-ckpt] epoch {epoch_int} -> {dest}", flush=True)
                    except Exception as exc:  # never let bookkeeping kill training
                        print(f"[resume-ckpt] WARN {exc}", flush=True)
            return control

        if local_rank == 0:
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt.exists() and self.trainer_ref is not None:
                _save_memory_state(self.trainer_ref.proact_memory, ckpt)
                if epoch_int > 0:
                    alias = Path(args.output_dir) / f"epoch_{epoch_int}"
                    _safe_update_symlink(alias, ckpt)

        if epoch_int <= 0 or epoch_int in self._completed:
            return control
        self._completed.add(epoch_int)

        _safe_dist_barrier(f"on_save_ckpt_{state.global_step}")

        if self.runner is not None:
            cli_args = self.runner.args
            if getattr(cli_args, "skip_inline_on_save_eval", True):
                if local_rank == 0:
                    print(
                        f"[on_save] skip_inline_on_save_eval: epoch={epoch_int} "
                        f"(checkpoint saved; run dedicated eval separately)",
                        flush=True,
                    )
                return control
            try:
                if self.trainer_ref is not None:
                    # === DLV2 on_save in-memory inline eval ===
                    # Use GPU weights from the step that just finished training/saving.
                    # Do NOT reload from disk (avoids safetensors write race) and do NOT call
                    # set_adapter() on the DDP wrapper (breaks w2 activation → empty phase2).
                    # w1/w2 stages: adapters were set once at init; training never re-switches.
                    if cli_args.lora_stage == "combined":
                        peft_model = _unwrap_model(self.trainer_ref.model)
                        try:
                            peft_model.set_adapter(["default", "w1", "w2"])
                        except Exception:
                            peft_model.set_adapter("w2")
                    self.trainer_ref.callback_handler.model = self.trainer_ref.model

                train_loss = _latest_logged_train_loss(state)
                print(
                    f"[INLINE EVAL — in-memory] epoch={epoch_int} "
                    f"(using training GPU weights; no disk reload, no DDP set_adapter)",
                    flush=True,
                )
                self.runner.run(
                    epoch_tag=epoch_int,
                    train_loss=train_loss,
                    val_loss=self._pending_eval_loss,
                )
                self._pending_eval_loss = None

                if local_rank == 0:
                    ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    _update_best_checkpoint_aliases(Path(args.output_dir), ckpt, epoch_int)
            except Exception as exc:
                import traceback

                print(
                    f"[on_save] ERROR: inline eval failed at epoch={epoch_int} "
                    f"(training continues): {exc}",
                    flush=True,
                )
                traceback.print_exc()
                self._pending_eval_loss = None
        return control


# ============================================================================
# Save / Load
# ============================================================================

def _save_memory_state(mem: nn.Module, ckpt_dir):
    target = Path(ckpt_dir) / "proact_memory.safetensors"
    state = {k: v.detach().cpu() for k, v in mem.state_dict().items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(target))


def _checkpoint_adapter_readable(path: Path) -> bool:
    """True when adapter_model.safetensors exists and deserializes (not a partial write)."""
    if not path.exists():
        return False
    try:
        with safe_open(str(path), framework="pt") as handle:
            _ = handle.keys()
        return True
    except Exception:
        return False


def _checkpoint_trainer_state_readable(path: Path) -> bool:
    """True when trainer_state.json exists and parses (not a partial write)."""
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _wait_for_checkpoint_ready(ckpt_dir: Path, timeout_s: float = 300.0, poll_s: float = 1.0) -> bool:
    """Poll until adapter_model.safetensors and trainer_state.json are fully written."""
    adapter_path = ckpt_dir / "adapter_model.safetensors"
    state_path = ckpt_dir / "trainer_state.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (
            _checkpoint_adapter_readable(adapter_path)
            and _checkpoint_trainer_state_readable(state_path)
        ):
            return True
        time.sleep(poll_s)
    return (
        _checkpoint_adapter_readable(adapter_path)
        and _checkpoint_trainer_state_readable(state_path)
    )


def _wait_for_checkpoint_file(path: Path, timeout_s: float = 300.0, poll_s: float = 1.0) -> bool:
    """Poll until adapter_model.safetensors is fully written and readable."""
    return _wait_for_checkpoint_ready(path.parent, timeout_s=timeout_s, poll_s=poll_s)


def _load_memory_state(mem: nn.Module, ckpt_dir, *, required=False):
    path = Path(ckpt_dir) / "proact_memory.safetensors"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Memory weights not found: {path}")
        silence, is_main, _, _ = _get_env_silence_and_rank()
        if not silence and is_main:
            print(f"[resume] proact_memory weights missing: {path}")
        return False
    result = mem.load_state_dict(load_file(str(path)), strict=False)
    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main:
        print(
            f"[resume] loaded proact_memory from {path} "
            f"missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    return True




















































def _load_adapter_weights(model, ckpt_dir, *, strict_threshold: int = 100, permissive: bool = False):
    """Reload a previously-saved PEFT adapter (and modules_to_save) into a peft-wrapped model.

    Handles cross-PEFT-version compatibility: older checkpoints save LoRA keys
    without the ``.default.`` adapter-name segment and modules_to_save keys without
    the ``.modules_to_save.`` wrapper prefix.  We normalise checkpoint keys to the
    format the live PEFT model expects before loading, then do a shape-checked copy.
    """
    path = Path(ckpt_dir) / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {path}")

    raw_state = load_file(str(path))
    target = _unwrap_model(model)
    target_sd = target.state_dict()
    target_keys = set(target_sd.keys())

    def _canonicalise(k: str) -> str:
        """Remove PEFT-version-specific segments to get a comparable key."""
        k = k.replace(".modules_to_save.default.", ".")
        k = k.replace(".modules_to_save.", ".")
        k = k.replace(".lora_A.default.", ".lora_A.")
        k = k.replace(".lora_B.default.", ".lora_B.")
        return k

    canon_to_model = {}
    for mk in target_keys:
        ck = _canonicalise(mk)
        canon_to_model.setdefault(ck, mk)

    loaded, shape_mismatch, unmapped = 0, [], []
    for ck, cv in raw_state.items():
        canon = _canonicalise(ck)
        mk = canon_to_model.get(canon)
        if mk is None:
            unmapped.append(ck)
            continue
        if target_sd[mk].shape != cv.shape:
            shape_mismatch.append((ck, tuple(cv.shape), tuple(target_sd[mk].shape)))
            continue
        with torch.no_grad():
            target_sd[mk].copy_(cv.to(target_sd[mk].dtype))
        loaded += 1

    # For shape-mismatched embed_tokens / lm_head, do partial copy (rows that fit)
    partial_healed = []
    for ck, ck_shape, mk_shape in shape_mismatch:
        canon = _canonicalise(ck)
        mk = canon_to_model.get(canon)
        if mk is None:
            continue
        cv = raw_state[ck]
        mv = target_sd[mk]
        if len(cv.shape) == 2 and len(mv.shape) == 2 and cv.shape[1] == mv.shape[1]:
            n_rows = min(cv.shape[0], mv.shape[0])
            with torch.no_grad():
                mv[:n_rows].copy_(cv[:n_rows].to(mv.dtype))
            partial_healed.append((ck, n_rows, cv.shape[0], mv.shape[0]))

    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main:
        total_lora = sum(1 for k in raw_state if "lora_" in k)
        print(
            f"[load_adapter] from {path}\n"
            f"  ckpt_keys={len(raw_state)}  loaded={loaded}  "
            f"shape_mismatch={len(shape_mismatch)}  unmapped={len(unmapped)}  "
            f"partial_healed={len(partial_healed)}  lora_keys_in_ckpt={total_lora}"
        )
        if unmapped:
            print(f"  unmapped (first 5): {unmapped[:5]}")
        if shape_mismatch:
            print(f"  shape_mismatch (first 3): {shape_mismatch[:3]}")
        if partial_healed:
            for ck, nr, cr, mr in partial_healed:
                print(f"  partial_healed: {ck} copied {nr}/{cr} rows (model has {mr})")

    n_lora_unmapped = sum(1 for k in unmapped if "lora_" in k)
    if n_lora_unmapped > 0:
        msg = (
            f"_load_adapter_weights: {n_lora_unmapped} LoRA keys could not be mapped "
            f"to any model key (first 5: {[k for k in unmapped if 'lora_' in k][:5]})"
        )
        if permissive:
            if not silence and is_main:
                print(f"[load_adapter][WARN] {msg}")
        else:
            raise RuntimeError(msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--val_json", type=str, required=True)
    p.add_argument("--frame_root", type=str, required=True)
    p.add_argument("--annotation", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=5)
    p.add_argument("--window_stride", type=int, default=3)
    p.add_argument("--predict_steps", type=int, default=5)
    p.add_argument("--chunk_size", type=int, default=4)
    p.add_argument("--short_window", type=int, default=4)
    p.add_argument("--memory_hidden_size", type=int, default=512)
    p.add_argument("--bottleneck_dim", type=int, default=64)
    p.add_argument("--d_struct", type=int, default=256)
    p.add_argument("--max_image_long_edge", type=int, default=896)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--eval_every_n_epochs", type=int, default=2)
    p.add_argument("--eval_epochs", type=str, default="", help="Comma-separated epochs for eval/save (overrides every-n schedule)")
    p.add_argument("--resume_save_every_epochs", type=int, default=0,
                   help="Persist a resume-only checkpoint every N epochs into "
                        "<run_dir>/_resume. 0 disables. These are hidden from "
                        "resolve_ckpt_by_position and never trigger inline eval.")
    p.add_argument(
        "--skip_inline_on_save_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip generation eval in on_save (default False: inline uses in-memory GPU weights, "
        "no disk reload, no DDP set_adapter for w1/w2). Use --skip_inline_on_save_eval for train-only.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--load_from_checkpoint", type=str, default="")
    p.add_argument("--experiment_tag", type=str, default="proact2")
    p.add_argument("--smoke", action="store_true")
    # ProAct 2.0 specific
    p.add_argument("--use_graph", action="store_true")
    p.add_argument("--num_graph_tokens", type=int, default=4)
    p.add_argument("--graph_dropout_rate", type=float, default=0.5)
    p.add_argument("--struct_loss_weight", type=float, default=0.1)
    # Subgraph (ego-centric task graph)
    p.add_argument("--subgraph_mode", type=str, default="none",
                   choices=["none", "text", "image"])
    p.add_argument("--subgraph_dropout", type=float, default=0.5)
    p.add_argument("--subgraph_render_mode", type=str, default="default",
                   choices=["default", "cross_thread_frontier", "legal_only"],
                   help="Subgraph render filter: default=all legal frontier; "
                        "cross_thread_frontier=highlight only cross-thread legal; "
                        "legal_only=drop future (illegal) nodes from text/image yaml")
    p.add_argument("--subgraph_text_format", type=str, default="legacy",
                   choices=["legacy", "compact_yaml", "structured_json", "json"],
                   help="Text subgraph serializer when subgraph_mode=text: "
                        "legacy=dependency list; compact_yaml/structured_json "
                        "(json alias) from graph_text_format_probe (no gold in text)")
    p.add_argument("--subgraph_text_omit_legal_pool", action="store_true",
                   help="Train-only: drop frontier_names and legal_cross_thread_actions "
                        "from compact_yaml (also GRAPH_TEXT_OMIT_LEGAL_POOL=1)")
    # Binding loss
    p.add_argument("--bind_trigger_task", action="store_true")
    p.add_argument("--bind_task_step", action="store_true")
    p.add_argument("--bind_tt_weight", type=float, default=0.3)
    p.add_argument("--bind_ts_weight", type=float, default=0.5)
    p.add_argument("--bind_projection_dim", type=int, default=0)
    # QA joint training
    p.add_argument("--qa_train_json", type=str, default="",
                   help="Path to graph QA train JSON for joint training")
    p.add_argument("--qa_test_json", type=str, default="",
                   help="Path to graph QA test JSON for eval")
    p.add_argument("--qa_sample_manifest", type=str, default="",
                   help=("Frozen per-seed/per-epoch tuple manifest for matched "
                         "DLV3 QA controls"))
    p.add_argument("--qa_ratio", type=float, default=0.2,
                   help="QA samples as fraction of total (E2E+QA)")
    p.add_argument("--qa_loss_weight", type=float, default=0.5,
                   help="Weight for QA loss relative to E2E loss")
    p.add_argument("--qa_min_per_task", type=int, default=4,
                   help="Minimum QA samples per task per epoch")
    p.add_argument("--qa_eval_size", type=int, default=1000,
                   help="Number of QA samples in fixed eval set")
    p.add_argument("--qa_cached_image_dir", type=str, default="",
                   help="Dir with pre-rendered graph PNGs")
    p.add_argument("--qa_reasoning_mode", type=str, default="ignore",
                   choices=["ignore", "user_hint", "assistant_answer", "assistant_full"],
                   help=("How to use QA JSON reasoning fields. 'ignore' keeps "
                         "regular answer-only QA; 'user_hint' places rationale "
                         "in the user prompt; assistant_* modes reproduce older "
                         "assistant-side rationale ablations."))
    p.add_argument("--qa_format", type=str, default="open",
                   choices=["open", "mcq"],
                   help="QA supervision: open-ended text or MCQ letter")
    p.add_argument("--qa_sampling_mode", type=str, default="parallel_policy_only",
                   choices=["parallel_policy_only", "uniform_all_types", "policy_heavy_mixed", "dlv3_yaml"],
                   help="Graph QA sampling mode for ablations")
    p.add_argument("--qa_anneal_start_epoch", type=int, default=0,
                   help="Epoch at which QA weight begins to decay (0=no annealing)")
    p.add_argument("--qa_anneal_end_epoch", type=int, default=0,
                   help="Epoch at which QA weight reaches 0 (0=no annealing)")
    p.add_argument("--qa_every_n_steps", type=int, default=1,
                   help="Inject QA loss every N training steps (1=every step)")
    p.add_argument("--graph_mask_str", action="store_true",
                   help=("When the subgraph image is present, zero-out str tokens "
                         "before injection so the model cannot use latent-state "
                         "structure shortcuts. When subgraph is absent (dropout), "
                         "str tokens are injected normally as fallback. Combined "
                         "with graph_conditional_struct for maximal effect."))
    p.add_argument("--graph_conditional_struct", action="store_true",
                   help=("When set, L_front (frontier supervision) is only applied "
                         "on steps WITHOUT a subgraph image. Steps WITH a subgraph "
                         "must learn frontier from the graph, making the graph the "
                         "unique source of 'what comes next'. L_prog (completion) "
                         "is always applied. This prevents memory tokens from "
                         "short-circuiting the graph."))
    p.add_argument("--perception_anchor_weight", type=float, default=0.0,
                   help=("Auxiliary CE weight on trigger/task/step fields only. "
                         "Keeps stage-2 graph tuning from drifting perception "
                         "without reducing graph supervision or constraining decision fields."))
    # Resume
    p.add_argument("--resume_from_checkpoint", type=str, default="")
    p.add_argument("--train_jump", action="store_true")
    # DLV2 dual-LoRA v2.2
    p.add_argument("--lora_stage", type=str, default="",
                   choices=["", "w1", "w2", "combined"],
                   help="Train w1 (perception) or w2 (decision) LoRA adapter")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze loaded checkpoint adapter; train only lora_stage adapter")
    p.add_argument("--no_proact_memory", action="store_true",
                   help="Skip ProActMemory injection (w1 protocol alignment)")
    p.add_argument("--future_actions", action="store_true",
                   help="Train CE on future_steps (W1 always; W2 only when set)")
    p.add_argument("--w1_checkpoint", type=str, default="",
                   help="B4 combined eval: w1 LoRA checkpoint path")
    p.add_argument("--w2_checkpoint", type=str, default="",
                   help="B4 combined eval: w2 LoRA checkpoint path")
    p.add_argument("--max_train_steps", type=int, default=0,
                   help="Cap training steps (0=full epoch; B0 smoke)")
    p.add_argument("--eval_protocol", type=str, default="e2e",
                   choices=["e2e", "e_w1", "e_w2"],
                   help="Generation eval: e_w1=phase1 perception only; e_w2=GT oracle phase2; e2e=default two-phase pred")
    p.add_argument("--onestep_action_selector", type=str, default="model",
                   choices=["model", "entropy", "greedy", "random", "llm"],
                   help="Onestep APA/SS: model=use pred_next_action; entropy=legacy min-entropy on pred_future_steps")
    # W2g history injection
    p.add_argument("--history_memory_mode", type=str, default="none",
                   choices=["none", "recent_only", "set_only", "oracle_past"],
                   help="History text mode for E2E prompt (none=disabled; recent_only=last K steps; oracle_past=all+recent)")
    p.add_argument("--history_recent_k", type=int, default=12,
                   help="Number of recent completed steps to include when history_memory_mode=recent_only")
    p.add_argument("--thread_hint_mode", type=str, default="none",
                   choices=["none", "oracle_cross_thread", "cross_thread_legal_pool"],
                   help="Inject oracle human-thread hint before decision generation "
                        "(W2h=oracle_cross_thread; E1=cross_thread_legal_pool surfaces the "
                        "cross-thread legal-now action pool)")
    p.add_argument("--prompt_variant", type=str, default="baseline",
                   choices=["baseline", "C1", "C2", "C1D1", "C1C2", "D1", "D3", "E2"],
                   help="Append eval-tested guidance to system prompt (W2i: D1)")
    p.add_argument("--next_action_label_mode", type=str, default="teacher",
                   choices=["teacher", "apa_parallel", "ss_aligned", "metric_aligned", "saved_cross_strict"],
                   help=("Relabel next_action CE target: teacher=entropy teacher; "
                         "apa_parallel=cross-thread legal; ss_aligned=immediate_saved; "
                         "metric_aligned=APA+SS joint (W2i)"))
    p.add_argument("--distill_label_map", type=str, default="",
                   help=("Path to JSON {idx: pred_next_action} teacher-prediction map for "
                         "context-distillation students. Empty (default)=no effect. "
                         "Applies to train_dataset only, never val_dataset."))
    p.add_argument("--apa_wait_token", type=str, default=DEFAULT_WAIT_ACTION,
                   help="Wait label for next_action when no legal APA/SS action (match eval WAIT_ACTION)")
    p.add_argument("--future_steps_target", type=str, default="horizon",
                   choices=["horizon", "remaining"],
                   help="future_steps CE target: horizon=JSONL K-step slice; remaining=full human remaining + Terminate")
    return p.parse_args()


# ============================================================================
# Run name builder
# ============================================================================

def build_proact_run_name(args):
    class _V:
        def __init__(self, ns):
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
            self.history_memory_mode = getattr(ns, "history_memory_mode", "none")
            self.experiment_tag = ns.experiment_tag
    base = build_run_name_from_args(_V(args))
    graph_tag = "graphOn" if args.use_graph else "graphOff"
    sg_tag = ""
    if args.subgraph_mode != "none":
        sg_tag = f"_sg{args.subgraph_mode}"
    name = (
        f"{base}_e2e_{graph_tag}{sg_tag}"
        f"_bn{args.bottleneck_dim}_ds{args.d_struct}"
        f"_chunk{args.chunk_size}_sw{args.short_window}"
        f"_mh{args.memory_hidden_size}"
    )
    return name



def _resolve_hidden_size(model):
    """Qwen2.x has config.hidden_size; Qwen3-VL nests it under text_config."""
    cfg = getattr(model, "config", None)
    for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)):
        if obj is None:
            continue
        hs = getattr(obj, "hidden_size", None)
        if hs:
            return int(hs)
    emb = model.get_input_embeddings()
    weight = getattr(emb, "weight", None)
    if weight is not None and getattr(weight, "ndim", 0) >= 2:
        return int(weight.shape[-1])
    raise AttributeError("cannot resolve VLM hidden_size from config or embeddings")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    silence, is_main, _, local_rank = _get_env_silence_and_rank()

    if not silence and is_main:
        print("=" * 70)
        print("ProAct 2.0 E2E Training")
        print("=" * 70)
        print(f"  Episodic (GRU_long) -> ep_tokens (prompt prefix)")
        print(f"  Procedural (GRU_short+BN{args.bottleneck_dim}d) -> proc_tokens (mid-seq)")
        print(f"  Latent State Estimator (d_struct={args.d_struct}) -> str_tokens (prompt prefix)")
        if args.use_graph:
            print(f"  Graph Encoder -> {args.num_graph_tokens} graph_tokens (mid-seq, dropout={args.graph_dropout_rate})")
        if args.subgraph_mode != "none":
            print(f"  Subgraph mode: {args.subgraph_mode} (dropout={args.subgraph_dropout})")
        print(f"  struct_loss_weight={args.struct_loss_weight}")
        print("=" * 70)
        if args.lora_stage:
            print(f"  DLV2 lora_stage={args.lora_stage} freeze_backbone={args.freeze_backbone}")

    # Load model and processor
    model_path = args.model_name
    tokenizer_path = model_path
    if args.load_from_checkpoint:
        ckpt_path = Path(args.load_from_checkpoint)
        has_processor_config = (
            (ckpt_path / "preprocessor_config.json").exists()
            or (ckpt_path / "processor_config.json").exists()
        )
        has_tokenizer_config = (ckpt_path / "tokenizer_config.json").exists()
        has_model_config = (ckpt_path / "config.json").exists()
        if has_model_config and (has_processor_config or has_tokenizer_config):
            tokenizer_path = args.load_from_checkpoint
        elif not silence and is_main:
            print(
                f"[processor] {args.load_from_checkpoint} has no full processor/tokenizer config; "
                f"using base model processor from {model_path}"
            )
    processor = AutoProcessor.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False,
    )
    tokenizer = processor.tokenizer

    # Add all memory special tokens
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in ALL_MEMORY_TOKENS if t not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if not silence and is_main:
            print(f"Added special tokens: {new_tokens}")

    # Load VLM
    lr = int(os.environ.get("LOCAL_RANK", "-1") or "-1")
    kw = {"torch_dtype": torch.bfloat16 if args.bf16 else torch.float32}
    if args.load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if lr >= 0 and torch.cuda.is_available():
        kw["device_map"] = {"": lr}

    cls = Qwen3VLForConditionalGeneration if "Qwen3" in model_path else Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(model_path, **kw)

    if args.load_in_4bit:
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except TypeError:
            model = prepare_model_for_kbit_training(model)

    model.resize_token_embeddings(len(tokenizer))

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)

    if args.load_from_checkpoint:
        _load_adapter_weights(model, args.load_from_checkpoint, permissive=False)
    # === DLV2 dual-LoRA patch ===
    if args.lora_stage in ("w1", "w2", "combined"):
        from train.cot_sft_v2.dual_adapter_utils import (
            has_adapter,
            set_active_adapter_trainable,
        )
        frozen_name = "default"

        if args.lora_stage == "combined":
            if args.freeze_backbone:
                for _n, _p in model.named_parameters():
                    _p.requires_grad = False
            for stage_name in ("w1", "w2"):
                if not has_adapter(model, stage_name):
                    model.add_adapter(stage_name, lora_config)
            try:
                model.set_adapter([frozen_name, "w1", "w2"])
            except Exception:
                model.set_adapter("w2")
            for _n, _p in model.named_parameters():
                _p.requires_grad = False
            if getattr(args, "w1_checkpoint", ""):
                _load_stage_adapter_weights(model, args.w1_checkpoint, "w1")
                # A1 patch: also load W1's embed/lm_head (v1 bare heads) into w1 slot
                try:
                    from eval_loader import load_modules_to_save_for_eval as _lmts_w1
                    _rep_w1 = _lmts_w1(model, args.w1_checkpoint, "w1")
                    if is_main and not silence:
                        print(f"[A1-patch] w1 head load: {_rep_w1}", flush=True)
                except Exception as _e_w1h:
                    print(f"[A1-patch] w1 head load FAILED: {_e_w1h}", flush=True)
            if getattr(args, "w2_checkpoint", ""):
                _load_stage_adapter_weights(model, args.w2_checkpoint, "w2")
            args.no_proact_memory = False
            args.qa_train_json = ""
            args.qa_test_json = ""
            if not silence and is_main:
                print(f"[DLV2] lora_stage=combined w1={getattr(args, 'w1_checkpoint', '')} "
                      f"w2={getattr(args, 'w2_checkpoint', '')}")
        else:
            # === DLV2 combined control-flow repair ===
            stage_name = args.lora_stage
            if args.freeze_backbone:
                for _n, _p in model.named_parameters():
                    _p.requires_grad = False
            if not has_adapter(model, stage_name):
                model.add_adapter(stage_name, lora_config)
            try:
                model.set_adapter([frozen_name, stage_name])
            except Exception:
                model.set_adapter(stage_name)
            set_active_adapter_trainable(model, stage_name)
            if not silence and is_main:
                print(f"[DLV2] lora_stage={stage_name} freeze_backbone={args.freeze_backbone} "
                      f"adapters={[frozen_name, stage_name]}")
            if args.lora_stage == "w1":
                args.subgraph_mode = "none"
                args.qa_train_json = ""
                args.qa_test_json = ""
                args.struct_loss_weight = 0.0
                # W1': proc ON (W1a) or --no_proact_memory (W1b) via CLI
            if args.lora_stage == "w2":
                # Allow CLI --no_proact_memory to disable proc for ablation; default=proc enabled
                if not getattr(args, "no_proact_memory", False):
                    args.no_proact_memory = False  # ensure proc enabled by default
                if not (getattr(args, "qa_train_json", "") or "").strip():
                    args.qa_train_json = ""
                    args.qa_test_json = ""

    label_mode = str(getattr(args, "next_action_label_mode", "teacher") or "teacher").strip().lower()
    _sg_text_fmt = str(getattr(args, "subgraph_text_format", "legacy") or "legacy").strip().lower()
    _sg_omit_legal = bool(getattr(args, "subgraph_text_omit_legal_pool", False)) or os.environ.get("GRAPH_TEXT_OMIT_LEGAL_POOL", "").strip().lower() in ("1", "true", "yes", "on")
    if _sg_text_fmt == "json":
        _sg_text_fmt = "structured_json"
    if label_mode == "metric_aligned" and args.lora_stage == "w2" and not args.future_actions:
        args.future_actions = True
        if not silence and is_main:
            print("[W2i] metric_aligned: enabling --future_actions for ED (future_steps CE)")

    # Datasets
    train_dataset = ProActE2EDataset(
        jsonl_path=args.train_json,
        model_name=args.model_name,
        processor=processor,
        tokenizer=tokenizer,
        frame_root=args.frame_root,
        annotation_path=args.annotation,
        chunk_size=args.chunk_size,
        short_window=args.short_window,
        predict_steps=args.predict_steps,
        max_image_long_edge=args.max_image_long_edge,
        use_graph=args.use_graph,
        window_size=args.window_size,
        window_stride=args.window_stride,
        subgraph_mode=args.subgraph_mode,
        subgraph_dropout=args.subgraph_dropout,
        subgraph_render_mode=getattr(args, "subgraph_render_mode", "default"),
        subgraph_text_format=_sg_text_fmt,
        subgraph_text_omit_legal_pool=_sg_omit_legal,
        lora_stage=args.lora_stage,
        future_actions=getattr(args, "future_actions", False),
        history_memory_mode=getattr(args, "history_memory_mode", "none"),
        history_recent_k=getattr(args, "history_recent_k", 12),
        thread_hint_mode=getattr(args, "thread_hint_mode", "none"),
        prompt_variant=getattr(args, "prompt_variant", "baseline"),
        next_action_label_mode=label_mode,
        apa_wait_token=getattr(args, "apa_wait_token", DEFAULT_WAIT_ACTION),
        annotation_path_for_labels=args.annotation,
        future_steps_target=str(getattr(args, "future_steps_target", "horizon") or "horizon"),
        distill_label_map_path=(str(getattr(args, "distill_label_map", "") or "").strip() or None),
    )
    val_dataset = ProActE2EDataset(
        jsonl_path=args.val_json,
        model_name=args.model_name,
        processor=processor,
        tokenizer=tokenizer,
        frame_root=args.frame_root,
        annotation_path=args.annotation,
        chunk_size=args.chunk_size,
        short_window=args.short_window,
        predict_steps=args.predict_steps,
        max_image_long_edge=args.max_image_long_edge,
        use_graph=args.use_graph,
        window_size=args.window_size,
        window_stride=args.window_stride,
        subgraph_mode=args.subgraph_mode,
        subgraph_dropout=0.0,
        subgraph_render_mode=getattr(args, "subgraph_render_mode", "default"),
        subgraph_text_format=_sg_text_fmt,
        lora_stage=args.lora_stage,
        future_actions=getattr(args, "future_actions", False),
        history_memory_mode=getattr(args, "history_memory_mode", "none"),
        history_recent_k=getattr(args, "history_recent_k", 12),
        thread_hint_mode=getattr(args, "thread_hint_mode", "none"),
        prompt_variant=getattr(args, "prompt_variant", "baseline"),
        next_action_label_mode=label_mode,
        apa_wait_token=getattr(args, "apa_wait_token", DEFAULT_WAIT_ACTION),
        annotation_path_for_labels=args.annotation,
        future_steps_target=str(getattr(args, "future_steps_target", "horizon") or "horizon"),
    )

    num_actions = len(train_dataset.action_name_to_id)
    memory_token_dim = _resolve_hidden_size(model)

    if not silence and is_main:
        print(f"Tokenizer vocab: {len(tokenizer)}, Model embed: {model.get_input_embeddings().weight.shape[0]}")
        print(f"Action vocab: {num_actions}, Memory token dim: {memory_token_dim}")
        print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")
        if train_dataset.graph_registry:
            print(f"Graph registry: {len(train_dataset.graph_registry)} tasks")
        if train_dataset.taxonomy:
            print(f"Taxonomy loaded: {len(train_dataset.taxonomy)} tasks (subgraph_mode={args.subgraph_mode})")

    # Create ProActMemory
    proact_memory = ProActMemory(
        num_actions=num_actions,
        hidden_size=args.memory_hidden_size,
        bottleneck_dim=args.bottleneck_dim,
        short_window=args.short_window,
        memory_token_dim=memory_token_dim,
        ep_slots=len(EP_SLOT_TOKENS),
        str_slots=len(STR_SLOT_TOKENS),
        proc_slots=len(PROC_SLOT_TOKENS),
        d_struct=args.d_struct,
        max_nodes=2000,
        use_graph=args.use_graph,
        num_graph_tokens=args.num_graph_tokens,
        graph_dropout_rate=args.graph_dropout_rate,
    )

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        proact_memory = proact_memory.to(f"cuda:{local_rank}")
    proact_memory = proact_memory.to(dtype=torch.float32)

    # Token IDs
    ep_ids = [tokenizer.convert_tokens_to_ids(t) for t in EP_SLOT_TOKENS]
    str_ids = [tokenizer.convert_tokens_to_ids(t) for t in STR_SLOT_TOKENS]
    proc_ids = [tokenizer.convert_tokens_to_ids(t) for t in PROC_SLOT_TOKENS]
    graph_ids = (
        [tokenizer.convert_tokens_to_ids(t) for t in GRAPH_SLOT_TOKENS]
        if args.use_graph else []
    )

    if not silence and is_main:
        print(f"EP token IDs: {ep_ids}")
        print(f"STR token IDs: {str_ids}")
        print(f"PROC token IDs: {proc_ids}")
        if graph_ids:
            print(f"GRAPH token IDs: {graph_ids}")

    # Run name and output dir
    run_name = build_proact_run_name(args)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not silence and is_main:
        print(f"Output: {run_dir}")

    # Training args
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        save_total_limit=args.save_total_limit,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        report_to="none" if (args.smoke or os.getenv("WANDB_DISABLED", "").lower() in ("true", "1") or os.getenv("WANDB_MODE") == "disabled") else "wandb",
        run_name=run_name,
        ddp_timeout=7200,
        gradient_checkpointing=(os.environ.get("GRAD_CKPT", "1") != "0"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=(os.environ.get("DDP_FIND_UNUSED", "1") != "0"),
        max_steps=(int(args.max_train_steps) if int(getattr(args, "max_train_steps", 0) or 0) > 0 else -1),
    )

    base_collator = make_collate_fn(tokenizer.pad_token_id)
    _eval_epoch_set = None
    if str(getattr(args, "eval_epochs", "") or "").strip():
        _eval_epoch_set = {int(x.strip()) for x in str(args.eval_epochs).split(",") if x.strip()}
    epoch_callback = EveryNEpochCallback(
        args.eval_every_n_epochs, epochs=_eval_epoch_set,
        resume_every=int(getattr(args, "resume_save_every_epochs", 0) or 0))
    disk_callback = None
    if args.lora_stage in ("w1", "w2"):
        disk_callback = DiskAwareSaveCallback(run_name)

    trainer = ProActTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=chunk_collate_fn,
        base_collator=base_collator,
        proact_memory=proact_memory,
        ep_token_ids=ep_ids,
        str_token_ids=str_ids,
        proc_token_ids=proc_ids,
        graph_token_ids=graph_ids,
        short_window=args.short_window,
        use_graph=args.use_graph,
        struct_loss_weight=args.struct_loss_weight,
        bind_trigger_task=args.bind_trigger_task,
        bind_task_step=args.bind_task_step,
        bind_tt_weight=args.bind_tt_weight,
        bind_ts_weight=args.bind_ts_weight,
        bind_projection_dim=args.bind_projection_dim,
        perception_anchor_weight=args.perception_anchor_weight,
        no_proact_memory=args.no_proact_memory,
        callbacks=[epoch_callback],
    )
    epoch_callback.set_trainer(trainer)
    if disk_callback is not None:
        trainer.add_callback(disk_callback)
    if args.graph_conditional_struct:
        trainer._graph_conditional_struct = True
    if args.graph_mask_str:
        trainer._graph_mask_str = True
        os.environ["PROACT_EVAL_GRAPH_MASK_STR"] = "1"
        if not silence and is_main:
            print("[graph_mask_str] str tokens zeroed when subgraph present (train+eval)")
        if not silence and is_main:
            print("[graph_conditional_struct] L_front disabled when subgraph present; graph is unique frontier source")
    trainer._qa_reasoning_mode = args.qa_reasoning_mode
    if args.perception_anchor_weight > 0 and not silence and is_main:
        print(f"[perception_anchor] weight={args.perception_anchor_weight} on trigger/task/step fields")

    # QA joint training setup
    qa_resample_callback = None
    if args.qa_train_json and _QA_IMPORTS_READY:
        with open(args.qa_train_json) as f:
            qa_train_data = json.load(f)

        e2e_task_counts = {}
        for chunk in train_dataset.chunks:
            for step_m in chunk["steps"]:
                tn = step_m.get("task_name", "")
                if tn:
                    e2e_task_counts[tn] = e2e_task_counts.get(tn, 0) + 1

        total_e2e_steps = sum(len(c["steps"]) for c in train_dataset.chunks)
        total_qa_budget = int(total_e2e_steps * args.qa_ratio / (1 - args.qa_ratio))

        if not silence and is_main:
            print(f"[QA] E2E steps/epoch: {total_e2e_steps}")
            print(f"[QA] QA budget/epoch: {total_qa_budget} ({args.qa_ratio:.0%})")
            print(f"[QA] QA loss weight: {args.qa_loss_weight}")
            print(f"[QA] reasoning mode: {args.qa_reasoning_mode}")
            print(f"[QA] sampling mode: {args.qa_sampling_mode}")

        taxonomy = _dataset_taxonomy(train_dataset)
        if not taxonomy:
            raise RuntimeError("[QA] Train taxonomy is empty; cannot render graph QA samples")

        qa_resample_callback = QAEpochResampleCallback(
            qa_data=qa_train_data,
            e2e_task_counts=e2e_task_counts,
            total_qa_budget=total_qa_budget,
            taxonomy=taxonomy,
            processor=processor,
            tokenizer=tokenizer,
            min_per_task=args.qa_min_per_task,
            base_seed=args.seed,
            qa_loss_weight=args.qa_loss_weight,
            cached_image_dir=args.qa_cached_image_dir,
            reasoning_mode=args.qa_reasoning_mode,
            sampling_mode=args.qa_sampling_mode,
            sample_manifest=args.qa_sample_manifest,
            qa_source_path=args.qa_train_json,
        )
        qa_resample_callback.set_trainer(trainer)
        trainer.add_callback(qa_resample_callback)
        if args.qa_anneal_start_epoch > 0 and args.qa_anneal_end_epoch > 0:
            trainer._qa_anneal_schedule = (args.qa_anneal_start_epoch, args.qa_anneal_end_epoch)
            print(f"[QA] Annealing: weight decays from epoch {args.qa_anneal_start_epoch} "
                  f"to 0 at epoch {args.qa_anneal_end_epoch}")
        trainer._qa_every_n_steps = args.qa_every_n_steps
        if args.qa_every_n_steps > 1:
            print(f"[QA] Injecting QA loss every {args.qa_every_n_steps} steps")

    # QA eval setup
    qa_eval_set = None
    if args.qa_test_json and _QA_IMPORTS_READY:
        with open(args.qa_test_json) as f:
            qa_test_data = json.load(f)
        qa_eval_set = build_fixed_eval_qa(
            qa_test_data, total=args.qa_eval_size, seed=12345,
            sampling_mode=args.qa_sampling_mode,
        )
        if not silence and is_main:
            print(f"[QA] Fixed eval set: {len(qa_eval_set)} samples")

    eval_runner = ProActGenerationRunner(
        trainer=trainer,
        run_dir=str(run_dir),
        processor=processor,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        args=args,
    )
    epoch_callback.set_runner(eval_runner)

    # Attach QA eval set to runner
    if qa_eval_set:
        eval_runner._qa_eval_set = qa_eval_set

    # Resume logic
    resume_ckpt = args.resume_from_checkpoint or None
    if resume_ckpt:
        resolved = resolve_resume_checkpoint(resume_ckpt)
        if not resolved:
            resolved = resolve_resume_checkpoint(str(run_dir))
        if not resolved:
            raise FileNotFoundError(f"Cannot resolve resume: {resume_ckpt}")
        resume_ckpt = resolved

    if args.train_jump and not resume_ckpt:
        resolved = resolve_resume_checkpoint(str(run_dir))
        if resolved:
            resume_ckpt = resolved
        else:
            raise ValueError("train_jump needs existing checkpoint")

    if args.train_jump:
        expected_world = int(os.environ.get("WORLD_SIZE", "1") or "1")
        ckpt_map = collect_checkpoint_map(str(run_dir))
        pending = discover_epochs_needing_metrics(
            run_dir=str(run_dir),
            checkpoint_map=ckpt_map,
            resume_epoch=0,
            expected_world=expected_world,
        )
        if not silence and is_main:
            print(f"[train_jump] checkpoints: {sorted(ckpt_map.keys())}")
            print(f"[train_jump] pending metrics: {pending}")

        latest_resume = resume_ckpt
        for epoch in sorted(ckpt_map):
            cp = ckpt_map[epoch]
            latest_resume = cp
            if epoch not in pending:
                continue
            tl, vl = _checkpoint_logged_losses(cp, epoch)
            if args.lora_stage in ("w1", "w2"):
                rep = load_checkpoint_for_eval(
                    trainer.model, cp, stage=args.lora_stage,
                )
                if not silence and is_main:
                    print(f"[train_jump] load_checkpoint_for_eval {rep}", flush=True)
            elif args.lora_stage == "combined":
                if getattr(args, "w1_checkpoint", ""):
                    _load_stage_adapter_weights(trainer.model, args.w1_checkpoint, "w1")
                if getattr(args, "w2_checkpoint", ""):
                    _load_stage_adapter_weights(trainer.model, args.w2_checkpoint, "w2")
                peft_model = _unwrap_model(trainer.model)
                try:
                    peft_model.set_adapter(["default", "w1", "w2"])
                except Exception:
                    peft_model.set_adapter("w2")
            else:
                _load_adapter_weights(trainer.model, cp)
            _load_memory_state(proact_memory, cp, required=False)
            trainer.callback_handler.model = trainer.model
            if not silence and is_main:
                print(f"[train_jump] eval epoch_{epoch}: {cp}")
            eval_runner.run(epoch_tag=epoch, train_loss=tl, val_loss=vl)

        latest_epoch = checkpoint_epoch(latest_resume) or 0
        if latest_epoch >= int(args.num_train_epochs):
            if not silence and is_main:
                print("[train_jump] All epochs done.")
        else:
            _load_adapter_weights(trainer.model, latest_resume)
            _load_memory_state(proact_memory, latest_resume, required=False)
            if not silence and is_main:
                print(f"[train_jump] resume from {latest_resume}")
            trainer.train(resume_from_checkpoint=latest_resume)
    elif resume_ckpt:
        _load_memory_state(proact_memory, resume_ckpt, required=False)
        if not silence and is_main:
            print(f"[resume] from {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    if is_main:
        final_dir = run_dir / "final"
        try:
            trainer.save_model(str(final_dir))
            _save_memory_state(proact_memory, final_dir)
        except Exception as save_exc:
            print(f"[DLV2][save] primary failed: {save_exc}")
            try:
                from utils.checkpoint_dir import resolve_save_root, used_fallback
                fb_parent = resolve_save_root("<REDACTED_PATH>")
                final_dir = fb_parent / "dlv2_b0" / run_dir.name / "final"
                final_dir.parent.mkdir(parents=True, exist_ok=True)
                trainer.save_model(str(final_dir))
                _save_memory_state(proact_memory, final_dir)
                print(f"[DLV2][save] fallback -> {final_dir} (used_fallback={used_fallback()})")
            except Exception as fb_exc:
                print(f"[DLV2][save] fallback also failed: {fb_exc}")
                raise
        print(f"Done! Final model: {final_dir}")


if __name__ == "__main__":
    main()
