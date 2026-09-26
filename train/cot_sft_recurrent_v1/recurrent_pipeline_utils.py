from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from train.graph_add.tool.apa_next_action import resolve_future_steps_label


def hydrate_samples_for_chunking(
    samples_meta: Sequence[Dict[str, Any]],
    lazy_labels: Sequence[Dict[str, Any] | None],
) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for idx, meta in enumerate(samples_meta):
        merged = dict(meta)
        labels = lazy_labels[idx] if idx < len(lazy_labels) else None
        labels = labels or {}
        merged["completed_steps"] = list(
            labels.get("completed_steps", merged.get("completed_steps", []) or [])
        )
        merged["future_steps"] = list(
            labels.get("future_steps", merged.get("future_steps", []) or [])
        )
        merged["next_action"] = str(
            labels.get("next_action", merged.get("next_action", "") or "")
        )
        hydrated.append(merged)
    return hydrated


def build_generation_eval_record(
    *,
    meta: Dict[str, Any],
    pred_is: bool,
    pred_task: str,
    pred_step: str,
    pred_future_steps: Sequence[str],
    pred_next_action: str,
    normalized_text: str,
    generation_time: float,
    pred_task_matched: str = "",
    pred_step_matched: str = "",
    future_steps_target: str = "horizon",
    annotation_path: str = "",
    tg_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    horizon_future = list(meta.get("future_steps", []) or [])
    gt_future = horizon_future
    if bool(meta.get("is_trigger", False)):
        gt_future = resolve_future_steps_label(
            future_steps_target,
            task_name=str(meta.get("task_name", "") or ""),
            completed_steps=list(meta.get("completed_steps", []) or []),
            horizon_future_steps=horizon_future,
            annotation_path=annotation_path,
            tg_cache=tg_cache if tg_cache is not None else {},
            is_trigger=True,
        )

    return {
        "idx": int(meta["index"]),
        "video_id": str(meta["video_id"]),
        "pred_text": normalized_text,
        "raw_pred_text": normalized_text,
        "pred": int(bool(pred_is)),
        "gt": int(bool(meta.get("is_trigger", False))),
        "pred_is_trigger": bool(pred_is),
        "pred_task": pred_task,
        "pred_task_matched": pred_task_matched or pred_task,
        "gt_task": str(meta.get("task_name", "") or ""),
        "pred_step": pred_step,
        "pred_step_matched": pred_step_matched or pred_step,
        "gt_step": str(meta.get("step_name", "") or ""),
        "pred_future_steps": list(pred_future_steps or []),
        "gt_future_steps": list(gt_future or []),
        "pred_next_action": pred_next_action,
        "gt_next_action": str(meta.get("next_action", "") or ""),
        "generation_time": float(generation_time),
    }
