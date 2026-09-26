from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def ordered_suffix_delta(previous: Sequence[str], current: Sequence[str]) -> List[str]:
    prev = [str(x).strip() for x in previous if str(x).strip()]
    cur = [str(x).strip() for x in current if str(x).strip()]
    prefix_len = 0
    max_prefix = min(len(prev), len(cur))
    while prefix_len < max_prefix and prev[prefix_len] == cur[prefix_len]:
        prefix_len += 1
    return cur[prefix_len:]


def build_video_chunks(
    samples_meta: Sequence[Dict[str, Any]],
    chunk_size: int,
    history_window: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples_meta:
        vid = str(sample["video_id"])
        grouped.setdefault(vid, []).append(dict(sample))

    chunks: List[Dict[str, Any]] = []
    for video_id, samples in grouped.items():
        ordered = sorted(samples, key=lambda item: int(item.get("index", item.get("end", 0))))
        for start in range(0, len(ordered), chunk_size):
            chunk_samples = ordered[start : start + chunk_size]
            if len(chunk_samples) < chunk_size:
                # Drop trailing partial chunks so each sample has identical recurrent unroll length.
                # This avoids rank-divergent step counts under DDP.
                continue
            first_completed = list(chunk_samples[0].get("completed_steps", []) or [])
            if history_window is not None and history_window > 0:
                first_completed = first_completed[-history_window:]
            steps: List[Dict[str, Any]] = []
            for idx, sample in enumerate(chunk_samples):
                next_completed = (
                    list(chunk_samples[idx + 1].get("completed_steps", []) or [])
                    if idx + 1 < len(chunk_samples)
                    else list(sample.get("completed_steps", []) or [])
                )
                current_completed = list(sample.get("completed_steps", []) or [])
                step_entry = dict(sample)
                step_entry["update_actions_after"] = ordered_suffix_delta(current_completed, next_completed)
                steps.append(step_entry)
            chunks.append(
                {
                    "video_id": video_id,
                    "initial_completed_steps": first_completed,
                    "steps": steps,
                }
            )
    return chunks
