#!/usr/bin/env python3
"""Build DLV3-QA-yaml JSON (3 types, no evidence-executable)."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from typing import Any, Dict, List

from train.graph_add.gen_apa_aligned_qa_medium import (
    ANNOTATION,
    _apa_label,
    _get_mgr,
    _is_cross_thread,
    _legal_now,
    _remaining,
)
from train.graph_add.tool.apa_next_action import DEFAULT_WAIT_ACTION, _thread_id


def _same_thread_frontiers(mgr, legal: List[str], human_now: str) -> List[str]:
    ht = _thread_id(mgr, human_now)
    return sorted(a for a in legal if _thread_id(mgr, a) == ht)


def _tuple_id(
    task: str,
    completed: List[str],
    current: str,
    hypothetical_done: str,
    target_after: str,
) -> str:
    payload = {
        "schema": "dlv3_control_tuple.v1",
        "task_name": str(task),
        "completed_steps": sorted(str(step) for step in completed),
        "current_step": str(current),
        "hypothetical_done": str(hypothetical_done),
        "target_after": str(target_after),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_dlv3_questions(rec: dict, mgr_cache: Dict[str, Any]) -> List[dict]:
    task = rec["task_name"]
    completed = list(rec.get("completed_steps", []) or [])
    current = rec.get("current_step", "") or ""
    mgr = _get_mgr(task, ANNOTATION, mgr_cache)
    if mgr is None:
        return []
    future = _remaining(mgr, completed)
    legal = sorted(_legal_now(mgr, completed, future))
    legal = [a for a in legal if a != current]
    cross = [a for a in legal if _is_cross_thread(mgr, a, current)]
    apa = _apa_label(mgr_cache, task, completed, current, future)
    cf_step = None
    for cand in (cross + legal + future):
        if cand and cand not in completed and cand != current:
            cf_step = cand
            break
    qs: List[dict] = []
    tuple_id = ""
    new_completed: List[str] = []
    apa_after = ""
    if cf_step is not None:
        new_completed = completed + [cf_step]
        new_future = _remaining(mgr, new_completed)
        apa_after = _apa_label(mgr_cache, task, new_completed, current, new_future)
        tuple_id = _tuple_id(task, completed, current, cf_step, apa_after)

    qs.append({
        "type": "next-action-apa",
        "needs_graph": True,
        "question": (
            "根据任务图与对话历史，机器人现在应启动哪一个跨线程动作？"
            "若无合适跨线程动作，答 Wait / None。"
        ),
        "answer_text": apa,
        "label_bad": False,
        "meta": {
            "apa_label": apa,
            "source": "dlv3_yaml",
            "tuple_id": tuple_id,
            "base_completed_steps": completed,
            "hypothetical_done": cf_step,
            "paired_target_after": apa_after,
        },
    })

    if cf_step is not None:
        qs.append({
            "type": "counterfactual-unlock",
            "needs_graph": True,
            "question": (
                f"假设机器人下一步完成了「{cf_step}」。完成后，应启动哪一个跨线程动作"
                "（或 Wait / None）？"
            ),
            "answer_text": apa_after,
            "label_bad": False,
            "meta": {
                "hypothetical_done": cf_step,
                "source": "dlv3_yaml",
                "tuple_id": tuple_id,
                "base_completed_steps": completed,
                "completed_steps_after": new_completed,
                "paired_type": "updated-state-apa",
            },
        })
        qs.append({
            "type": "updated-state-apa",
            "needs_graph": True,
            "question": (
                "图中已完成状态已经更新。根据更新后的任务图与对话历史，机器人"
                "现在应启动哪一个跨线程动作？若无合适动作，答 Wait / None。"
            ),
            "answer_text": apa_after,
            "label_bad": False,
            "meta": {
                "hypothetical_done": cf_step,
                "source": "dlv3_yaml_updated_state",
                "tuple_id": tuple_id,
                "base_completed_steps": completed,
                "completed_steps_override": new_completed,
                "paired_type": "counterfactual-unlock",
            },
        })

    same_thr = _same_thread_frontiers(mgr, legal, current)
    if len(same_thr) >= 2:
        ht = _thread_id(mgr, current)
        pick = apa if apa in same_thr else same_thr[0]
        qs.append({
            "type": "frontier-pick-on-thread",
            "needs_graph": True,
            "question": (
                f"在人类当前线程「{ht}」上，多个可行 frontier 中，应优先执行哪一步？"
            ),
            "answer_text": pick,
            "label_bad": False,
            "meta": {"thread": ht, "candidates": same_thr, "source": "dlv3_yaml"},
        })
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_states", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seed = json.load(open(args.seed_states))
    mgr_cache: Dict[str, Any] = {}
    out_records: List[dict] = []
    seen = set()
    type_counts = collections.Counter()
    for rec in seed:
        key = (rec["task_name"], tuple(sorted(rec.get("completed_steps", []) or [])), rec.get("current_step", ""))
        if key in seen:
            continue
        seen.add(key)
        qs = build_dlv3_questions(rec, mgr_cache)
        if not qs:
            continue
        for q in qs:
            type_counts[q["type"]] += 1
        out_records.append({
            "video_id": rec.get("video_id", ""),
            "task_name": rec["task_name"],
            "completed_steps": rec.get("completed_steps", []),
            "current_step": rec.get("current_step", ""),
            "questions": qs,
        })
    json.dump(out_records, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[dlv3qa-yaml] wrote {len(out_records)} states -> {args.out}")
    print(f"[dlv3qa-yaml] types: {dict(type_counts)} wait_answers={sum(1 for r in out_records for q in r['questions'] if q.get('answer_text')==DEFAULT_WAIT_ACTION)}")


if __name__ == "__main__":
    main()
