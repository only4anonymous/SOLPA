#!/usr/bin/env python3
"""Generate APA-aligned QA for medium split.

Reframes graph QA from PERCEPTION (saved-binary / blocked-by) to ACTION-SELECTION:
every answer is in the apa_parallel label space (cross-thread legal action ∪
"Wait / None"), computed by the SAME resolver the E-w2 eval target uses
(``train/graph_add/tool/apa_next_action.resolve_next_action_label``).

Three QA types:
  * next-action-apa     : "which action should the robot start now?"  -> apa label
  * counterfactual-unlock: hypothetically complete a frontier step F  -> apa label after
  * evidence-executable  : list frontier (legal_pool) then the chosen cross-thread action

Output schema is identical to graph_qa_train_medium_daa_v1_filtered_sbbb.json so it
drops into qa_sampler / GraphQAMiniDataset unchanged.

Run on gpu8:
  cd <REPO_ROOT>
  PYTHONPATH=. python3 train/graph_add/gen_apa_aligned_qa_medium.py \
      --seed_states data/QA_graph/graph_qa_train_medium_daa_v1_filtered_sbbb.json \
      --out data/QA_graph/graph_qa_train_medium_apaaligned_v1.json
  PYTHONPATH=. python3 train/graph_add/gen_apa_aligned_qa_medium.py \
      --seed_states data/QA_graph/graph_qa_test_medium_daa_v1_filtered_sbbb.json \
      --out data/QA_graph/graph_qa_test_medium_apaaligned_v1.json
"""
from __future__ import annotations

import argparse
import collections
import json
from typing import Any, Dict, List

from train.graph_add.tool.apa_next_action import (
    DEFAULT_WAIT_ACTION,
    resolve_next_action_label,
    _get_mgr,
    _legal_now,
    _is_cross_thread,
)

ANNOTATION = "test/onestep_planning_oracle/models_output/all_annotations.json"


def _remaining(mgr, completed: List[str]) -> List[str]:
    if hasattr(mgr, "get_remaining_steps"):
        try:
            return list(mgr.get_remaining_steps(list(completed)) or [])
        except Exception:
            return []
    return []


def _apa_label(mgr_cache, task, completed, current, future):
    return resolve_next_action_label(
        "apa_parallel",
        task_name=task,
        completed_steps=list(completed),
        current_step=current,
        future_steps=list(future),
        teacher_action="",
        annotation_path=ANNOTATION,
        tg_cache=mgr_cache,
        wait_token=DEFAULT_WAIT_ACTION,
        is_trigger=True,
    )


def build_questions(rec: dict, mgr_cache: Dict[str, Any]) -> List[dict]:
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

    qs: List[dict] = []

    # 1) next-action-apa (core action-selection)
    qs.append({
        "type": "next-action-apa",
        "needs_graph": True,
        "question": (
            "Given the current task graph and history, which single action should the "
            "robot start now? Pick a currently-legal action on a different thread from the "
            "human, or answer 'Wait / None' if none is appropriate."
        ),
        "answer_text": apa,
        "label_bad": False,
        "meta": {
            "apa_label": apa,
            "legal_pool": legal,
            "cross_thread_pool": cross,
            "source": "real",
        },
    })

    # 2) counterfactual-unlock (ties to future_steps -> ED)
    #    Hypothetically complete the first not-done frontier/future step.
    cf_step = None
    for cand in (cross + legal + future):
        if cand and cand not in completed and cand != current:
            cf_step = cand
            break
    if cf_step is not None:
        new_completed = completed + [cf_step]
        new_future = _remaining(mgr, new_completed)
        apa_after = _apa_label(mgr_cache, task, new_completed, current, new_future)
        qs.append({
            "type": "counterfactual-unlock",
            "needs_graph": True,
            "question": (
                f"Suppose the robot completes '{cf_step}' next. After that, which single "
                "action should the robot start? Answer 'Wait / None' if none is appropriate."
            ),
            "answer_text": apa_after,
            "label_bad": False,
            "meta": {
                "hypothetical_done": cf_step,
                "apa_label_after": apa_after,
                "source": "counterfactual",
            },
        })

    # 3) evidence-executable (answer grounded in graph nodes / legal_pool)
    if legal:
        frontier_str = ", ".join(legal)
        ev_answer = f"Executable: {frontier_str} ⇒ {apa}"
        qs.append({
            "type": "evidence-executable",
            "needs_graph": True,
            "question": (
                "List the currently-executable (frontier) actions in the graph, then name "
                "the single cross-thread action the robot should start now (or 'Wait / None')."
            ),
            "answer_text": ev_answer,
            "label_bad": False,
            "meta": {
                "frontier": legal,
                "apa_label": apa,
                "source": "evidence",
            },
        })

    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_states", required=True,
                    help="existing QA json whose (task, completed, current) states are reused")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed = json.load(open(args.seed_states))
    mgr_cache: Dict[str, Any] = {}
    out_records: List[dict] = []
    seen = set()
    type_counts = collections.Counter()
    ans_counts = collections.Counter()

    for rec in seed:
        key = (rec["task_name"], tuple(sorted(rec.get("completed_steps", []) or [])),
               rec.get("current_step", ""))
        if key in seen:
            continue
        seen.add(key)
        qs = build_questions(rec, mgr_cache)
        if not qs:
            continue
        for q in qs:
            type_counts[q["type"]] += 1
            ans_counts[q["answer_text"]] += 1
        out_records.append({
            "video_id": rec.get("video_id", ""),
            "task_name": rec["task_name"],
            "completed_steps": rec.get("completed_steps", []),
            "current_step": rec.get("current_step", ""),
            "questions": qs,
        })

    json.dump(out_records, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[gen] wrote {len(out_records)} records -> {args.out}")
    print(f"[gen] question types: {dict(type_counts)}")
    print(f"[gen] distinct answers: {len(ans_counts)} "
          f"(wait={ans_counts.get(DEFAULT_WAIT_ACTION,0)})")
    print("[gen] top answers:")
    for a, c in ans_counts.most_common(12):
        print(f"        {c:4d}x  {a[:70]!r}")


if __name__ == "__main__":
    main()
