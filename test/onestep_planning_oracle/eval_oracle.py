#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle evaluation: compute one-step planning metrics (APA, immediate_saved_rate, etc.)
when the model has *perfect* perception and *perfect* future action prediction.

This establishes the theoretical upper bound for the pipeline:
  trigger = GT,  task = GT,  step = GT,  future_steps = GT

The script:
1. Loads L1 test data, builds L2 index (sliding window stride=3).
2. For each GT-trigger sample, constructs a PlanningState with GT labels.
3. Generates a "perfect prediction" JSONL that exactly matches the GT labels.
4. Feeds it through the same eval_onestep_end2end evaluation pipeline.

Usage:
    python test/onestep_planning/eval_oracle.py \
        --l1_json data/l1/l1_test_best_exo_fixed.jsonl \
        --annotation /path/to/all_annotations.json \
        --out_dir save/oracle_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Reuse helpers from eval_onestep_end2end
REPO_ROOT = Path(__file__).resolve().parents[2]
ONESTEP_DIR = REPO_ROOT / "test" / "onestep_planning"
if str(ONESTEP_DIR) not in sys.path:
    sys.path.insert(0, str(ONESTEP_DIR))

from eval_onestep_end2end import (
    Agg,
    OneStepOutcome,
    PlanningState,
    Segment,
    VideoRow,
    build_l2_index,
    build_state_for_sample,
    compress_segments,
    load_l1_rows,
    load_vocabulary_from_annotation,
    simulate_one_step,
    write_metrics_csv,
    _is_real_action_node,
    _select_min_entropy_action,
    SampleMeta,
)
from task_planner import TaskGraphManager, EntropyPlanner

from tqdm import tqdm


def collect_future_actions_for_oracle(
    frame_labels: List[int],
    current_idx: int,
    predict_steps: int,
    vocab_map: Dict[int, str],
) -> List[str]:
    """
    Exactly replicates the training-time collect_future_actions logic:
    From current_idx+1 forward, collect up to predict_steps distinct step names
    (skipping consecutive-same labels and non-positive labels).
    """
    if predict_steps <= 0 or not frame_labels:
        return []
    n = len(frame_labels)
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
        if lbl <= 0:
            continue
        if last_label is not None and lbl == last_label:
            continue
        name = vocab_map.get(lbl, "")
        if not name:
            name = str(lbl)
        future_steps.append(str(name).strip())
        last_label = lbl
        if len(future_steps) >= predict_steps:
            break
    return future_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle one-step planning evaluation (GT perception + GT future)")
    parser.add_argument("--l1_json", type=Path, default=Path(REPO_ROOT / "data/l1/l1_test_best_exo_fixed.jsonl"))
    parser.add_argument("--annotation", type=Path, default=Path("<REPO_ROOT>/test/onestep_planning_oracle/models_output/all_annotations.backup.json"))
    parser.add_argument("--window_stride", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5, help="Future horizon K")
    parser.add_argument("--human_mode", type=str, default="hmin", choices=["hmin", "switch", "noswitch"])
    parser.add_argument("--immediate_M", type=int, default=1)
    parser.add_argument("--out_dir", type=Path, default=Path(REPO_ROOT / "save/oracle_eval"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--include_entropy_baseline", action="store_true", default=True)
    args = parser.parse_args()

    print(f"[Oracle Eval] Loading L1 test data from {args.l1_json}")
    rows = load_l1_rows(args.l1_json)
    if not rows:
        raise SystemExit(f"[ERROR] failed to load l1 rows from {args.l1_json}")
    print(f"[Oracle Eval] Loaded {len(rows)} videos")

    metas = build_l2_index(rows, window_stride=args.window_stride)
    vocab = load_vocabulary_from_annotation(str(args.annotation))
    print(f"[Oracle Eval] Built L2 index: {len(metas)} samples, vocabulary: {len(vocab)} entries")

    # Per-task graph cache + per-video segment cache
    graph_cache: Dict[str, TaskGraphManager] = {}
    seg_cache: Dict[str, List[Segment]] = {}
    by_vid: Dict[str, VideoRow] = {vr.video_id: vr for vr in rows}

    def _get_graph(task_name: str) -> Optional[TaskGraphManager]:
        t = (task_name or "").strip()
        if not t:
            return None
        if t not in graph_cache:
            try:
                graph_cache[t] = TaskGraphManager(str(args.annotation), t)
            except Exception as e:
                print(f"[WARN] Cannot load task graph for '{t}': {e}")
                return None
        return graph_cache.get(t)

    agg_oracle = Agg()
    agg_baseline = Agg()

    # Also generate oracle predictions JSONL for inspection
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oracle_pred_path = args.out_dir / "oracle_predictions.jsonl"
    pred_f = oracle_pred_path.open("w", encoding="utf-8")

    evaluated = 0
    skipped_no_graph = 0
    skipped_no_trigger = 0
    skipped_no_remaining = 0

    pbar = tqdm(enumerate(metas), total=len(metas), desc="Oracle eval", ncols=0, disable=False)
    for idx, meta in pbar:
        vr = by_vid.get(meta.video_id)
        if vr is None:
            continue

        # Skip UCF-Crime
        if meta.video_id.startswith("UCF_CRIME_"):
            continue

        end = meta.end

        # Check GT trigger
        if end < 0 or end >= len(vr.frame_labels):
            continue
        lbl = int(vr.frame_labels[end]) if end < len(vr.frame_labels) else 0
        if lbl <= 0:
            skipped_no_trigger += 1
            continue

        # Build segments (cached)
        if meta.video_id not in seg_cache:
            seg_cache[meta.video_id] = compress_segments(vr.frame_labels, vr.frame_task_labels, vocab)

        # Build planning state
        st = build_state_for_sample(
            idx=idx,
            vr=vr,
            end=end,
            vocab=vocab,
            horizon=int(args.horizon),
            append_terminate=True,
            precomputed_segs=seg_cache.get(meta.video_id),
        )
        if st is None:
            continue

        # Require at least one remaining step besides Terminate
        rem = [x for x in (st.human_remaining_gt or []) if x and x.strip() and x.strip().lower() != "terminate"]
        if not rem:
            skipped_no_remaining += 1
            continue

        graph_env = _get_graph(st.task_name)
        if graph_env is None:
            skipped_no_graph += 1
            continue

        # ============================================================
        # Oracle prediction: use GT future_steps
        # ============================================================
        gt_future = st.human_future_steps_gt  # horizon-K from GT

        # For oracle, also compute future_steps using collect_future_actions
        # (same as training code) for cross-validation
        oracle_future = collect_future_actions_for_oracle(
            vr.frame_labels, end, args.horizon, vocab
        )

        # Use the state-based GT future (from segments, more reliable)
        future_for_eval = gt_future if gt_future else oracle_future

        # Write oracle prediction record
        pred_rec = {
            "idx": idx,
            "pred": 1,  # GT trigger = True
            "pred_is_trigger": True,
            "pred_task": st.task_name,
            "pred_task_matched": st.task_name,
            "pred_step": st.gt_step_now,
            "pred_step_matched": st.gt_step_now,
            "pred_future_steps": future_for_eval,
            "gt": 1,
            "gt_task": st.task_name,
            "gt_step": st.gt_step_now,
            "gt_future_steps": future_for_eval,
        }
        pred_f.write(json.dumps(pred_rec, ensure_ascii=False) + "\n")

        # ============================================================
        # Baseline: Entropy selector using GT future + GT legal set
        # ============================================================
        if args.include_entropy_baseline:
            predicted_src = [x for x in (future_for_eval or []) if _is_real_action_node(str(x), graph_env)]
            human_immediate_gt = st.human_remaining_gt[:max(0, int(args.immediate_M))] if st.human_remaining_gt else []
            try:
                legal_raw = graph_env.get_legal_robot_actions(list(st.completed_steps), human_immediate_gt)
                legal_set = set([a for a in (legal_raw or []) if _is_real_action_node(str(a), graph_env)])
            except Exception:
                legal_set = set()
            future_for_entropy = [x for x in predicted_src if x in legal_set]
            a_base, e_base = _select_min_entropy_action(
                graph_env=graph_env,
                completed_steps=list(st.completed_steps),
                candidates=list(future_for_entropy or []),
                immediate_M=int(args.immediate_M),
            )
            o_base = simulate_one_step(
                state=st,
                graph_env=graph_env,
                robot_action=a_base,
                robot_entropy=e_base,
                human_mode=args.human_mode,
                immediate_M=int(args.immediate_M),
            )
            agg_baseline.add(o_base)

        # ============================================================
        # Oracle: same entropy selector as eval_onestep_end2end.py
        # (candidates = GT future steps ∩ legal set, then min entropy)
        # ============================================================
        predicted_src = [x for x in (future_for_eval or []) if _is_real_action_node(str(x), graph_env)]
        human_immediate_gt = st.human_remaining_gt[:max(0, int(args.immediate_M))] if st.human_remaining_gt else []
        try:
            legal_raw = graph_env.get_legal_robot_actions(list(st.completed_steps), human_immediate_gt)
            legal_set = set([a for a in (legal_raw or []) if _is_real_action_node(str(a), graph_env)])
        except Exception:
            legal_set = set()
        cand_inter = [c for c in predicted_src if c in legal_set]
        a_oracle, e_oracle = _select_min_entropy_action(
            graph_env=graph_env,
            completed_steps=list(st.completed_steps),
            candidates=list(cand_inter),
            immediate_M=int(args.immediate_M),
        )
        o_oracle = simulate_one_step(
            state=st,
            graph_env=graph_env,
            robot_action=a_oracle,
            robot_entropy=e_oracle,
            human_mode=args.human_mode,
            immediate_M=int(args.immediate_M),
        )
        agg_oracle.add(o_oracle)

        evaluated += 1
        pbar.set_postfix_str(f"eval={evaluated}")
        if isinstance(args.max_samples, int) and args.max_samples > 0 and evaluated >= args.max_samples:
            break

    pbar.close()
    pred_f.close()

    # ============================================================
    # Also compute frame-level metrics (all perfect for oracle)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Oracle Evaluation Results")
    print(f"{'='*60}")
    print(f"Total L2 samples: {len(metas)}")
    print(f"Evaluated (GT trigger=True, has remaining): {evaluated}")
    print(f"Skipped (no trigger): {skipped_no_trigger}")
    print(f"Skipped (no remaining): {skipped_no_remaining}")
    print(f"Skipped (no graph): {skipped_no_graph}")

    # Write metrics
    metrics_rows: List[Dict[str, Any]] = []
    if args.include_entropy_baseline:
        baseline_row = agg_baseline.to_row("Entropy_Baseline_GT")
        metrics_rows.append(baseline_row)
    oracle_row = agg_oracle.to_row("Oracle_GT")
    metrics_rows.append(oracle_row)

    metrics_path = args.out_dir / "oracle_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)

    # Print results nicely
    print(f"\n--- Oracle (GT perception + GT future) ---")
    print(f"  immediate_saved_rate: {oracle_row['immediate_saved_rate']:.6f}")
    print(f"  effective_avg_entropy: {oracle_row['effective_avg_entropy']:.6f}")
    print(f"  APA:                  {oracle_row['APA']:.6f}")
    print(f"  robot_action_rate:    {oracle_row['robot_action_rate']:.6f}")
    print(f"  robot_feasible_rate:  {oracle_row['robot_feasible_rate']:.6f}")
    print(f"  human_idle:           {oracle_row['human_idle']:.6f}")
    print(f"  detour:               {oracle_row['detour']:.6f}")
    print(f"  cross_det:            {oracle_row['cross_det']:.6f}")
    print(f"  thr_spread:           {oracle_row['thr_spread']:.6f}")

    if args.include_entropy_baseline:
        print(f"\n--- Entropy Baseline (GT, should be identical) ---")
        print(f"  immediate_saved_rate: {baseline_row['immediate_saved_rate']:.6f}")
        print(f"  effective_avg_entropy: {baseline_row['effective_avg_entropy']:.6f}")
        print(f"  APA:                  {baseline_row['APA']:.6f}")
        print(f"  robot_action_rate:    {baseline_row['robot_action_rate']:.6f}")

    # Also compute what "perfect frame-level" metrics would look like
    # (trigger, task, step all perfect, future_edit_dist = 0)
    print(f"\n--- Frame-Level Metrics (Oracle, by definition) ---")
    print(f"  trig_Acc:          1.000000")
    print(f"  trig_F1:           1.000000")
    print(f"  task_Acc:          1.000000")
    print(f"  task_F1:           1.000000")
    print(f"  step_Acc:          1.000000")
    print(f"  step_F1:           1.000000")
    print(f"  future_edit_dist:  0.000000")

    print(f"\n[Output] Metrics CSV:  {metrics_path}")
    print(f"[Output] Predictions:  {oracle_pred_path}")
    print(f"[Done]")


if __name__ == "__main__":
    main()
