#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate oracle (GT) predictions JSONL for eval_onestep_end2end.py.

For each L2 sample where the GT trigger is positive, write a prediction record
with perfect trigger, task, step, and future_steps derived from GT labels.

Usage:
    python test/onestep_planning/gen_oracle_predictions.py \
        --l1_json data/l1/l1_test_best_exo_fixed.jsonl \
        --annotation /path/to/all_annotations.json \
        --out save/oracle_eval/oracle_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
ONESTEP_DIR = REPO_ROOT / "test" / "onestep_planning"
if str(ONESTEP_DIR) not in sys.path:
    sys.path.insert(0, str(ONESTEP_DIR))

from eval_onestep_end2end import (
    VideoRow,
    Segment,
    build_l2_index,
    build_state_for_sample,
    compress_segments,
    load_l1_rows,
    load_vocabulary_from_annotation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate oracle predictions JSONL")
    parser.add_argument("--l1_json", type=Path, default=Path(REPO_ROOT / "data/l1/l1_test_best_exo_fixed.jsonl"))
    parser.add_argument("--annotation", type=Path, default=Path("<REPO_ROOT>/test/onestep_planning_oracle/models_output/all_annotations.backup.json"))
    parser.add_argument("--window_stride", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path(REPO_ROOT / "save/oracle_eval/oracle_predictions.jsonl"))
    args = parser.parse_args()

    print(f"[GenOracle] Loading L1 test data from {args.l1_json}")
    rows = load_l1_rows(args.l1_json)
    if not rows:
        raise SystemExit(f"[ERROR] failed to load l1 rows from {args.l1_json}")
    print(f"[GenOracle] Loaded {len(rows)} videos")

    metas = build_l2_index(rows, window_stride=args.window_stride)
    vocab = load_vocabulary_from_annotation(str(args.annotation))
    print(f"[GenOracle] Built L2 index: {len(metas)} samples, vocabulary: {len(vocab)} entries")

    by_vid: Dict[str, VideoRow] = {vr.video_id: vr for vr in rows}
    seg_cache: Dict[str, List[Segment]] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.out.open("w", encoding="utf-8") as f:
        for idx, meta in tqdm(enumerate(metas), total=len(metas), desc="Generating oracle preds", ncols=0, dynamic_ncols=True):
            vr = by_vid.get(meta.video_id)
            if vr is None:
                continue
            # Skip UCF-Crime
            if meta.video_id.startswith("UCF_CRIME_"):
                continue

            end = meta.end
            if end < 0 or end >= len(vr.frame_labels):
                continue
            lbl = int(vr.frame_labels[end]) if end < len(vr.frame_labels) else 0

            # Non-trigger frames: write pred=0 so eval sees them as "no trigger"
            if lbl <= 0:
                rec = {
                    "idx": idx,
                    "pred": 0,
                    "pred_is_trigger": False,
                    "pred_task": "",
                    "pred_step": "",
                    "pred_future_steps": [],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                continue

            # Build segments (cached)
            if meta.video_id not in seg_cache:
                seg_cache[meta.video_id] = compress_segments(vr.frame_labels, vr.frame_task_labels, vocab)

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
                skipped += 1
                continue

            rec = {
                "idx": idx,
                "pred": 1,
                "pred_is_trigger": True,
                "pred_task": st.task_name,
                "pred_task_matched": st.task_name,
                "pred_step": st.gt_step_now,
                "pred_step_matched": st.gt_step_now,
                "pred_future_steps": st.human_future_steps_gt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[GenOracle] Written {written} records to {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
