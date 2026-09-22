#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate sharded one-step eval Agg states (Scheme C).

Each shard is produced by eval_onestep_end2end.py with --dump_agg_json.
We sum raw counts and recompute the same metrics as Agg.to_row().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _sum_states(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "n": 0,
        "robot_nonwait": 0,
        "robot_prereq_ok": 0,
        "robot_feasible": 0,
        "robot_conflict_head": 0,
        "immediate_saved": 0,
        "robot_parallel_action": 0,
        "entropy_sum": 0.0,
        "entropy_sum_saved": 0.0,
        "human_exec": 0,
        "human_idle": 0,
        "idle_due_to_in_progress": 0,
        "idle_due_to_prereq": 0,
        "detour": 0,
        "cross_detour": 0,
        "switch": 0,
        "cross_detour_threads": {},
    }
    for s in states:
        if not isinstance(s, dict):
            continue
        for k in [
            "n",
            "robot_nonwait",
            "robot_prereq_ok",
            "robot_feasible",
            "robot_conflict_head",
            "immediate_saved",
            "robot_parallel_action",
            "human_exec",
            "human_idle",
            "idle_due_to_in_progress",
            "idle_due_to_prereq",
            "detour",
            "cross_detour",
            "switch",
        ]:
            out[k] += int(s.get(k, 0) or 0)
        out["entropy_sum"] += float(s.get("entropy_sum", 0.0) or 0.0)
        out["entropy_sum_saved"] += float(s.get("entropy_sum_saved", 0.0) or 0.0)
        cdt = s.get("cross_detour_threads") or {}
        if isinstance(cdt, dict):
            for tid, v in cdt.items():
                try:
                    out["cross_detour_threads"][str(tid)] = out["cross_detour_threads"].get(str(tid), 0) + int(v or 0)
                except Exception:
                    continue
    return out


def _to_row(method: str, s: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, int(s.get("n", 0)))
    human_exec = max(1, int(s.get("human_exec", 0)))
    forced_h = _shannon_entropy_from_counts(dict(s.get("cross_detour_threads") or {}))
    forced_ratio = (int(s.get("cross_detour", 0)) / human_exec) if human_exec > 0 else 0.0
    thr_spread = float(forced_h * forced_ratio)
    immediate_saved_rate = int(s.get("immediate_saved", 0)) / n
    robot_nonwait = int(s.get("robot_nonwait", 0))
    avg_entropy = (float(s.get("entropy_sum", 0.0)) / robot_nonwait) if robot_nonwait > 0 else 0.0
    avg_entropy_saved = (float(s.get("entropy_sum_saved", 0.0)) / int(s.get("immediate_saved", 0))) if int(s.get("immediate_saved", 0)) > 0 else 0.0
    avg_entropy_all = float(s.get("entropy_sum", 0.0)) / n
    return {
        "method": method,
        "n_samples": int(s.get("n", 0)),
        "immediate_saved_rate": immediate_saved_rate,
        "effective_avg_entropy": avg_entropy,
        "avg_entropy": avg_entropy,
        "avg_entropy_saved": avg_entropy_saved,
        "avg_entropy_all": avg_entropy_all,
        "APA": (int(s.get("robot_parallel_action", 0)) / n),
        "thr_spread": thr_spread,
        "cross_det": (int(s.get("cross_detour", 0)) / human_exec),
        "detour": (int(s.get("detour", 0)) / human_exec),
        "human_switch": (int(s.get("switch", 0)) / human_exec),
        "human_idle": (int(s.get("human_idle", 0)) / n),
        "idle_due_to_in_progress": (int(s.get("idle_due_to_in_progress", 0)) / n),
        "idle_due_to_prereq": (int(s.get("idle_due_to_prereq", 0)) / n),
        "robot_action_rate": (int(s.get("robot_nonwait", 0)) / n),
        "robot_prereq_ok_rate": (int(s.get("robot_prereq_ok", 0)) / n),
        "robot_conflict_head_rate": (int(s.get("robot_conflict_head", 0)) / n),
        "robot_feasible_rate": (int(s.get("robot_feasible", 0)) / n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=str, required=True)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--metrics_prefix", type=str, default="onestep_metrics",
                    help="Subdirectory name under run_dir containing .agg.json files (default: onestep_metrics)")
    args = ap.parse_args()

    metrics_dir = args.metrics_prefix  # e.g. "onestep_metrics" or "onestep_metrics_no_ucf"

    states_model: List[Dict[str, Any]] = []
    states_base: List[Dict[str, Any]] = []
    for r in range(int(args.world)):
        # Backward/forward compatibility:
        # - old naming: epoch_<epoch>.rank<r>.agg.json
        # - eval_only naming in v2 sync utils: <epoch>.rank<r>.agg.json
        p_epoch = args.run_dir / metrics_dir / f"epoch_{args.epoch}.rank{r}.agg.json"
        p_plain = args.run_dir / metrics_dir / f"{args.epoch}.rank{r}.agg.json"
        p = p_epoch if p_epoch.exists() else p_plain
        if not p.exists():
            raise SystemExit(
                f"[ERROR] missing shard agg json: tried {p_epoch} and {p_plain}"
            )
        obj = json.loads(p.read_text(encoding="utf-8"))
        states_model.append(obj.get("agg_model") or {})
        b = obj.get("agg_baseline")
        if isinstance(b, dict):
            states_base.append(b)

    sum_model = _sum_states(states_model)
    sum_base = _sum_states(states_base) if states_base else None

    rows: List[Dict[str, Any]] = []
    if sum_base is not None and int(sum_base.get("n", 0)) > 0:
        rows.append(_to_row("Entropy", sum_base))
    rows.append(_to_row("Model", sum_model))
    # attach epoch column
    for r in rows:
        r["epoch"] = str(args.epoch)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

