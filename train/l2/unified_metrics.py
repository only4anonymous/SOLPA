"""
Unified metrics computation module.

Consolidates all eval metrics logic into a single file so that every training
method (cot_sft, cot_sft_v2, recurrent_v1, ablation_memory, dual_memory, ...)
can call ONE function to compute **all** metrics:

    from train.l2.unified_metrics import compute_all_metrics

    compute_all_metrics(
        run_dir=self.run_dir,
        epoch_tag=epoch_tag,
        args=self.args,
        train_loss=train_loss,
        val_loss=val_loss,
        method_name="ablation",
    )

This replaces the previous pattern where every training script manually:
  1. called merge_global_metrics(...)
  2. called _run_onestep_eval(...)   (subprocess → eval_onestep_end2end.py)
  3. called _restore_onestep_metrics_columns(...)
All three steps are now handled inside compute_all_metrics().

Note: This file MUST remain importable without heavy deps (no torch at module level)
so that offline re-computation scripts can also use it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ====================================================================
#  Part 1 — Classification + Future-Step Metrics  (merge_global_metrics)
# ====================================================================

def _norm_label(val: str) -> str:
    return (val or "").strip().lower()


def _compute_macro_acc_f1(stats_dict: Dict[str, Dict[str, int]]) -> Tuple[float, float]:
    """
    Macro-average Accuracy & F1:
      per-class: acc = tp/(tp+fp+fn), prec = tp/(tp+fp), rec = tp/(tp+fn), f1 = 2pr/(p+r)
      then average across all classes.
    """
    accs: List[float] = []
    f1s: List[float] = []
    for cls_name, counts in stats_dict.items():
        if cls_name is None:
            continue
        tp_cls = counts.get("tp", 0)
        fp_cls = counts.get("fp", 0)
        fn_cls = counts.get("fn", 0)
        denom_acc = tp_cls + fp_cls + fn_cls
        acc_cls = tp_cls / denom_acc if denom_acc > 0 else 0.0
        prec_cls = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0.0
        rec_cls = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0.0
        f1_cls = 2 * prec_cls * rec_cls / (prec_cls + rec_cls) if (prec_cls + rec_cls) > 0 else 0.0
        accs.append(acc_cls)
        f1s.append(f1_cls)
    macro_acc = sum(accs) / len(accs) if accs else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return macro_acc, macro_f1


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Sequence-level edit distance for future-step evaluation."""
    a_norm = ["".join(str(x or "").lower().split()) for x in a if str(x or "").strip()]
    b_norm = ["".join(str(x or "").lower().split()) for x in b if str(x or "").strip()]
    m, n = len(a_norm), len(b_norm)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a_norm[i - 1] == b_norm[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _strip_terminate_steps(steps: List[str]) -> List[str]:
    """Drop terminal sentinel before ED; training may include Terminate while horizon GT may not."""
    return [s for s in steps if str(s or "").strip().lower() != "terminate"]


def merge_global_metrics(
    run_dir: str,
    predict_steps: int = 0,
    epoch_train_val_override: Optional[Dict[str, Tuple[str, str]]] = None,
) -> None:
    """
    Merge eval_pred/epoch_*_rank*.jsonl → metrics.csv with trigger/task/step/
    future_edit_dist metrics.  Preserves extra columns (e.g. onestep_*) that
    already exist in the current metrics.csv.
    """
    eval_dir = os.path.join(run_dir, "eval_pred")
    if not os.path.isdir(eval_dir):
        print(f"[merge_global_metrics] eval_pred dir not found: {eval_dir}")
        os.makedirs(eval_dir, exist_ok=True)

    files = [f for f in os.listdir(eval_dir) if "rank" in f and f.endswith(".jsonl")]
    if not files:
        print("[merge_global_metrics] no prediction files found, skipping")
        return

    pattern = re.compile(r"(?:epoch_)?(?P<epoch>[^._]+)[._]rank\d+\.jsonl$")
    eval_only_pattern = re.compile(r"eval_only[._]rank\d+\.jsonl$")
    epoch_to_files: Dict[str, list] = defaultdict(list)
    for name in files:
        m = pattern.match(name)
        if m:
            epoch_to_files[m.group("epoch")].append(os.path.join(eval_dir, name))
            continue
        if eval_only_pattern.match(name):
            epoch_to_files["eval_only"].append(os.path.join(eval_dir, name))

    if not epoch_to_files:
        print("[merge_global_metrics] no epoch files matched, skipping")
        return

    # ---- preserve extra columns from existing metrics.csv ----
    train_val_map: Dict[str, Tuple[str, str]] = {}
    base_metric_cols = {
        "epoch", "train_loss", "val_loss",
        "trig_mAcc", "trig_mF1", "trig_Acc", "trig_F1",
        "task_mAcc", "task_mF1", "task_Acc", "task_F1",
        "step_mAcc", "step_mF1", "step_Acc", "step_F1",
        "avg_generation_time", "future_edit_dist",
    }
    extra_cols: List[str] = []
    extra_vals_by_epoch: Dict[str, Dict[str, str]] = {}
    orig_metrics = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(orig_metrics):
        try:
            with open(orig_metrics, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fns = list(reader.fieldnames or [])
                for c in fns:
                    if c not in base_metric_cols and c not in extra_cols:
                        extra_cols.append(c)
                for row in reader:
                    ep = str(row.get("epoch", "")).strip()
                    if not ep:
                        continue
                    tl = str(row.get("train_loss", "")).strip()
                    vl = str(row.get("val_loss", "")).strip()
                    train_val_map[str(ep)] = (tl, vl)
                    if extra_cols:
                        extra_vals_by_epoch[str(ep)] = {c: str(row.get(c, "") or "") for c in extra_cols}
        except Exception:
            pass
    if epoch_train_val_override:
        for ep, v in epoch_train_val_override.items():
            train_val_map[str(ep)] = v

    # ---- detect if future steps exist ----
    future_any = predict_steps > 0
    if not future_any:
        for ep_files in epoch_to_files.values():
            if future_any:
                break
            for path in ep_files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            if (rec.get("pred_future_steps") or []) or (rec.get("gt_future_steps") or []):
                                future_any = True
                                break
                except Exception:
                    continue

    # ---- world-size completeness check ----
    try:
        expected_world = int(os.environ.get("WORLD_SIZE", "0"))
    except Exception:
        expected_world = 0
    if expected_world > 0:
        incomplete = [ep for ep, fs in epoch_to_files.items() if len(fs) < expected_world]
        for ep in incomplete:
            print(f"[merge_global_metrics] epoch {ep}: {len(epoch_to_files[ep])}/{expected_world} rank files")
            epoch_to_files.pop(ep, None)
        if not epoch_to_files:
            print("[merge_global_metrics] all epochs incomplete, skipping")
            return

    # ---- build header ----
    header_cols = [
        "epoch", "train_loss", "val_loss",
        "trig_mAcc", "trig_mF1", "trig_Acc", "trig_F1",
        "task_mAcc", "task_mF1", "task_Acc", "task_F1",
        "step_mAcc", "step_mF1", "step_Acc", "step_F1",
        "avg_generation_time",
    ]
    if future_any:
        header_cols.append("future_edit_dist")
    for c in extra_cols:
        if c not in header_cols:
            header_cols.append(c)

    out_csv = os.path.join(run_dir, "metrics.csv")
    with open(out_csv, "w", encoding="utf-8") as wf:
        wf.write(",".join(header_cols) + "\n")

        def _epoch_sort_key(x):
            s = str(x)
            return (0, int(s)) if s.isdigit() else (1, s)

        for ep in sorted(epoch_to_files, key=_epoch_sort_key):
            ep_files = epoch_to_files[ep]
            tp = fp = tn = fn = 0
            task_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            step_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            total_gen_time = 0.0
            gen_count = 0
            future_edit_sum = 0.0
            future_edit_count = 0
            future_enabled = False

            for path in ep_files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            pred_is = int(rec.get("pred", 0))
                            gt = int(rec.get("gt", 0))
                            pred_task = _norm_label(rec.get("pred_task_matched") or rec.get("pred_task") or "")
                            gt_task = _norm_label(rec.get("gt_task") or "")
                            pred_step = _norm_label(rec.get("pred_step_matched") or rec.get("pred_step") or "")
                            gt_step = _norm_label(rec.get("gt_step") or "")
                            gen_t = float(rec.get("generation_time", 0.0))
                            pred_future = rec.get("pred_future_steps") or []
                            gt_future = rec.get("gt_future_steps") or []
                            if pred_future or gt_future:
                                future_enabled = True
                                try:
                                    future_edit_sum += _levenshtein(
                                        _strip_terminate_steps(
                                            pred_future if isinstance(pred_future, list) else []
                                        ),
                                        _strip_terminate_steps(
                                            gt_future if isinstance(gt_future, list) else []
                                        ),
                                    )
                                    future_edit_count += 1
                                except Exception:
                                    pass
                            total_gen_time += gen_t
                            gen_count += 1

                            # trigger confusion matrix
                            if pred_is == 1 and gt == 1:
                                tp += 1
                            elif pred_is == 1 and gt == 0:
                                fp += 1
                            elif pred_is == 0 and gt == 0:
                                tn += 1
                            else:
                                fn += 1

                            # task stats
                            if gt == 1:
                                if pred_is == 1:
                                    if gt_task:
                                        if pred_task == gt_task:
                                            task_stats[gt_task]["tp"] += 1
                                        else:
                                            task_stats[gt_task]["fn"] += 1
                                            if pred_task:
                                                task_stats[pred_task]["fp"] += 1
                                    elif pred_task:
                                        task_stats[pred_task]["fp"] += 1
                                else:
                                    if gt_task:
                                        task_stats[gt_task]["fn"] += 1
                            elif pred_is == 1 and pred_task:
                                task_stats[pred_task]["fp"] += 1

                            # step stats
                            if gt == 1:
                                if pred_is == 1:
                                    if gt_step:
                                        if pred_step == gt_step:
                                            step_stats[gt_step]["tp"] += 1
                                        else:
                                            step_stats[gt_step]["fn"] += 1
                                            if pred_step:
                                                step_stats[pred_step]["fp"] += 1
                                    elif pred_step:
                                        step_stats[pred_step]["fp"] += 1
                                else:
                                    if gt_step:
                                        step_stats[gt_step]["fn"] += 1
                            elif pred_is == 1 and pred_step:
                                step_stats[pred_step]["fp"] += 1
                except Exception as e:
                    print(f"[merge_global_metrics] failed reading {path}: {e}")

            # ---- compute metrics ----
            pos_prec = tp / (tp + fp + 1e-9)
            pos_recall = tp / (tp + fn + 1e-9)
            val_acc = (tp + tn) / max(1, (tp + tn + fp + fn))
            trig_f1_micro = 2 * (pos_prec * pos_recall) / (pos_prec + pos_recall + 1e-9)
            trig_stats = {
                "pos": {"tp": tp, "fp": fp, "fn": fn},
                "neg": {"tp": tn, "fp": fn, "fn": fp},
            }
            trig_macc, trig_mf1 = _compute_macro_acc_f1(trig_stats)

            task_tp = sum(v["tp"] for v in task_stats.values())
            task_fp = sum(v["fp"] for v in task_stats.values())
            task_fn = sum(v["fn"] for v in task_stats.values())
            task_acc = task_tp / max(1, (task_tp + task_fp + task_fn))
            task_prec = task_tp / max(1, (task_tp + task_fp))
            task_rec = task_tp / max(1, (task_tp + task_fn))
            task_f1 = 2 * (task_prec * task_rec) / max(1e-9, (task_prec + task_rec))
            task_macc, task_mf1 = _compute_macro_acc_f1(task_stats)

            step_tp = sum(v["tp"] for v in step_stats.values())
            step_fp = sum(v["fp"] for v in step_stats.values())
            step_fn = sum(v["fn"] for v in step_stats.values())
            step_acc = step_tp / max(1, (step_tp + step_fp + step_fn))
            step_prec = step_tp / max(1, (step_tp + step_fp))
            step_rec = step_tp / max(1, (step_tp + step_fn))
            step_f1 = 2 * (step_prec * step_rec) / max(1e-9, (step_prec + step_rec))
            step_macc, step_mf1 = _compute_macro_acc_f1(step_stats)
            avg_generation_time = total_gen_time / max(1, gen_count)
            future_edit_avg = future_edit_sum / max(1, future_edit_count) if future_enabled else None

            train_loss, val_loss_s = train_val_map.get(str(ep), ("", ""))
            tl_str = str(train_loss) if train_loss not in (None, "") else ""
            vl_str = str(val_loss_s) if val_loss_s not in (None, "") else ""
            fields = [
                ep, tl_str, vl_str,
                f"{trig_macc:.6f}", f"{trig_mf1:.6f}", f"{val_acc:.6f}", f"{trig_f1_micro:.6f}",
                f"{task_macc:.6f}", f"{task_mf1:.6f}", f"{task_acc:.6f}", f"{task_f1:.6f}",
                f"{step_macc:.6f}", f"{step_mf1:.6f}", f"{step_acc:.6f}", f"{step_f1:.6f}",
                f"{avg_generation_time:.6f}",
            ]
            if future_any:
                fields.append(f"{future_edit_avg:.6f}" if future_enabled and future_edit_avg is not None else "")
            if extra_cols:
                ev = extra_vals_by_epoch.get(str(ep), {})
                for c in extra_cols:
                    fields.append(str(ev.get(c, "")))
            wf.write(",".join(fields) + "\n")

    print(f"[merge_global_metrics] metrics written to: {out_csv}")


# ====================================================================
#  Part 2 — One-Step Planning Eval  (APA / Saved / Ent)
# ====================================================================

def _get_repo_root() -> Path:
    """Return the project repo root (two levels above train/l2/)."""
    return Path(__file__).resolve().parents[2]


def _get_onestep_scripts() -> Tuple[Optional[Path], Optional[Path]]:
    """Locate eval_onestep_end2end.py and aggregate_onestep_shards.py."""
    root = _get_repo_root()
    candidates = [
        root / "test" / "onestep_planning" / "eval_onestep_end2end.py",
        root / "test" / "onestep_planning_oracle" / "eval_onestep_end2end.py",
    ]
    eval_script = next((p for p in candidates if p.exists()), None)
    agg_script = root / "test" / "onestep_planning" / "aggregate_onestep_shards.py"
    if not agg_script.exists():
        agg_script = root / "test" / "onestep_planning_oracle" / "aggregate_onestep_shards.py"
    if eval_script is None or not agg_script.exists():
        return None, None
    return eval_script, agg_script


def _get_dist_info() -> Tuple[int, int]:
    """Return (rank, world_size) in a safe manner."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


def _safe_dist_barrier(tag: str = "") -> None:
    """Best-effort distributed barrier."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def run_onestep_eval(
    *,
    run_dir: str | Path,
    epoch_tag: int,
    args: argparse.Namespace,
    method_name: str = "model",
    barrier_prefix: str = "unified",
) -> None:
    """
    Run the one-step planning eval across all ranks (subprocess), then
    aggregate on rank 0.

    After this function returns, run_dir/onestep_metrics/ will contain:
      - per-rank shard files (.csv, .agg.json, .actions.jsonl, .done)
      - epoch_{epoch_tag}.agg.csv  (rank0 aggregated)
      - onestep_metrics.csv (appended with the 3 key metrics)

    Parameters
    ----------
    run_dir : Path to the experiment run directory.
    epoch_tag : Which epoch this eval corresponds to.
    args : Namespace that must contain at minimum:
           val_json, annotation, window_stride, predict_steps
    method_name : A human-readable method name for logging.
    barrier_prefix : Prefix for dist.barrier tags (avoid collision).
    """
    eval_script, agg_script = _get_onestep_scripts()
    if eval_script is None or agg_script is None:
        print("[unified_metrics] one-step eval scripts not found, skipping")
        return

    rank, world = _get_dist_info()
    one_dir = Path(run_dir) / "onestep_metrics"
    one_dir.mkdir(parents=True, exist_ok=True)

    # ---- clean stale shards ----
    shard_csv = one_dir / f"epoch_{epoch_tag}.rank{rank}.csv"
    shard_done = one_dir / f"epoch_{epoch_tag}.rank{rank}.done"
    shard_agg = one_dir / f"epoch_{epoch_tag}.rank{rank}.agg.json"
    shard_actions = one_dir / f"epoch_{epoch_tag}.rank{rank}.actions.jsonl"
    for stale in (shard_csv, shard_done, shard_agg, shard_actions):
        try:
            if stale.exists():
                stale.unlink()
        except Exception:
            pass

    # ---- per-rank eval subprocess ----
    predict_steps = getattr(args, "predict_steps", 0)
    horizon = predict_steps if predict_steps > 0 else 5

    cmd = [
        sys.executable,
        str(eval_script),
        "--pred_path", str(Path(run_dir) / "eval_pred"),
        "--epoch_filter", str(epoch_tag),
        "--method_name", f"{method_name}_epoch{epoch_tag}",
        "--l1_json", str(args.val_json),
        "--annotation", str(args.annotation),
        "--window_stride", str(getattr(args, "window_stride", 3)),
        "--horizon", str(horizon),
        "--append_terminate",
        "--human_mode", "hmin",
        "--action_selector", str(getattr(args, "onestep_action_selector", "model") or "model"),
        "--entropy_candidate_mode", "future",
        "--immediate_M", "1",
        "--shard_rank", str(rank),
        "--shard_world", str(world),
        "--tqdm_position", str(rank),
        "--out_dir", str(run_dir),
        "--metrics_subpath", f"onestep_metrics/epoch_{epoch_tag}.rank{rank}.csv",
        "--dump_agg_json", str(shard_agg),
        "--done_path", str(shard_done),
        "--action_log", str(shard_actions),
    ]
    subprocess.run(cmd, check=True)
    _safe_dist_barrier(f"{barrier_prefix}_onestep_done_{epoch_tag}")

    # ---- rank 0: aggregate shards ----
    if rank == 0:
        # merge action logs
        merged_actions = Path(run_dir) / "planning_actions.jsonl"
        with merged_actions.open("w", encoding="utf-8") as handle:
            for sr in range(world):
                sp = one_dir / f"epoch_{epoch_tag}.rank{sr}.actions.jsonl"
                if not sp.exists():
                    continue
                with sp.open("r", encoding="utf-8") as src:
                    for line in src:
                        handle.write(line)

        # aggregate metrics
        out_csv = one_dir / f"epoch_{epoch_tag}.agg.csv"
        cmd2 = [
            sys.executable,
            str(agg_script),
            "--epoch", str(epoch_tag),
            "--run_dir", str(run_dir),
            "--world", str(world),
            "--out_csv", str(out_csv),
        ]
        subprocess.run(cmd2, check=True)

        if out_csv.exists():
            with out_csv.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                # Find the "Model" row (or last row as fallback)
                model_row = None
                for row in rows:
                    if str(row.get("method", "")).strip().lower() == "model":
                        model_row = row
                        break
                if model_row is None:
                    model_row = rows[-1]

                history_csv = one_dir / "onestep_metrics.csv"
                write_header = not history_csv.exists()
                with history_csv.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["epoch", "immediate_saved_rate", "effective_avg_entropy", "APA"],
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow({
                        "epoch": str(epoch_tag),
                        "immediate_saved_rate": model_row.get("immediate_saved_rate", ""),
                        "effective_avg_entropy": model_row.get("effective_avg_entropy", ""),
                        "APA": model_row.get("APA", ""),
                    })
        print(f"[unified_metrics] onestep metrics merged for epoch={epoch_tag}")


# ====================================================================
#  Part 3 — Restore Onestep Columns to metrics.csv
# ====================================================================

def restore_onestep_metrics_columns(run_dir: str | Path) -> None:
    """
    Read onestep_metrics/onestep_metrics.csv and back-fill the three
    onestep columns into the main metrics.csv (so one CSV has everything).
    """
    base = Path(run_dir)
    metrics_path = base / "metrics.csv"
    one_hist = base / "onestep_metrics" / "onestep_metrics.csv"
    if (not metrics_path.exists()) or (not one_hist.exists()):
        return

    one_map: Dict[str, Tuple[str, str, str]] = {}
    with one_hist.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epoch = str(row.get("epoch", "")).strip()
            if not epoch:
                continue
            one_map[epoch] = (
                str(row.get("immediate_saved_rate", "")).strip(),
                str(row.get("effective_avg_entropy", "")).strip(),
                str(row.get("APA", "")).strip(),
            )
    if not one_map:
        return

    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows or "epoch" not in fields:
        return

    extra_cols = [
        "onestep_immediate_saved_rate",
        "onestep_effective_avg_entropy",
        "onestep_APA",
    ]
    for col in extra_cols:
        if col not in fields:
            fields.append(col)

    changed = False
    for row in rows:
        epoch = str(row.get("epoch", "")).strip()
        if epoch not in one_map:
            continue
        imm, ent, apa = one_map[epoch]
        row["onestep_immediate_saved_rate"] = imm
        row["onestep_effective_avg_entropy"] = ent
        row["onestep_APA"] = apa
        changed = True
    if not changed:
        return

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


# ====================================================================
#  Part 4 — One-Call Entry Point
# ====================================================================

def compute_all_metrics(
    *,
    run_dir: str | Path,
    epoch_tag: int,
    args: argparse.Namespace,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    predict_steps: Optional[int] = None,
    method_name: str = "model",
    barrier_prefix: str = "unified",
    skip_onestep: bool = False,
) -> None:
    """
    One-call metrics computation that every training method can use.

    Steps executed:
      1. [rank 0] merge_global_metrics  → writes trigger/task/step/future metrics to metrics.csv
      2. [all ranks] run_onestep_eval   → subprocess per rank → rank 0 aggregates
      3. [rank 0] restore_onestep_metrics_columns → back-fill APA/Saved/Ent into metrics.csv

    Parameters
    ----------
    run_dir : Experiment run directory (must contain eval_pred/).
    epoch_tag : Integer epoch number.
    args : Namespace with at least: val_json, annotation, window_stride, predict_steps
    train_loss : Optional training loss for this epoch.
    val_loss : Optional validation loss for this epoch.
    predict_steps : Override for args.predict_steps (if None, uses args.predict_steps).
    method_name : Human-readable method name for onestep eval logging.
    barrier_prefix : Prefix for distributed barrier tags.
    skip_onestep : If True, skip the onestep planning eval (useful for L1/L2 only methods).
    """
    rank, world = _get_dist_info()
    run_dir = str(run_dir)

    # --- Step 1: merge classification + future-step metrics (rank 0 only) ---
    if rank == 0:
        override: Optional[Dict[str, Tuple[str, str]]] = None
        if train_loss is not None or val_loss is not None:
            override = {
                str(epoch_tag): (
                    "" if train_loss is None else f"{float(train_loss):.6f}",
                    "" if val_loss is None else f"{float(val_loss):.6f}",
                )
            }
        ps = predict_steps if predict_steps is not None else getattr(args, "predict_steps", 0)
        merge_global_metrics(run_dir, ps, epoch_train_val_override=override)

    _safe_dist_barrier(f"{barrier_prefix}_merge_{epoch_tag}")

    # --- Step 2: one-step planning eval (all ranks) ---
    if not skip_onestep:
        run_onestep_eval(
            run_dir=run_dir,
            epoch_tag=epoch_tag,
            args=args,
            method_name=method_name,
            barrier_prefix=barrier_prefix,
        )
        _safe_dist_barrier(f"{barrier_prefix}_onestep_{epoch_tag}")

        # --- Step 3: restore onestep columns into metrics.csv (rank 0 only) ---
        if rank == 0:
            restore_onestep_metrics_columns(run_dir)

    _safe_dist_barrier(f"{barrier_prefix}_all_done_{epoch_tag}")
