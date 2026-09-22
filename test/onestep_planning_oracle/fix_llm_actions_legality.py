#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix LLM action legality in llm_planning_actions_new.jsonl files.

This script re-validates all LLM actions using the same legal checking logic as
test/planning/planning_eval.py and updates the actions if they are invalid.
Then recomputes metrics_new.csv if it exists.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure we can import our modules
REPO_ROOT = Path(__file__).resolve().parents[2]
ONESTEP_DIR = REPO_ROOT / "test" / "onestep_planning"
PLANNING_DIR = REPO_ROOT / "test" / "planning"
for d in (ONESTEP_DIR, PLANNING_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from task_planner import TaskGraphManager  # noqa: E402
from eval_onestep_end2end import (  # noqa: E402
    load_l1_rows, build_l2_index, build_state_for_sample,
    compress_segments, load_vocabulary_from_annotation,
    _is_real_action_node, write_metrics_csv, Agg, OneStepOutcome,
    simulate_one_step
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Save list of dictionaries to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_and_fix_action(
    record: Dict[str, Any],
    graph_env: TaskGraphManager,
    completed_steps: List[str],
    immediate_M: int = 1,
) -> Dict[str, Any]:
    """
    Validate and fix a single LLM action record using the same legal checking
    logic as test/planning/planning_eval.py.
    
    Returns updated record with corrected action and reason.
    """
    # Extract action and current state info
    raw_action = str(record.get("robot_action_raw", "") or "").strip()
    if not raw_action:
        raw_action = str(record.get("action", "") or "").strip()
    
    # Get human remaining steps for legal check
    # Try to reconstruct from the record context or use reasonable defaults
    remaining_gt = record.get("human_remaining_gt", [])
    if not remaining_gt:
        # Fallback: try to get from task context if available
        remaining_gt = []
    
    # Use immediate_M steps for legal check (align with planning_eval.py)
    human_immediate = remaining_gt[:max(0, int(immediate_M))] if remaining_gt else []
    
    # Get legal actions using same logic as planning_eval.py
    try:
        legal_raw = graph_env.get_legal_robot_actions(completed_steps, human_immediate)
        # Only allow real action nodes for robot (exclude midlevel/Terminate).
        legal_set = set([a for a in (legal_raw or []) if _is_real_action_node(a, graph_env)])
    except Exception:
        legal_set = set()
    
    # Validate the action
    action = raw_action
    forced_wait_reason = ""
    
    if action.strip().lower() in {"wait", "none", "wait/none", "wait / none"}:
        action = "Wait / None"
    elif action == "Wait / None":
        pass  # Already valid
    else:
        # Check if action is in task graph
        node = graph_env.name2node.get(action)
        if node is None:
            action = "Wait / None"
            forced_wait_reason = "action_not_in_task_graph"
        else:
            # Check if it's a real action node
            if not _is_real_action_node(action, graph_env):
                action = "Wait / None"
                forced_wait_reason = "not_action_node"
            else:
                # Check prerequisites
                cond = node.get("activation_condition", "TRUE")
                try:
                    prereq_ok = graph_env.check_condition(cond, completed_steps)
                except Exception:
                    prereq_ok = False
                
                if not prereq_ok:
                    action = "Wait / None"
                    forced_wait_reason = "prereq_not_met"
                # Check if legal (not in human immediate, etc.)
                elif legal_set and action not in legal_set:
                    action = "Wait / None"
                    forced_wait_reason = "not_legal"
    
    # Update record
    updated_record = dict(record)
    updated_record["robot_action_effective"] = action
    updated_record["robot_forced_wait_reason"] = forced_wait_reason
    
    # If we changed the action, update related fields
    if action != raw_action:
        print(f"  Fixed action: {raw_action} -> {action} (reason: {forced_wait_reason})")
    
    return updated_record


def recompute_metrics_if_needed(model_dir: Path, fixed_records: List[Dict[str, Any]]) -> None:
    """Recompute metrics_new.csv if it exists in the model directory."""
    metrics_path = model_dir / "metrics_new.csv"
    if not metrics_path.exists():
        return
    
    print(f"  Recomputing {metrics_path}")
    
    # Group records by method for aggregation
    agg = Agg()
    
    for record in fixed_records:
        # Convert record to OneStepOutcome-like structure for aggregation
        try:
            outcome = OneStepOutcome(
                robot_action_raw=record.get("robot_action_raw", ""),
                robot_action=record.get("robot_action_effective", ""),
                robot_entropy=float(record.get("robot_entropy", 0.0)),
                robot_prereq_ok=1 if record.get("robot_forced_wait_reason", "") == "" and record.get("robot_action_effective", "") != "Wait / None" else 0,
                robot_conflict_head=int(record.get("robot_conflict_head", 0)),
                robot_feasible=int(record.get("robot_feasible", 1)),
                robot_forced_wait_reason=record.get("robot_forced_wait_reason", ""),
                immediate_saved=int(record.get("immediate_saved", 0)) if record.get("robot_action_effective", "") != "Wait / None" else 0,
                human_action=record.get("human_next_sim", ""),
                human_idle=int(record.get("human_idle_sim", 0)),
                idle_due_to_in_progress=int(record.get("idle_due_to_in_progress_sim", 0)),
                idle_due_to_prereq=int(record.get("idle_due_to_prereq_sim", 0)),
                detour=int(record.get("detour_sim", 0)),
                cross_detour=int(record.get("cross_detour_sim", 0)),
                switch=int(record.get("switch_sim", 0)),
                cross_detour_thread="",  # Not available in record
            )
            agg.add(outcome)
        except Exception as e:
            print(f"    Warning: Could not process record {record.get('idx', '?')}: {e}")
            continue
    
    # Extract method name from first record
    method_name = fixed_records[0].get("method", "Unknown") if fixed_records else "Unknown"
    
    # Generate metrics row
    metrics_row = agg.to_row(method_name)
    
    # Write updated metrics
    write_metrics_csv(metrics_path, [metrics_row])
    print(f"  Updated metrics for {method_name}: {metrics_row}")


def main():
    """Main function to fix all llm_planning_actions_new.jsonl files."""
    
    # Load L1 data and annotation for state reconstruction
    l1_path = Path("<REPO_ROOT>/data/l1/l1_test_best_exo_fixed.jsonl")
    annotation_path = Path("<REPO_ROOT>/test/onestep_planning_oracle/models_output/all_annotations.backup.json")
    
    print("Loading L1 data and vocabulary...")
    l1_rows = load_l1_rows(l1_path)
    vocab = load_vocabulary_from_annotation(str(annotation_path))
    metas = build_l2_index(l1_rows, window_stride=3)  # Default window_stride
    
    # Build video lookup
    by_vid = {vr.video_id: vr for vr in l1_rows}
    
    # Cache for task graphs and segments
    graph_cache: Dict[str, TaskGraphManager] = {}
    seg_cache: Dict[str, List] = {}
    
    def get_graph(task_name: str) -> Optional[TaskGraphManager]:
        t = (task_name or "").strip()
        if not t or t in graph_cache:
            return graph_cache.get(t)
        try:
            graph_cache[t] = TaskGraphManager(str(annotation_path), t)
            return graph_cache[t]
        except Exception:
            return None
    
    # Find all llm_planning_actions_new.jsonl files
    models_output_dir = Path("<REPO_ROOT>/test/onestep_planning/models_output")
    
    llm_action_files = list(models_output_dir.glob("*/llm_planning_actions_new.jsonl"))
    
    if not llm_action_files:
        print("No llm_planning_actions_new.jsonl files found.")
        return
    
    print(f"Found {len(llm_action_files)} files to process:")
    for f in llm_action_files:
        print(f"  {f}")
    
    for action_file in llm_action_files:
        model_name = action_file.parent.name
        print(f"\nProcessing {model_name}...")
        
        # Load records
        try:
            records = load_jsonl(action_file)
        except Exception as e:
            print(f"  Error loading {action_file}: {e}")
            continue
        
        if not records:
            print(f"  No records found in {action_file}")
            continue
        
        print(f"  Loaded {len(records)} records")
        
        # Process each record
        fixed_records = []
        changes = 0
        
        for record in records:
            try:
                # Reconstruct state for this record
                idx = record.get("idx")
                video_id = record.get("video_id")
                task_name_gt = record.get("task_name_gt")
                
                if idx is None or not video_id or not task_name_gt:
                    # Keep record as-is if we can't reconstruct context
                    fixed_records.append(record)
                    continue
                
                # Get task graph
                graph_env = get_graph(task_name_gt)
                if graph_env is None:
                    fixed_records.append(record)
                    continue
                
                # Reconstruct planning state from L1 data
                vr = by_vid.get(video_id)
                if vr is None or idx >= len(metas):
                    fixed_records.append(record)
                    continue
                
                meta = metas[idx]
                if meta.video_id != video_id:
                    # Index mismatch, skip
                    fixed_records.append(record)
                    continue
                
                # Get compressed segments
                if video_id not in seg_cache:
                    seg_cache[video_id] = compress_segments(vr.frame_labels, vr.frame_task_labels, vocab)
                
                # Build state
                state = build_state_for_sample(
                    idx=idx,
                    vr=vr,
                    end=meta.end,
                    vocab=vocab,
                    horizon=5,  # Default horizon
                    append_terminate=True,
                    precomputed_segs=seg_cache.get(video_id)
                )
                
                if state is None:
                    fixed_records.append(record)
                    continue
                
                # Validate and fix the action
                fixed_record = validate_and_fix_action(
                    record, graph_env, state.completed_steps, immediate_M=1
                )
                
                if fixed_record != record:
                    changes += 1
                
                fixed_records.append(fixed_record)
                
            except Exception as e:
                print(f"    Error processing record {record.get('idx', '?')}: {e}")
                fixed_records.append(record)
                continue
        
        print(f"  Made {changes} changes")
        
        # Save fixed records
        if changes > 0:
            save_jsonl(action_file, fixed_records)
            print(f"  Saved updated {action_file}")
            
            # Recompute metrics if available
            recompute_metrics_if_needed(action_file.parent, fixed_records)
        else:
            print(f"  No changes needed for {action_file}")
    
    print("\nAll files processed!")


if __name__ == "__main__":
    main()