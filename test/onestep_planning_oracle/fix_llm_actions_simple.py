#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to fix LLM action legality in llm_planning_actions_new.jsonl files.
This version doesn't try to recreate the full environment state, just applies basic fixes.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[2]
ONESTEP_DIR = REPO_ROOT / "test" / "onestep_planning"
if str(ONESTEP_DIR) not in sys.path:
    sys.path.insert(0, str(ONESTEP_DIR))

from task_planner import TaskGraphManager  # noqa: E402


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


def fix_action_simple(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply basic fixes to a single LLM action record.
    This is a simplified version that just standardizes wait actions.
    """
    # Get the raw action
    raw_action = str(record.get("robot_action_raw", "") or "").strip()
    if not raw_action:
        raw_action = str(record.get("action", "") or "").strip()
    
    # Standardize Wait actions
    action = raw_action
    if action.strip().lower() in {"wait", "none", "wait/none", "wait / none"}:
        action = "Wait / None"
    
    # Update record
    updated_record = dict(record)
    updated_record["robot_action_effective"] = action
    
    # If we standardized the action, note it
    if action != raw_action:
        print(f"  Standardized: {raw_action} -> {action}")
        updated_record["robot_forced_wait_reason"] = "standardized"
    
    return updated_record


def main():
    """Main function to fix all llm_planning_actions_new.jsonl files."""
    
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
                fixed_record = fix_action_simple(record)
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
        else:
            print(f"  No changes needed for {action_file}")
    
    print("\nSimple fixes applied!")
    print("For full legal action validation, please run the one-step evaluation script.")


if __name__ == "__main__":
    main()