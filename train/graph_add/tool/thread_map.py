"""Eval-aligned thread labels (TaskGraphManager._build_thread_map)."""
from __future__ import annotations

from typing import Dict, Optional

_CACHE: Dict[str, Dict[str, str]] = {}


def get_thread_map(
    taxonomy_graph: Dict,
    *,
    thread_fallback: bool = True,
    cache_key: Optional[str] = None,
) -> Dict[str, str]:
    """Return action-name -> thread_id using the same rules as evaluation."""
    if cache_key and cache_key in _CACHE:
        return _CACHE[cache_key]

    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    tp_path = root / "test/onestep_planning/task_planner.py"
    spec = importlib.util.spec_from_file_location("proact_task_planner", tp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load TaskGraphManager from {tp_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TaskGraphManager = mod.TaskGraphManager

    mgr = TaskGraphManager.__new__(TaskGraphManager)
    mgr.thread_fallback = bool(thread_fallback)
    mgr.taxonomy = {str(k): v for k, v in taxonomy_graph.items()}
    mgr.id2node = mgr.taxonomy
    mgr.name2node = {
        v.get("name", ""): v for v in mgr.id2node.values() if v.get("name")
    }
    mgr.name2id = {
        v.get("name", ""): str(k) for k, v in mgr.taxonomy.items() if v.get("name")
    }
    mgr.enable_or_fallback = False
    mgr.and_gates = mgr._build_and_gates()
    mgr.or_gates = mgr._build_or_gates()
    mgr.thread_map = mgr._build_thread_map()
    if cache_key:
        _CACHE[cache_key] = mgr.thread_map
    return mgr.thread_map


def thread_id_of_name(action_name: str, thread_map: Dict[str, str]) -> str:
    if not action_name:
        return "serial_main"
    return thread_map.get(action_name, "serial_main")
