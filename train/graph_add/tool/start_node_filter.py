"""Exclude synthetic task-root start nodes from robot legal-action pools."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def is_task_root_start_node(node_id: str, node: Dict[str, Any]) -> bool:
    """Return True for taxonomy node 0 entries like ``Start {task name}``.

    These are structural task-entry anchors (never appear in video segment
    labels). Concrete annotated steps such as ``Start the washing machine``
    (node 4) or ``Start cooking`` (node 10) are *not* node 0 and stay legal.
    """
    if str(node_id) != "0":
        return False
    name = str(node.get("name") or "").strip()
    if not name:
        return False
    return name.startswith("Start ") or name.startswith("start ")


def filter_task_root_starts_from_names(mgr: Any, action_names: Iterable[str]) -> List[str]:
    """Drop task-root start tokens from a list of step names."""
    name2id = getattr(mgr, "name2id", None) or {}
    id2node = getattr(mgr, "id2node", None) or {}
    out: List[str] = []
    for action in action_names:
        s = str(action or "").strip()
        if not s:
            continue
        nid = name2id.get(s)
        if nid is None:
            out.append(s)
            continue
        node = id2node.get(str(nid), {})
        if is_task_root_start_node(str(nid), node):
            continue
        out.append(s)
    return out


def filter_task_root_starts_set(mgr: Any, action_names: Set[str]) -> Set[str]:
    return set(filter_task_root_starts_from_names(mgr, action_names))
