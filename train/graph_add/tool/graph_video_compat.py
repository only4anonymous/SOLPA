"""Detect when the *current* video step violates task-graph prerequisites."""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .subgraph import build_adjacency


def _is_real_action(node: dict) -> bool:
    name = str(node.get("name", "") or "").strip()
    if not name or name.lower() == "terminate":
        return False
    if node.get("is_midlevel"):
        return False
    ml = str(node.get("midlevel_type") or "").lower()
    if ml in ("parallel", "subtask"):
        return False
    if str(node.get("midlevel_category") or "").lower() in ("start", "end"):
        return False
    return True


def _direct_real_prereq_names(graph: Dict, parents_map: Dict[str, List[str]], nid: str) -> List[str]:
    out: List[str] = []
    for pid in parents_map.get(str(nid), []):
        pnode = graph.get(str(pid), {})
        if _is_real_action(pnode):
            pname = str(pnode.get("name", "") or "").strip()
            if pname:
                out.append(pname)
    return out


def _completed_set(completed_steps: Sequence[str] | None) -> Set[str]:
    return {str(s or "").strip() for s in (completed_steps or []) if str(s or "").strip()}


def check_video_graph_alignment(
    taxonomy_graph: Dict,
    completed_steps: Sequence[str] | None,
    current_step: str | None,
) -> Tuple[bool, str]:
    """
    Graph is unusable when the *current* human step is not graph-feasible:
    a direct real-action prerequisite is neither completed nor current.

    This matches the common E2E failure mode (execute before graph allows) without
    falsely flagging unordered ``completed_steps`` lists or midlevel-only gaps.
    """
    cur = str(current_step or "").strip()
    if not cur:
        return True, "ok_no_current"

    graph = {str(k): v for k, v in taxonomy_graph.items()}
    _, parents = build_adjacency(graph)
    name2id = {
        str(v.get("name", "") or "").strip(): str(k)
        for k, v in graph.items()
        if v.get("name")
    }
    done = _completed_set(completed_steps)
    nid = name2id.get(cur)
    if not nid:
        return True, "ok_unknown_current"

    for pname in _direct_real_prereq_names(graph, parents, nid):
        if pname not in done and pname != cur:
            return False, f"current_missing_prereq:{cur}_needs:{pname}"
    return True, "ok"


def is_graph_usable_for_state(
    taxonomy: Dict,
    task_name: str,
    completed_steps: Sequence[str] | None,
    current_step: str | None,
) -> Tuple[bool, str]:
    if task_name not in taxonomy:
        return False, "unknown_task"
    g = taxonomy[task_name]
    if not isinstance(g, dict):
        return False, "bad_taxonomy"
    return check_video_graph_alignment(g, completed_steps, current_step)
