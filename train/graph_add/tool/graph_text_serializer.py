"""Text serializers for task-graph subgraphs (no gold / no answer leak).

Used when ``subgraph_mode=text``. Formats align with GRAPH_TEXT_FORMAT_PROBE:
``compact_yaml`` and ``structured_json`` include ``frontier_names`` and
``legal_cross_thread_actions`` derived from the same legal pool as training.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

from train.graph_add.tool.subgraph import _get_node_label


def _frontier_names(
    new_sub: Dict[str, str],
    graph: Dict[str, Dict],
    collapse_info: Dict[str, Dict],
    *,
    status: str = "frontier",
) -> List[str]:
    out: List[str] = []
    for nid, st in new_sub.items():
        if st != status or str(nid).startswith("_"):
            continue
        name = _get_node_label(graph, collapse_info, nid)
        if name:
            out.append(name)
    return sorted(set(out))


def _thread_for(
    graph: Dict[str, Dict],
    collapse_info: Dict[str, Dict],
    nid: str,
    thread_map: Optional[Dict[str, str]],
) -> str:
    name = _get_node_label(graph, collapse_info, nid)
    if thread_map and name in thread_map:
        return str(thread_map[name])
    return "serial_main"


def _cross_legal_actions(
    *,
    task_name: str,
    completed_steps: List[str],
    current_step: str,
    future_steps: Optional[List[str]],
    annotation_path: Optional[str],
    tg_cache: Optional[Dict[str, Any]],
) -> List[str]:
    if not annotation_path:
        return []
    try:
        from train.graph_add.tool.apa_next_action import (
            _get_mgr,
            _is_cross_thread,
            _legal_now,
        )
    except ImportError:
        return []

    mgr = _get_mgr(task_name, annotation_path, tg_cache or {})
    if mgr is None:
        return []
    human_now = (current_step or "").strip()
    return sorted(
        a
        for a in _legal_now(mgr, list(completed_steps or []), list(future_steps or []))
        if _is_cross_thread(mgr, a, human_now)
    )


def _status_tag(status: str) -> str:
    """Bracket tag for compact_yaml node names (matches probe mermaid/dot)."""
    st = "done" if status == "done_col" else str(status or "")
    if st in ("current", "frontier", "done", "future", "wrapper"):
        return f" [{st}]"
    return ""


def serialize_compact_yaml(
    *,
    task_name: str,
    current_step: str,
    graph: Dict[str, Dict],
    new_sub: Dict[str, str],
    new_ch: Dict[str, List[str]],
    collapse_info: Dict[str, Dict],
    thread_map: Optional[Dict[str, str]],
    frontier_names: List[str],
    legal_cross_thread_actions: List[str],
    omit_legal_pool: bool = False,
) -> str:
    lines = [
        "<|subgraph|>",
        f"task: {task_name}",
        f"human_now: {current_step}",
        "nodes:",
    ]
    for nid in sorted(new_sub, key=lambda x: (new_sub[x], x)):
        name = _get_node_label(graph, collapse_info, nid)
        status = new_sub[nid]
        if status == "done_col":
            status = "done"
        thread = _thread_for(graph, collapse_info, nid, thread_map)
        tagged_name = f"{name}{_status_tag(status)}"
        lines.append(f"  - id: {nid}")
        lines.append(f"    name: {tagged_name}")
        lines.append(f"    status: {status}")
        lines.append(f"    thread: {thread}")
    lines.append("edges:")
    for nid in new_sub:
        for c in new_ch.get(nid, []):
            if c in new_sub:
                lines.append(f"  - [{nid}, {c}]")
    if not omit_legal_pool:
        lines.append(f"frontier_names: {json.dumps(frontier_names, ensure_ascii=False)}")
        lines.append(
            f"legal_cross_thread_actions: {json.dumps(legal_cross_thread_actions, ensure_ascii=False)}"
        )
    lines.append("<|/subgraph|>")
    return "\n".join(lines)


def serialize_structured_json(
    *,
    task_name: str,
    current_step: str,
    completed_steps: List[str],
    graph: Dict[str, Dict],
    new_sub: Dict[str, str],
    new_ch: Dict[str, List[str]],
    collapse_info: Dict[str, Dict],
    thread_map: Optional[Dict[str, str]],
    frontier_names: List[str],
    legal_cross_thread_actions: List[str],
) -> str:
    frontier_set: Set[str] = set(frontier_names)
    nodes = []
    for nid, status in new_sub.items():
        name = _get_node_label(graph, collapse_info, nid)
        st = "done" if status == "done_col" else status
        nodes.append({
            "id": nid,
            "name": name,
            "status": st,
            "thread": _thread_for(graph, collapse_info, nid, thread_map),
            "is_frontier": name in frontier_set,
        })
    edges = []
    for nid in new_sub:
        for c in new_ch.get(nid, []):
            if c in new_sub:
                edges.append({"from": nid, "to": c})
    payload = {
        "task": task_name,
        "human_current_step": current_step,
        "completed_steps": list(completed_steps or []),
        "nodes": nodes,
        "edges": edges,
        "frontier_names": frontier_names,
        "legal_cross_thread_actions": legal_cross_thread_actions,
    }
    return "<|subgraph|>\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n<|/subgraph|>"


_TEXT_FORMATS = frozenset({"legacy", "compact_yaml", "structured_json"})


def render_subgraph_text(
    *,
    text_format: str,
    task_name: str,
    current_step: str,
    completed_steps: List[str],
    graph: Dict[str, Dict],
    new_sub: Dict[str, str],
    new_ch: Dict[str, List[str]],
    new_pa: Dict[str, List[str]],
    collapse_info: Dict[str, Dict],
    thread_map: Optional[Dict[str, str]] = None,
    future_steps: Optional[List[str]] = None,
    annotation_path: Optional[str] = None,
    tg_cache: Optional[Dict[str, Any]] = None,
    omit_legal_pool: bool = False,
) -> str:
    """Serialize ego subgraph to plain text for ``subgraph_mode=text``."""
    from train.graph_add.tool.subgraph import render_text_subgraph

    fmt = str(text_format or "legacy").strip().lower()
    if fmt == "json":
        fmt = "structured_json"
    if fmt not in _TEXT_FORMATS:
        fmt = "legacy"


    _env_omit = os.environ.get("GRAPH_TEXT_OMIT_LEGAL_POOL", "").strip().lower()
    omit = bool(omit_legal_pool) or _env_omit in ("1", "true", "yes", "on")

    frontier_names = _frontier_names(new_sub, graph, collapse_info, status="frontier")
    cross_legal = _cross_legal_actions(
        task_name=task_name,
        completed_steps=list(completed_steps or []),
        current_step=current_step,
        future_steps=future_steps,
        annotation_path=annotation_path,
        tg_cache=tg_cache,
    )
    if str(os.environ.get("OMIT_LEGAL_POOL_FOR_QA", "")).strip().lower() in {"1", "true", "yes"}:
        cross_legal = []

    if fmt == "compact_yaml":
        return serialize_compact_yaml(
            task_name=task_name,
            current_step=current_step,
            graph=graph,
            new_sub=new_sub,
            new_ch=new_ch,
            collapse_info=collapse_info,
            thread_map=thread_map,
            frontier_names=frontier_names,
            legal_cross_thread_actions=cross_legal,
            omit_legal_pool=omit,
        )
    if fmt == "structured_json":
        return serialize_structured_json(
            task_name=task_name,
            current_step=current_step,
            completed_steps=list(completed_steps or []),
            graph=graph,
            new_sub=new_sub,
            new_ch=new_ch,
            collapse_info=collapse_info,
            thread_map=thread_map,
            frontier_names=frontier_names,
            legal_cross_thread_actions=cross_legal,
        )

    return render_text_subgraph(
        graph, new_ch, new_pa, new_sub, collapse_info, task_name,
    )
