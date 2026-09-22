"""
Ego-centric subgraph extraction and rendering for ProAct 2.0.

Given a task graph (taxonomy), the current completed steps, and the current step,
extracts a local subgraph containing:
  - Done nodes (completed steps)
  - Current node
  - Forward-reachable nodes (up to max_depth=5)

Supports midlevel-phase collapsing: fully-done phases are collapsed into
single summary nodes to reduce visual/textual clutter.

Output formats:
  - Plain text (structured, for LLM text input)
  - Graphviz PNG image (for VLM image input)
"""

from __future__ import annotations

import io
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image


# ============================================================================
# Graph adjacency helpers
# ============================================================================

def build_adjacency(
    graph: Dict[str, Dict],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    children: Dict[str, List[str]] = {}
    parents: Dict[str, List[str]] = {}
    for nid, node in graph.items():
        parents[nid] = []
        pid = node.get("parent_id")
        if pid is None:
            continue
        if not isinstance(pid, list):
            pid = [pid]
        parents[nid] = [str(p) for p in pid if p is not None]
        for p in parents[nid]:
            children.setdefault(p, []).append(nid)
    return children, parents


# ============================================================================
# Map step names -> node IDs in a task graph
# ============================================================================

def map_steps_to_node_ids(
    graph: Dict[str, Dict],
    completed_steps: List[str],
    current_step: str,
) -> Tuple[Set[str], Optional[str]]:
    """Map step name strings to graph node IDs.

    Uses case-insensitive stripped matching.
    """
    name_to_nids: Dict[str, List[str]] = {}
    for nid, node in graph.items():
        name = node.get("name", "").strip().lower()
        if name:
            name_to_nids.setdefault(name, []).append(nid)

    done_nodes: Set[str] = set()
    for step in completed_steps:
        key = step.strip().lower()
        if key in name_to_nids:
            done_nodes.update(name_to_nids[key])

    current_node: Optional[str] = None
    cur_key = current_step.strip().lower() if current_step else ""
    if cur_key and cur_key in name_to_nids:
        current_node = name_to_nids[cur_key][0]

    return done_nodes, current_node


# ============================================================================
# Subgraph extraction
# ============================================================================

def extract_ego_subgraph(
    graph: Dict[str, Dict],
    children: Dict[str, List[str]],
    parents: Dict[str, List[str]],
    done_nodes: Set[str],
    current_node: Optional[str],
    max_depth: int = 5,
) -> Dict[str, str]:
    """Extract ego-centric subgraph with status labels.

    Returns dict: node_id -> status ("done", "current", "next", "future").
    """
    sub: Dict[str, str] = {}

    for n in done_nodes:
        if n in graph:
            sub[n] = "done"

    if current_node is None:
        return sub

    sub[current_node] = "current"

    queue = deque([(current_node, 0)])
    visited_forward = {current_node}
    while queue:
        n, depth = queue.popleft()
        for c in children.get(n, []):
            if c not in visited_forward and depth < max_depth:
                visited_forward.add(c)
                queue.append((c, depth + 1))
                if c not in sub:
                    sub[c] = "next" if n == current_node else "future"
    return sub


# ============================================================================
# Midlevel phase collapsing
# ============================================================================

def _find_midlevel_owner(
    nid: str, graph: Dict[str, Dict], parents: Dict[str, List[str]],
) -> Optional[str]:
    node = graph.get(nid, {})
    if node.get("is_midlevel") and node.get("midlevel_category") == "start":
        return nid
    visited = {nid}
    queue = deque([nid])
    while queue:
        cur = queue.popleft()
        for p in parents.get(cur, []):
            if p in visited:
                continue
            visited.add(p)
            pnode = graph.get(p, {})
            if pnode.get("is_midlevel") and pnode.get("midlevel_category") == "start":
                return p
            queue.append(p)
    return None


def _build_midlevel_phases(
    graph: Dict[str, Dict], parents: Dict[str, List[str]],
) -> Dict[str, Dict]:
    phases: Dict[str, Dict] = {}
    for nid, node in graph.items():
        if node.get("is_midlevel") and node.get("midlevel_category") == "start":
            name = node.get("name", "")
            for prefix in ("Start ", "start "):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            phases[nid] = {"name": name, "start": nid, "ends": [], "members": {nid}}

    for nid, node in graph.items():
        if node.get("is_midlevel") and node.get("midlevel_category") == "end":
            owner = _find_midlevel_owner(nid, graph, parents)
            if owner and owner in phases:
                phases[owner]["ends"].append(nid)
                phases[owner]["members"].add(nid)

    for nid in graph:
        if graph[nid].get("is_midlevel"):
            continue
        owner = _find_midlevel_owner(nid, graph, parents)
        if owner and owner in phases:
            phases[owner]["members"].add(nid)

    return phases


def collapse_done_phases(
    sub: Dict[str, str],
    graph: Dict[str, Dict],
    children: Dict[str, List[str]],
    parents: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict]]:
    """Collapse midlevel phases where ALL members in the subgraph are done."""
    phases = _build_midlevel_phases(graph, parents)

    collapsible: Dict[str, Dict] = {}
    for start_id, phase in phases.items():
        members_in_sub = {m for m in phase["members"] if m in sub}
        if len(members_in_sub) < 2:
            continue
        if all(sub[m] == "done" for m in members_in_sub):
            collapsible[start_id] = {
                **phase,
                "members_in_sub": members_in_sub,
                "count": len(members_in_sub),
            }

    if not collapsible:
        return sub, children, parents, {}

    member_to_summary: Dict[str, str] = {}
    collapse_info: Dict[str, Dict] = {}
    for start_id, info in collapsible.items():
        summary_id = f"_phase_{start_id}"
        collapse_info[summary_id] = info
        for m in info["members_in_sub"]:
            member_to_summary[m] = summary_id

    new_sub: Dict[str, str] = {}
    for nid, status in sub.items():
        if nid in member_to_summary:
            sid = member_to_summary[nid]
            if sid not in new_sub:
                new_sub[sid] = "done_col"
        else:
            new_sub[nid] = status

    new_children: Dict[str, List[str]] = {}
    new_parents: Dict[str, List[str]] = {}
    for nid in sub:
        real_nid = member_to_summary.get(nid, nid)
        for c in children.get(nid, []):
            if c not in sub:
                continue
            real_c = member_to_summary.get(c, c)
            if real_nid == real_c:
                continue
            new_children.setdefault(real_nid, [])
            if real_c not in new_children[real_nid]:
                new_children[real_nid].append(real_c)
            new_parents.setdefault(real_c, [])
            if real_nid not in new_parents[real_c]:
                new_parents[real_c].append(real_nid)

    for nid in new_sub:
        new_children.setdefault(nid, [])
        new_parents.setdefault(nid, [])

    return new_sub, new_children, new_parents, collapse_info


# ============================================================================
# Label helpers
# ============================================================================

def _get_label(graph: Dict[str, Dict], nid: str) -> str:
    name = graph[nid].get("name", f"Node {nid}")
    return name[:40] + "..." if len(name) > 40 else name


def _get_summary_label(collapse_info: Dict[str, Dict], sid: str) -> str:
    info = collapse_info[sid]
    return f"{info['name']} ({info['count']} steps)"


def _get_node_label(
    graph: Dict[str, Dict], collapse_info: Dict[str, Dict], nid: str,
) -> str:
    if nid.startswith("_phase_"):
        return _get_summary_label(collapse_info, nid)
    return _get_label(graph, nid)


def _get_cond(graph: Dict[str, Dict], nid: str) -> Tuple[str, str]:
    if nid.startswith("_phase_"):
        return "", ""
    node = graph[nid]
    return node.get("activation_condition", ""), node.get("condition_type", "")


def _sort_key(x: str) -> int:
    if x.startswith("_phase_"):
        return int(x.split("_")[-1])
    return int(x)


# ============================================================================
# Text subgraph rendering
# ============================================================================

def render_text_subgraph(
    graph: Dict[str, Dict],
    new_children: Dict[str, List[str]],
    new_parents: Dict[str, List[str]],
    new_sub: Dict[str, str],
    collapse_info: Dict[str, Dict],
    task_name: str,
) -> str:
    """Render a subgraph as structured plain text for LLM input."""
    lines = ["<|subgraph|>", f"Task: {task_name}"]

    for nid in sorted(new_sub, key=_sort_key):
        status = new_sub[nid]
        label = _get_node_label(graph, collapse_info, nid)
        if status == "done_col":
            status_str = "done"
            label = f"[collapsed] {label}"
        else:
            status_str = status
        _, ctype = _get_cond(graph, nid)
        pids = [p for p in new_parents.get(nid, []) if p in new_sub]
        dep = ""
        if pids:
            plabels = [_get_node_label(graph, collapse_info, p) for p in pids]
            gate = f" [{ctype}]" if ctype and len(pids) > 1 else ""
            dep = f" (after{gate}: {', '.join(plabels)})"
        lines.append(f"  [{status_str}] {label}{dep}")

    lines.append("")
    lines.append("Edges:")
    for nid in sorted(new_sub, key=_sort_key):
        for c in new_children.get(nid, []):
            if c in new_sub:
                sl = _get_node_label(graph, collapse_info, nid)
                dl = _get_node_label(graph, collapse_info, c)
                _, ct = _get_cond(graph, c)
                pids = [p for p in new_parents.get(c, []) if p in new_sub]
                gate = f" [{ct}]" if ct and len(pids) > 1 else ""
                lines.append(f"  {sl} --{gate}--> {dl}")
    lines.append("<|/subgraph|>")
    return "\n".join(lines)


# ============================================================================
# Image subgraph rendering
# ============================================================================

STATUS_COLORS = {
    "done":     {"fill": "#90EE90", "stroke": "#2E7D32", "font": "#000000"},
    "done_col": {"fill": "#4CAF50", "stroke": "#1B5E20", "font": "#FFFFFF"},
    "current":  {"fill": "#FFD700", "stroke": "#FF8C00", "font": "#000000"},
    "next":     {"fill": "#FF6347", "stroke": "#CC0000", "font": "#FFFFFF"},
    "future":   {"fill": "#D3D3D3", "stroke": "#808080", "font": "#000000"},
}


def render_image_subgraph(
    graph: Dict[str, Dict],
    new_children: Dict[str, List[str]],
    new_parents: Dict[str, List[str]],
    new_sub: Dict[str, str],
    collapse_info: Dict[str, Dict],
    task_name: str,
) -> Image.Image:
    """Render a subgraph as a PIL Image using Graphviz."""
    import graphviz

    dot = graphviz.Digraph(format="png", engine="dot")
    collapsed_count = sum(1 for s in new_sub.values() if s == "done_col")
    total_nodes = len(graph)
    dot.attr(
        rankdir="TB", bgcolor="white", fontname="Helvetica",
        label=(
            f"  {task_name}\n"
            f"  {len(new_sub)} visible / {total_nodes} total"
            f" | {collapsed_count} collapsed  "
        ),
        labelloc="t", fontsize="14", pad="0.5", dpi="150",
    )
    dot.attr("node", shape="box", style="filled,rounded",
             fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="8")

    for nid in sorted(new_sub, key=_sort_key):
        status = new_sub[nid]
        label = _get_node_label(graph, collapse_info, nid)
        c = STATUS_COLORS[status]
        _, ctype = _get_cond(graph, nid)
        pids = [p for p in new_parents.get(nid, []) if p in new_sub]

        if status == "done_col":
            nl = f"[DONE]\n{label}"
            shape = "box3d"
        else:
            nl = f"[{status.upper()}]\n{label}"
            shape = "box"
            if len(pids) > 1 and ctype:
                nl += f"\n({ctype} gate)"

        pw = "3.0" if status == "current" else "1.5"
        dot.node(
            f"N{nid}", label=nl, fillcolor=c["fill"], color=c["stroke"],
            fontcolor=c["font"], penwidth=pw, shape=shape,
        )

    for nid in sorted(new_sub, key=_sort_key):
        for ch in new_children.get(nid, []):
            if ch in new_sub:
                _, ct = _get_cond(graph, ch)
                pids = [p for p in new_parents.get(ch, []) if p in new_sub]
                ea: Dict[str, str] = {}
                if len(pids) > 1 and ct:
                    ea = {
                        "label": f" {ct} ",
                        "fontcolor": "#CC0000",
                        "style": "bold",
                    }
                dot.edge(f"N{nid}", f"N{ch}", **ea)

    png_data = dot.pipe(format="png")
    return Image.open(io.BytesIO(png_data)).convert("RGB")


# ============================================================================
# High-level API: extract + collapse + render
# ============================================================================

def build_subgraph_for_step(
    taxonomy: Dict[str, Dict[str, Dict]],
    task_name: str,
    completed_steps: List[str],
    current_step: str,
    mode: str = "text",
    max_depth: int = 5,
) -> Optional[Any]:
    """Build a subgraph representation for the given step.

    Args:
        taxonomy: Full taxonomy dict (task_name -> {node_id_str -> node_info}).
        task_name: Current task name.
        completed_steps: List of completed step name strings.
        current_step: Current step name string.
        mode: "text" or "image".
        max_depth: Max forward depth from current node.

    Returns:
        str (for text mode) or PIL.Image (for image mode), or None if
        the task is not in the taxonomy or no current node can be found.
    """
    if task_name not in taxonomy:
        return None

    raw_graph = taxonomy[task_name]
    graph = {str(k): v for k, v in raw_graph.items()}
    children, parents = build_adjacency(graph)

    done_nodes, current_node = map_steps_to_node_ids(
        graph, completed_steps, current_step,
    )

    if current_node is None and not done_nodes:
        return None

    sub = extract_ego_subgraph(
        graph, children, parents, done_nodes, current_node,
        max_depth=max_depth,
    )

    if not sub:
        return None

    new_sub, new_ch, new_pa, collapse_info = collapse_done_phases(
        sub, graph, children, parents,
    )

    if mode == "text":
        return render_text_subgraph(
            graph, new_ch, new_pa, new_sub, collapse_info, task_name,
        )
    elif mode == "image":
        return render_image_subgraph(
            graph, new_ch, new_pa, new_sub, collapse_info, task_name,
        )
    else:
        raise ValueError(f"Unknown subgraph mode: {mode}")
