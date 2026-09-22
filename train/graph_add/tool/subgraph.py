"""
Ego-centric subgraph extraction and rendering for ProAct 2.0.

Given a task graph (taxonomy), completed steps, and the current step,
build a local subgraph with:
  - Done nodes (completed steps)
  - Current node (Phase-1 predicted step)
  - Forward-reachable nodes from current (up to max_depth hops)
  - Midlevel-phase collapsing for fully-done phases

Rendering:
  - Image: Graphviz PNG, OCR-friendly (all info as text in nodes)
  - Text: structured dependency list with [done]/[current]/[future]

Design for VLM OCR:
  - [done] prefix on done nodes; [current]/[future] on others
  - AND/OR activation as edge labels
  - DPI (96), no downscaling — let VLM processor resize
"""

from __future__ import annotations

import io
import re
import textwrap
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageChops

# Hard upper bound on the rendered subgraph pixel count. Qwen2.5-VL consumes
# one visual token per 28x28 pixels, so this keeps each task graph at <=512
# visual tokens. The renderer preserves OCR readability by wrapping long node
# labels and compressing only whitespace before falling back to any resize.
PIXELS_PER_VISION_TOKEN = 28 * 28
MAX_RENDER_TOKENS = 1536
MAX_RENDER_PIXELS = MAX_RENDER_TOKENS * PIXELS_PER_VISION_TOKEN

# Subgraph horizon: DEFAULT_FORWARD_DEPTH (=5) aligns with E2E predict_steps.
# Readability is enforced by MAX_RENDER_TOKENS=512 at render time only.
DEFAULT_FORWARD_DEPTH = 5


# ---------------------------------------------------------------------------
# Graph adjacency
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Map step names -> node IDs
# ---------------------------------------------------------------------------

def map_steps_to_node_ids(
    graph: Dict[str, Dict],
    completed_steps: List[str],
    current_step: str,
) -> Tuple[Set[str], Optional[str]]:
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


# ---------------------------------------------------------------------------
# Subgraph extraction (with current node, depth=5)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wrapper / hub detection
# ---------------------------------------------------------------------------
#
# Many task graphs are inflated by structural "wrapper" nodes (Get
# Ingredients / Put away kitchenware / Start <phase> / Finish <phase>).
# These nodes do not represent observable actions: they are container
# markers whose purpose is to group leaf operations. They contribute
# nothing to next-step prediction but explode forward fan-out (one such
# hub can have 30+ children).
#
# We treat as a wrapper:
#   * any node with ``is_midlevel=True`` (taxonomy-explicit), OR
#   * any node whose direct child count >= ``WRAPPER_FANOUT_THRESHOLD``
#     (catches the few fanout=11 outliers that taxonomy missed -- see
#     diagnostics on the 75-recipe taxonomy: every fanout>=10 node is a
#     midlevel hub; the threshold of 8 is a defensive lower bound).
WRAPPER_FANOUT_THRESHOLD = 8


def _is_wrapper_node(
    nid: str,
    graph: Dict[str, Dict],
    children: Dict[str, List[str]],
    parents: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Decide whether ``nid`` is a structural container vs a real action.

    The taxonomy is noisy: some single-child concrete steps (e.g.
    'Dispose waste', 'Read the instructions') carry
    ``is_midlevel=True`` / ``midlevel_category='start'`` despite being
    actionable. Filtering those as wrappers removes legitimate QA
    candidates. We therefore require *structural* evidence of
    container-hood:

      * ``midlevel_category='start'`` AND fan-out >= 2  -- a phase
        header that fans out to multiple sub-steps.
      * ``midlevel_category='end'``   AND fan-in  >= 2  -- a phase
        footer that merges multiple sub-steps.
      * High direct fan-out (>= ``WRAPPER_FANOUT_THRESHOLD``) -- a
        hub regardless of taxonomy tags.

    A pure ``is_midlevel=True`` flag is no longer enough on its own.
    """
    node = graph.get(nid, {})
    fanout = len(children.get(nid, []))
    fanin = len(parents.get(nid, [])) if parents is not None else 0
    cat = node.get("midlevel_category")
    name = str(node.get("name", "") or "").strip().lower()
    if cat == "start" and (fanout >= 2 or name.startswith("start ")):
        return True
    if cat == "end":
        return True
    if fanout >= WRAPPER_FANOUT_THRESHOLD:
        return True
    return False


def _compute_wrapper_set(
    graph: Dict[str, Dict],
    children: Dict[str, List[str]],
    parents: Optional[Dict[str, List[str]]] = None,
) -> Set[str]:
    return {nid for nid in graph
            if _is_wrapper_node(nid, graph, children, parents)}


def _effective_children(
    graph: Dict[str, Dict],
    children: Dict[str, List[str]],
    wrapper_set: Set[str],
) -> Dict[str, List[str]]:
    """For every node, return its non-wrapper descendants reachable
    without going through any non-wrapper node first.

    Wrappers act as transparent connectors: edges through them are
    rewired so a non-wrapper node directly points to the next
    non-wrapper node(s) reachable through any chain of wrappers.
    """
    eff: Dict[str, List[str]] = {}
    for nid in graph:
        out: List[str] = []
        seen_wrap: Set[str] = set()
        seen_out: Set[str] = set()
        stack = list(children.get(nid, []))
        while stack:
            c = stack.pop()
            if c in wrapper_set:
                if c in seen_wrap:
                    continue
                seen_wrap.add(c)
                stack.extend(children.get(c, []))
            else:
                if c in seen_out:
                    continue
                seen_out.add(c)
                out.append(c)
        eff[nid] = out
    return eff


def _effective_parents(
    graph: Dict[str, Dict],
    parents: Dict[str, List[str]],
    wrapper_set: Set[str],
) -> Dict[str, List[str]]:
    """Mirror of :func:`_effective_children`: each node's gating
    predecessors with wrappers transparently bypassed."""
    eff: Dict[str, List[str]] = {}
    for nid in graph:
        out: List[str] = []
        seen_wrap: Set[str] = set()
        seen_out: Set[str] = set()
        stack = list(parents.get(nid, []))
        while stack:
            p = stack.pop()
            if p in wrapper_set:
                if p in seen_wrap:
                    continue
                seen_wrap.add(p)
                stack.extend(parents.get(p, []))
            else:
                if p in seen_out:
                    continue
                seen_out.add(p)
                out.append(p)
        eff[nid] = out
    return eff



# ---------------------------------------------------------------------------
# task_planner-aligned legality check
# ---------------------------------------------------------------------------
#
# These two helpers mirror the semantics of
# ``test/onestep_planning/task_planner.TaskGraphManager``:
# * ``check_condition``             -- evaluates ``((1) AND (2)) OR (3)``
#   condition strings, applies ancestor closure on the completed set,
#   and recurses through ``is_midlevel`` parents so phase wrappers
#   transparently propagate completion.
# * ``get_legal_robot_actions``     -- excludes ``midlevel_type=='parallel'``
#   and ``is_midlevel=True`` nodes, excludes nodes whose name is in the
#   done set (raw -- no closure exclusion!), and accepts whatever passes
#   ``check_condition``.
#
# We replicate both here so that:
#   * the QA helpers can call ``is_legal_action`` instead of the
#     simpler in-scope AND/OR check, and
#   * ``extract_ego_subgraph`` can annotate which visible future nodes
#     are LEGAL-NOW (status="frontier"); those that are still blocked
#     stay status="future".
#
# A self-test against ``TaskGraphManager.get_legal_robot_actions`` on
# 200 train + 200 test states is in
# ``scripts/proact_legality_self_test.py`` and must report 0 mismatch.

def _ancestors_of(graph: Dict[str, Dict], parents: Dict[str, List[str]],
                   nid: str) -> Set[str]:
    """Return all transitive parent ids of ``nid`` (excluding ``nid``)."""
    out: Set[str] = set()
    stack = [nid]
    while stack:
        n = stack.pop()
        for p in parents.get(n, []):
            if p in out or p not in graph:
                continue
            out.add(p)
            stack.append(p)
    return out


def check_condition(
    graph: Dict[str, Dict],
    parents: Dict[str, List[str]],
    condition_str: Any,
    completed_step_names: List[str],
) -> bool:
    """Mirror of ``TaskGraphManager.check_condition`` (id-based).

    * Builds completed-id closure from ``completed_step_names`` once.
    * Recurses into ``is_midlevel`` parents (with cycle guard) so a
      phase wrapper is ``True`` whenever its own
      ``activation_condition`` is satisfied -- this matches the
      planner's "中间版语义" exactly.
    """
    cond = condition_str
    if cond is None:
        return True
    if not isinstance(cond, str):
        cond = str(cond)
    if not cond or cond.strip().upper() == "TRUE":
        return True

    # Match TaskGraphManager exactly: when two nodes share the same
    # name (taxonomy duplicates exist for ~5% of recipes), the LATER
    # entry overwrites the earlier one (the planner uses a plain dict
    # comprehension which has the same semantics).
    name2id: Dict[str, str] = {}
    for nid, node in graph.items():
        nm = node.get("name", "")
        if nm:
            name2id[nm] = str(nid)

    completed_ids: Set[str] = set()
    for nm in completed_step_names:
        if nm in name2id:
            completed_ids.add(name2id[nm])
    if completed_ids:
        closure: Set[str] = set(completed_ids)
        for cid in list(completed_ids):
            closure |= _ancestors_of(graph, parents, cid)
        completed_ids = closure

    visiting: Set[str] = set()

    def _eval_id(nid: str) -> bool:
        nid = str(nid)
        if nid in completed_ids:
            return True
        node = graph.get(nid)
        if node is None:
            return False
        ml = node.get("is_midlevel", False)
        if isinstance(ml, str):
            ml_bool = ml.strip().lower() in {"true", "1", "yes", "y", "t"}
        else:
            ml_bool = bool(ml)
        if ml_bool:
            if nid in visiting:
                return False
            visiting.add(nid)
            try:
                inner = node.get("activation_condition", "TRUE")
                return _eval_condition(str(inner))
            finally:
                visiting.discard(nid)
        return False

    def _eval_condition(cond_inner: str) -> bool:
        if not cond_inner or cond_inner.strip().upper() == "TRUE":
            return True
        eval_str = re.sub(
            r"\((\d+)\)",
            lambda m: "True" if _eval_id(m.group(1)) else "False",
            cond_inner,
        )
        eval_str = re.sub(r"\bAND\b", "and", eval_str, flags=re.IGNORECASE)
        eval_str = re.sub(r"\bOR\b", "or", eval_str, flags=re.IGNORECASE)
        try:
            return bool(eval(eval_str))
        except Exception:
            return False

    return _eval_condition(cond)


def is_legal_action(
    nid: str,
    graph: Dict[str, Dict],
    parents: Dict[str, List[str]],
    completed_step_names: List[str],
) -> bool:
    """Mirror of ``TaskGraphManager.get_legal_robot_actions`` semantics
    on a single node. Returns True iff:

    * ``nid`` is a non-midlevel, non-parallel node with a name,
    * ``nid`` is not a synthetic task-root start (node 0 ``Start …``),
    * ``nid``'s name is NOT in ``completed_step_names`` (raw, no
      closure exclusion -- matches the planner exactly), and
    * ``check_condition(activation_condition, completed_step_names)``
      is satisfied.

    Note: ``terminate``-named nodes are not filtered here because
    ``TaskGraphManager.get_legal_robot_actions`` itself does not filter
    them; keeping our semantics 1:1 with the planner avoids subtle
    train/eval drift.
    """
    from train.graph_add.tool.start_node_filter import is_task_root_start_node

    node = graph.get(str(nid))
    if node is None:
        return False
    if is_task_root_start_node(str(nid), node):
        return False
    name = node.get("name", "")
    if not name:
        return False
    ml = node.get("is_midlevel", False)
    if isinstance(ml, str):
        ml_bool = ml.strip().lower() in {"true", "1", "yes", "y", "t"}
    else:
        ml_bool = bool(ml)
    if ml_bool:
        return False
    if str(node.get("midlevel_type") or "").lower() == "parallel":
        return False
    completed_set = set(completed_step_names)
    if name in completed_set:
        return False
    cond = node.get("activation_condition", "TRUE")
    return check_condition(graph, parents, cond, completed_step_names)


def annotate_frontier(
    sub: Dict[str, str],
    graph: Dict[str, Dict],
    parents: Dict[str, List[str]],
    completed_step_names: List[str],
    *,
    action_depths: Optional[Dict[str, int]] = None,
    frontier_depths: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """Promote all visible legal-now action nodes to ``frontier``.

    The subgraph already contains only wave-1 legal-now actions; this step
    applies the blue [frontier] label (W2g) before optional cross-thread
    filtering (W2j).  ``action_depths`` / ``frontier_depths`` are ignored
    (kept for call-site compatibility).
    """
    del action_depths, frontier_depths
    for nid, status in list(sub.items()):
        if status != "future":
            continue
        if str(nid).startswith("_phase_") or str(nid).startswith("_dp_"):
            continue
        if is_legal_action(nid, graph, parents, completed_step_names):
            sub[nid] = "frontier"
    return sub


_RENDER_MODES = frozenset({"default", "cross_thread_frontier", "legal_only"})

# Subtle thread tint palette (P1) — keyed by thread_map thread_id strings.
_THREAD_TINT_PALETTE: Dict[str, str] = {
    "serial_main": "#E8E8FF",
    "parallel_0": "#FFE8E8",
    "parallel_1": "#E8FFE8",
    "parallel_2": "#FFF8E8",
    "parallel_3": "#E8FFFF",
    "_default": "#F0F0F0",
}


def _blend_hex(fill: str, tint: str, alpha: float = 0.35) -> str:
    """Blend two #RRGGBB colors; ``alpha`` is tint weight."""
    def _parse(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r1, g1, b1 = _parse(fill)
    r2, g2, b2 = _parse(tint)
    a = max(0.0, min(1.0, float(alpha)))
    r = int(r1 * (1 - a) + r2 * a)
    g = int(g1 * (1 - a) + g2 * a)
    b = int(b1 * (1 - a) + b2 * a)
    return f"#{r:02x}{g:02x}{b:02x}"


def _node_name(graph: Dict[str, Dict], nid: str) -> str:
    return str(graph.get(str(nid), {}).get("name", "") or "").strip()


def apply_cross_thread_frontier_filter(
    sub: Dict[str, str],
    graph: Dict[str, Dict],
    *,
    task_name: str,
    completed_steps: List[str],
    current_step: str,
    future_steps: Optional[List[str]] = None,
    annotation_path: Optional[str] = None,
    tg_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
    """Keep frontier highlight only on cross-thread legal actions (W2j).

    Visible ``frontier`` nodes are already legal-now from ``annotate_frontier``
    (``is_legal_action`` on ``completed + current``).  Apply thread filter on
    those nodes: cross-thread → keep ``frontier``, same-thread → ``future``.
    Optionally promote ``future`` nodes in mgr ``_legal_now`` that are
    cross-thread but missing from the visible frontier set.
    Returns ``(sub, thread_map)`` for optional thread tinting.
    """
    from train.graph_add.tool.apa_next_action import (
        _get_mgr,
        _is_cross_thread,
        _legal_now,
    )

    human_now = (current_step or "").strip()
    if not human_now or not annotation_path:
        return sub, None

    mgr = _get_mgr(task_name, annotation_path, tg_cache or {})
    if mgr is None:
        return sub, None

    cross_legal = {
        a
        for a in _legal_now(mgr, list(completed_steps or []), list(future_steps or []))
        if _is_cross_thread(mgr, a, human_now)
    }

    out = dict(sub)
    for nid, status in list(sub.items()):
        if str(nid).startswith("_phase_") or str(nid).startswith("_dp_"):
            continue
        name = _node_name(graph, nid)
        if not name:
            continue
        if status == "frontier":
            out[nid] = "frontier" if _is_cross_thread(mgr, name, human_now) else "future"
        elif status == "future" and name in cross_legal:
            out[nid] = "frontier"

    thread_map = getattr(mgr, "thread_map", None) or {}
    return out, thread_map


def extract_ego_subgraph(
    graph: Dict[str, Dict],
    children: Dict[str, List[str]],
    parents: Dict[str, List[str]],
    done_nodes: Set[str],
    current_node: Optional[str],
    max_depth: int = DEFAULT_FORWARD_DEPTH,
    max_back_depth: int = 1,
    include_all_done: bool = False,
    mark_wrappers: bool = True,
    annotate_legal: bool = True,
    completed_step_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build ego subgraph with wave-1 (depth-1) legal-now actions for next-step selection.

    Action nodes are limited to real steps at action-depth==1 from ``current_node``
    that are legal immediately after ``completed_step_names`` plus current
    (:func:`is_legal_action`).  Done / wrapper nodes are kept only as structural
    bridges.  ``max_depth`` bounds the depth BFS helper only; depth-2..5 futures
    are not included.
    """
    sub: Dict[str, str] = {}
    wrapper_set = _compute_wrapper_set(graph, children, parents) if mark_wrappers else set()
    completed_step_names = list(completed_step_names or [])

    def _real_action(nid: str) -> bool:
        node = graph.get(str(nid), {})
        name = str(node.get("name", "") or "").strip()
        if not name:
            return False
        ml = node.get("is_midlevel", False)
        ml_bool = ml.strip().lower() in {"true", "1", "yes", "y", "t"} if isinstance(ml, str) else bool(ml)
        if ml_bool:
            return False
        return str(node.get("midlevel_type") or "").lower() != "parallel"

    def _name(nid: Optional[str]) -> str:
        if nid is None:
            return ""
        return str(graph.get(str(nid), {}).get("name", "") or "").strip()

    def _action_forward_depths() -> Dict[str, int]:
        """Action-hop distance from ``current_node`` (wrappers transparent)."""
        if current_node is None:
            return {}
        best: Dict[str, int] = {current_node: 0}
        q = deque([(current_node, 0)])
        while q:
            n, depth = q.popleft()
            for c in children.get(n, []):
                if c not in graph:
                    continue
                if c in wrapper_set or not _real_action(c):
                    nd = depth
                else:
                    nd = depth + 1
                if nd > max_depth:
                    continue
                old = best.get(c)
                if old is not None and old <= nd:
                    continue
                best[c] = nd
                q.append((c, nd))
        return best

    if include_all_done:
        for n in done_nodes:
            if n in graph and n not in wrapper_set:
                sub[n] = "done"
    if current_node is None:
        return sub
    sub[current_node] = "current"

    action_depths = _action_forward_depths()
    frontier_completed = list(completed_step_names)
    cur_name = _name(current_node)
    if cur_name and cur_name not in frontier_completed:
        frontier_completed.append(cur_name)

    # Wave-1: depth-1 legal-now actions only (no depth-2..5 horizon).
    legal_now_nodes: Set[str] = {
        nid for nid in action_depths
        if nid != current_node
        and _real_action(nid)
        and action_depths.get(nid) == 1
        and is_legal_action(nid, graph, parents, frontier_completed)
    }

    for nid in sorted(legal_now_nodes, key=_sort_key):
        sub[nid] = "future"

    # Wrapper / done bridges on paths from current to each legal-now node.
    for target in legal_now_nodes:
        stack = [target]
        seen: Set[str] = {target}
        while stack:
            n = stack.pop()
            for p in parents.get(n, []):
                if p not in graph or p in seen or p == current_node:
                    continue
                seen.add(p)
                if p in wrapper_set:
                    sub.setdefault(p, "wrapper")
                    stack.append(p)
                elif p in done_nodes:
                    sub.setdefault(p, "done")
                    stack.append(p)

    if max_back_depth > 0:
        seeds = [n for n in sub.keys() if sub.get(n) != "wrapper"]
        seen = set(seeds)
        q = deque([(n, 0) for n in seeds])
        while q:
            n, d = q.popleft()
            if d >= max_back_depth:
                continue
            for p in parents.get(n, []):
                if p in seen or p not in graph:
                    continue
                seen.add(p)
                q.append((p, d + 1))
                if p in wrapper_set:
                    sub.setdefault(p, "wrapper")
                elif p in done_nodes:
                    sub.setdefault(p, "done")

    if annotate_legal:
        annotate_frontier(sub, graph, parents, frontier_completed)

    return sub

# ---------------------------------------------------------------------------
# Midlevel phase collapsing
# ---------------------------------------------------------------------------

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
    """Collapse midlevel phases where ALL members in the subgraph are 'done'."""
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


# ---------------------------------------------------------------------------
# Fan-in done-parent collapsing (compresses dense AND-convergence patterns)
# ---------------------------------------------------------------------------
#
# After ``collapse_done_phases`` and ``extract_ego_subgraph`` (legal-now
# action nodes only), the remaining clutter source is dense
# AND-convergence: a single visible node can have many done parents
# (e.g. "Mix everything" with 12 ingredient-prep parents). All those
# done parents convey one bit of information: "all prerequisites
# satisfied". We replace each such fan-in with a single virtual node
# ``_dp_<v>`` ("N done prereqs") so the model still sees that ``v`` is
# unblocked from the AND side without rendering 12 redundant boxes.
#
# We are conservative: we only collapse done parents whose only
# in-sub child is ``v`` (so dropping them does not strand any other
# visible node). Threshold defaults to 4 so we never collapse the
# common 1-2 parent case.

DONE_FAN_IN_THRESHOLD = 4


def collapse_fan_in_done_parents(
    sub: Dict[str, str],
    children: Dict[str, List[str]],
    parents: Dict[str, List[str]],
    collapse_info: Dict[str, Dict],
    *,
    threshold: int = DONE_FAN_IN_THRESHOLD,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict]]:
    """Replace dense done-parent fan-in with a single virtual summary node.

    For each visible non-done node ``v``:
      * Find done parents in sub whose only in-sub child is ``v``.
      * If their count >= ``threshold``, drop them and insert
        ``_dp_<v>`` -> ``v`` carrying the count.

    Done parents that gate other visible nodes are left untouched, so
    no QA-relevant edge is lost.
    """
    new_sub = dict(sub)
    new_ch = {n: list(children.get(n, [])) for n in sub}
    new_pa = {n: list(parents.get(n, [])) for n in sub}
    new_collapse = dict(collapse_info)

    for v, status in list(sub.items()):
        if status not in ("current", "future"):
            continue
        done_parents = [
            p for p in new_pa.get(v, [])
            if p in new_sub and new_sub[p] in ("done", "done_col")
        ]
        if len(done_parents) < threshold:
            continue
        # Only keep done parents whose ONLY in-sub child is v: anything
        # else means the parent also gates a different visible and must
        # stay so dependency edges remain explicit.
        collapsible = [
            p for p in done_parents
            if [c for c in new_ch.get(p, []) if c in new_sub] == [v]
        ]
        if len(collapsible) < threshold:
            continue

        virt = f"_dp_{v}"
        new_sub[virt] = "done_col"
        new_ch[virt] = [v]
        new_pa[virt] = []
        new_collapse[virt] = {
            "name": f"{len(collapsible)} done prereqs",
            "count": len(collapsible),
            "members": set(collapsible),
            "kind": "fan_in_done",
        }

        for p in collapsible:
            new_sub.pop(p, None)
            new_ch.pop(p, None)
            new_pa.pop(p, None)
            # Strip references to p from any remaining adjacency list.
            for adj in (new_ch, new_pa):
                for nid, lst in adj.items():
                    if p in lst:
                        adj[nid] = [x for x in lst if x != p]
        new_pa[v].append(virt)
        # Keep adjacency for v cleaned of removed parents (already
        # handled by the per-parent strip above).

    for nid in new_sub:
        new_ch.setdefault(nid, [])
        new_pa.setdefault(nid, [])

    return new_sub, new_ch, new_pa, new_collapse


# ---------------------------------------------------------------------------
# Label / condition helpers
# ---------------------------------------------------------------------------

def _sort_key(x: str) -> int:
    # Virtual nodes are ordered next to their referent so the rendered
    # layout keeps related boxes close.
    if x.startswith("_phase_") or x.startswith("_dp_"):
        try:
            return int(x.split("_")[-1])
        except ValueError:
            return 0
    return int(x)


def _get_label(graph: Dict[str, Dict], nid: str) -> str:
    name = graph[nid].get("name", f"Node {nid}")
    return name[:40] + "..." if len(name) > 40 else name


def _get_node_label(
    graph: Dict[str, Dict], collapse_info: Dict[str, Dict], nid: str,
) -> str:
    if nid.startswith("_phase_"):
        info = collapse_info[nid]
        return f"{info['name']} ({info['count']} steps)"
    if nid.startswith("_dp_"):
        info = collapse_info.get(nid, {})
        n = info.get("count", 0)
        return f"{n} done prereqs"
    return _get_label(graph, nid)


def _wrap_image_label(label: str, *, width: int = 22, max_chars: int = 64) -> str:
    """Wrap node labels before Graphviz layout so text stays readable.

    Wide one-line labels force DOT to create very wide, shallow graphs. That
    wastes visual tokens and makes text unreadable after processor resizing.
    Wrapping preserves font size while letting the graph become slightly taller
    instead of extremely wide.
    """
    clean = " ".join(str(label).split())
    if not clean:
        return clean
    if len(clean) > max_chars:
        clean = clean[: max_chars - 3].rstrip() + "..."
    return "\n".join(
        textwrap.wrap(
            clean,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        or [clean]
    )


def _get_cond(graph: Dict[str, Dict], nid: str) -> Tuple[str, str]:
    if nid.startswith("_phase_") or nid.startswith("_dp_"):
        return "", ""
    node = graph[nid]
    return node.get("activation_condition", ""), node.get("condition_type", "")


# ---------------------------------------------------------------------------
# Image rendering (v1-style, high quality, OCR-friendly)
# ---------------------------------------------------------------------------

STATUS_COLORS = {
    "done":     {"fill": "#90EE90", "stroke": "#2E7D32", "font": "#000000"},
    "done_col": {"fill": "#4CAF50", "stroke": "#1B5E20", "font": "#FFFFFF"},
    "current":  {"fill": "#FFD700", "stroke": "#FF8C00", "font": "#000000"},
    # Frontier (= currently legal next action under the task_planner
    # check_condition semantics). Light blue fill + heavier dark blue
    # border + thicker pen so the model sees a clear "blue frontier"
    # vs "gray blocked future" distinction. The label also carries the
    # `[frontier]` text tag so the cue is redundant (text + colour).
    "frontier": {"fill": "#9FD8FF", "stroke": "#0B6FB8", "font": "#000000"},
    "future":   {"fill": "#D3D3D3", "stroke": "#808080", "font": "#000000"},
    # Wrappers / phase containers: white fill + dashed gray border so
    # they read as structural scaffolding, not as actionable steps.
    "wrapper":  {"fill": "#FFFFFF", "stroke": "#909090", "font": "#606060"},
}
DECISION_STATUS_COLORS = {
    "done":     {"fill": "#90EE90", "stroke": "#2E7D32", "font": "#000000"},
    "done_col": {"fill": "#4CAF50", "stroke": "#1B5E20", "font": "#FFFFFF"},
    "current":  {"fill": "#1E90FF", "stroke": "#003399", "font": "#FFFFFF"},
    "frontier": {"fill": "#FF8C00", "stroke": "#8B3A00", "font": "#FFFFFF"},
    "future":   {"fill": "#F0F0F0", "stroke": "#C0C0C0", "font": "#888888"},
    "wrapper":  {"fill": "#FFFFFF", "stroke": "#909090", "font": "#606060"},
}



def _find_back_edges(
    new_sub: Dict[str, str],
    new_children: Dict[str, List[str]],
) -> Set[Tuple[str, str]]:
    """Find a *minimal* set of back-edges whose removal makes the subgraph a DAG.

    Strategy
    --------
    1. Compute strongly connected components (iterative Tarjan).
    2. For every non-trivial SCC, repeatedly pick the single edge whose
       *target* has the lowest in-degree from nodes *outside* the SCC
       (i.e. the node that acts most like a "false source" inside the cycle).
       Mark that edge as a back-edge, contract it out, and repeat until
       the SCC is broken.

    Back-edges are drawn by Graphviz with ``constraint=false`` so the
    arrow is still visible but does not distort the rank layout.
    """
    from collections import deque as _deque

    ch: Dict[str, List[str]] = {
        n: [c for c in new_children.get(n, []) if c in new_sub] for n in new_sub
    }

    # ── iterative Tarjan SCC ──────────────────────────────────────
    index_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    index_map: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    sccs: List[Set[str]] = []

    for v in new_sub:
        if v in index_map:
            continue
        work = [(v, 0)]          # (node, child_index)
        while work:
            node, ci = work[-1]
            if ci == 0:
                index_map[node] = lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)
            children = ch.get(node, [])
            if ci < len(children):
                work[-1] = (node, ci + 1)
                w = children[ci]
                if w not in index_map:
                    work.append((w, 0))
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], index_map[w])
            else:
                if lowlink[node] == index_map[node]:
                    scc: Set[str] = set()
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.add(w)
                        if w == node:
                            break
                    if len(scc) > 1:
                        sccs.append(scc)
                work.pop()
                if work:
                    parent_node = work[-1][0]
                    lowlink[parent_node] = min(lowlink[parent_node], lowlink[node])

    if not sccs:
        return set()

    # ── pick minimal back-edges per SCC ───────────────────────────
    back_edges: Set[Tuple[str, str]] = set()

    for scc in sccs:
        removed: Set[Tuple[str, str]] = set()
        for _ in range(len(scc)):
            # Rebuild in-degree within SCC (minus already-removed edges)
            scc_in: Dict[str, int] = {n: 0 for n in scc}
            ext_in: Dict[str, int] = {n: 0 for n in scc}
            scc_edges: List[Tuple[str, str]] = []
            for u in scc:
                for v in ch.get(u, []):
                    if v in scc and (u, v) not in removed:
                        scc_in[v] += 1
                        scc_edges.append((u, v))
            if not scc_edges:
                break

            # Check if still cyclic (Kahn inside SCC)
            in_deg = dict(scc_in)
            q = _deque([n for n in scc if in_deg[n] == 0])
            visited = set()
            while q:
                n = q.popleft()
                visited.add(n)
                for c in ch.get(n, []):
                    if c in scc and (n, c) not in removed:
                        in_deg[c] -= 1
                        if in_deg[c] == 0:
                            q.append(c)
            remaining = scc - visited
            if not remaining:
                break  # DAG now

            # Edges among remaining cycle nodes
            cycle_edges = [(u, v) for u, v in scc_edges if u in remaining and v in remaining]
            if not cycle_edges:
                break

            # Pick edge to cut: the edge whose TARGET has the lowest
            # total in-degree (ext + internal).  A node with very few
            # incoming edges is the most "false source" inside the cycle --
            # the edge arriving at it is the true feedback arc.
            for n in remaining:
                cnt = 0
                for u in new_sub:
                    if u not in scc:
                        if n in ch.get(u, []):
                            cnt += 1
                ext_in[n] = cnt

            best_edge = min(
                cycle_edges,
                key=lambda e: ext_in.get(e[1], 0) + scc_in.get(e[1], 0),
            )
            removed.add(best_edge)
            back_edges.add(best_edge)

    return back_edges


def render_image_subgraph(
    graph: Dict[str, Dict],
    new_children: Dict[str, List[str]],
    new_parents: Dict[str, List[str]],
    new_sub: Dict[str, str],
    collapse_info: Dict[str, Dict],
    task_name: str,
    back_edges: Optional[Set[Tuple[str, str]]] = None,
    color_scheme: Optional[Dict] = None,
    thread_map: Optional[Dict[str, str]] = None,
) -> Image.Image:
    """Render subgraph as PIL Image using Graphviz. V1-style, high quality."""
    import graphviz

    n_nodes = len(new_sub)

    # Preserve OCR quality first: never reduce DPI/font as node count grows.
    # Token control comes from label wrapping, whitespace crop/compression, and
    # the 512-token render budget, not from making text smaller.
    import os as _render_os
    _font_scale = max(0.5, float(_render_os.environ.get("PROACT_RENDER_FONT_SCALE", "1.0")))
    _dpi_base = int(_render_os.environ.get("PROACT_RENDER_DPI", "96"))
    dpi = max(72, int(_dpi_base * min(_font_scale, 2.0)))
    fnode = str(max(8, int(10 * _font_scale)))
    fedge = str(max(6, int(8 * _font_scale)))
    flabel = str(max(10, int(14 * _font_scale)))
    if n_nodes <= 18:
        rankdir = "TB"
        pad, ranksep, nodesep = "0.15", "0.20", "0.12"
        label_width, label_max_chars = 24, 68
    else:
        # Dense parallel frontiers become unreadable as a single horizontal
        # rank in TB layout. LR layout stacks siblings vertically while keeping
        # dependencies explicit from left -> right.
        rankdir = "LR"
        pad, ranksep, nodesep = "0.10", "0.20", "0.06"
        label_width, label_max_chars = 20, 60

    dot = graphviz.Digraph(format="png", engine="dot")
    dot.attr(
        rankdir=rankdir, bgcolor="white", fontname="Helvetica",
        pad=pad, dpi=str(dpi),
        ranksep=ranksep, nodesep=nodesep,
        margin="0",
    )
    dot.attr("node", shape="box", style="filled,rounded",
             fontname="Helvetica", fontsize=fnode,
             margin="0.08,0.04", width="0", height="0")
    dot.attr("edge", fontname="Helvetica", fontsize=fedge)

    for nid in sorted(new_sub, key=_sort_key):
        status = new_sub[nid]
        raw_label = _get_node_label(graph, collapse_info, nid)
        label = _wrap_image_label(
            raw_label,
            width=label_width if status != "wrapper" else max(16, label_width - 4),
            max_chars=label_max_chars if status != "wrapper" else max(48, label_max_chars - 12),
        )
        _sc = color_scheme if color_scheme is not None else STATUS_COLORS
        c = _sc.get(status, STATUS_COLORS.get(status, STATUS_COLORS["future"]))
        fill_color = c["fill"]
        if thread_map and status not in ("done", "done_col", "wrapper"):
            try:
                from train.graph_add.tool.thread_map import thread_id_of_name
                tid = thread_id_of_name(_node_name(graph, nid), thread_map)
                tint = _THREAD_TINT_PALETTE.get(tid, _THREAD_TINT_PALETTE["_default"])
                fill_color = _blend_hex(fill_color, tint, alpha=0.30)
            except Exception:
                pass
        _, ctype = _get_cond(graph, nid)
        pids = [p for p in new_parents.get(nid, []) if p in new_sub]

        node_kwargs: Dict[str, str] = {
            "label": "",
            "fillcolor": fill_color,
            "color": c["stroke"],
            "fontcolor": c["font"],
        }

        if status == "done_col":
            node_kwargs["label"] = f"[done]\n{label}"
            node_kwargs["shape"] = "box3d"
            node_kwargs["penwidth"] = "1.5"
            node_kwargs["fontsize"] = fnode
        elif status == "wrapper":
            # Structural container: dashed border, smaller font, no
            # status prefix. Conveys hierarchy without claiming to be
            # an action.
            node_kwargs["label"] = label
            node_kwargs["shape"] = "box"
            node_kwargs["style"] = "dashed,rounded"
            node_kwargs["penwidth"] = "1.0"
            node_kwargs["fontsize"] = fedge
        else:
            nl = f"[{status}]\n{label}"
            if len(pids) > 1 and ctype:
                nl += f"\n({ctype} gate)"
            node_kwargs["label"] = nl
            node_kwargs["shape"] = "box"
            if status == "current":
                node_kwargs["penwidth"] = "3.0"
            elif status == "frontier":
                # Heavier border than blocked futures so the legal-now
                # cue is hard to miss even when the page has dozens of
                # boxes.
                node_kwargs["penwidth"] = "2.0"
            else:
                node_kwargs["penwidth"] = "1.5"
            node_kwargs["fontsize"] = fnode

        dot.node(f"N{nid}", **node_kwargs)

    _be = back_edges or set()
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
                if (nid, ch) in _be:
                    ea["constraint"] = "false"
                dot.edge(f"N{nid}", f"N{ch}", **ea)

    png_data = dot.pipe(format="png")
    img = Image.open(io.BytesIO(png_data)).convert("RGB")

    # Get exact node bounding boxes from graphviz layout so we can do
    # node-aware whitespace compression without touching any text/edges.
    try:
        plain_data = dot.pipe(format="plain")
        node_bboxes = _parse_plain_node_bboxes(plain_data, img.size, dpi=dpi)
    except Exception:
        node_bboxes = []

    img, crop_off = _autocrop_whitespace(img)
    translated: List[Tuple[int, int, int, int]] = []
    if node_bboxes:
        cw, ch = img.size
        ox, oy = crop_off
        for x0, y0, x1, y1 in node_bboxes:
            tx0 = max(0, x0 - ox)
            ty0 = max(0, y0 - oy)
            tx1 = min(cw, x1 - ox)
            ty1 = min(ch, y1 - oy)
            if tx1 > tx0 and ty1 > ty0:
                translated.append((tx0, ty0, tx1, ty1))
    # Text-preserving adaptive whitespace compression.
    if translated:
        img = _adaptive_compress(img, translated, MAX_RENDER_PIXELS)
    # Last-resort safety net. This should be rare after label wrapping and
    # whitespace compression; if it triggers, the image is still bounded to the
    # 512-token training budget instead of risking OOM.
    if img.size[0] * img.size[1] > MAX_RENDER_PIXELS:
        img = _cap_pixels(img, MAX_RENDER_PIXELS)
    return img


def _autocrop_whitespace(
    img: Image.Image,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    threshold: int = 8,
    padding: int = 6,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Trim near-uniform background border so vision tokens cover only content.

    Returns the cropped image and the (left, top) crop offset in original
    image coordinates so callers can translate any external coordinates
    (e.g. graphviz node bboxes) into the cropped image space.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    bg = Image.new("RGB", img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    if threshold > 0:
        diff = diff.point(lambda v: 0 if v <= threshold else v)
    bbox = diff.getbbox()
    if bbox is None:
        return img, (0, 0)
    left, top, right, bottom = bbox
    width, height = img.size
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    if right <= left or bottom <= top:
        return img, (0, 0)
    return img.crop((left, top, right, bottom)), (left, top)


def _parse_plain_node_bboxes(
    plain_bytes: bytes,
    png_size: Tuple[int, int],
    dpi: int = 96,
) -> List[Tuple[int, int, int, int]]:
    """Parse `dot -Tplain` output into node bboxes in PNG pixel coordinates.

    Graphviz `plain` format starts with::

        graph <scale> <width_inches> <height_inches>
        node <name> <cx_in> <cy_in> <w_in> <h_in> ...

    The PNG is the graph rectangle plus a `pad` margin on every side; we
    infer that pad from the size delta. ``cy`` is given in y-up inches and
    we convert it to y-down pixels so the bboxes line up with PIL.
    """
    try:
        text = plain_bytes.decode("utf-8", errors="replace")
    except Exception:
        return []
    lines = text.strip().split("\n")
    if not lines:
        return []
    header = lines[0].split()
    if len(header) < 4 or header[0] != "graph":
        return []
    try:
        graph_w_in = float(header[2])
        graph_h_in = float(header[3])
    except ValueError:
        return []
    if graph_w_in <= 0 or graph_h_in <= 0:
        return []
    png_w, png_h = png_size
    pad_x_px = max(0.0, (png_w - graph_w_in * dpi) / 2.0)
    pad_y_px = max(0.0, (png_h - graph_h_in * dpi) / 2.0)
    bboxes: List[Tuple[int, int, int, int]] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts or parts[0] != "node" or len(parts) < 6:
            continue
        try:
            x_in = float(parts[2])
            y_in = float(parts[3])
            w_in = float(parts[4])
            h_in = float(parts[5])
        except ValueError:
            continue
        cx_px = pad_x_px + x_in * dpi
        cy_px = pad_y_px + (graph_h_in - y_in) * dpi
        wp = w_in * dpi
        hp = h_in * dpi
        bboxes.append((
            int(round(cx_px - wp / 2)),
            int(round(cy_px - hp / 2)),
            int(round(cx_px + wp / 2)),
            int(round(cy_px + hp / 2)),
        ))
    return bboxes


def _compress_axis(keep_mask: np.ndarray, min_band: int, target_band: int) -> List[int]:
    """Return list of source indices: keep all True positions; subsample
    contiguous False runs longer than ``min_band`` down to ``target_band``
    evenly spaced indices."""
    n = int(keep_mask.shape[0])
    out: List[int] = []
    i = 0
    while i < n:
        if keep_mask[i]:
            out.append(i)
            i += 1
            continue
        j = i
        while j < n and not keep_mask[j]:
            j += 1
        run = j - i
        if run > min_band and target_band > 0:
            idx = np.linspace(i, j - 1, target_band).round().astype(int)
            out.extend(int(v) for v in idx.tolist())
        else:
            out.extend(range(i, j))
        i = j
    return out


def _build_protection_masks(
    img: Image.Image,
    bboxes: List[Tuple[int, int, int, int]],
    *,
    padding: int = 8,
    dark_threshold: int = 180,
    dark_count_threshold: int = 20,
):
    """Build the row/col protection masks for an image. Runs the
    expensive dark-pixel + text scan exactly once per image; the masks
    are then re-used by every compression pass.

    Returns ``(arr, is_node_row, is_node_col)`` where ``arr`` is the
    numpy view of the image and the masks are 1-D bool arrays."""
    arr = np.asarray(img)
    if arr.ndim < 2:
        return arr, None, None
    h, w = arr.shape[:2]
    is_node_row = np.zeros(h, dtype=bool)
    is_node_col = np.zeros(w, dtype=bool)
    for x0, y0, x1, y1 in bboxes:
        ry0, ry1 = max(0, y0 - padding), min(h, y1 + padding)
        if ry1 > ry0:
            is_node_row[ry0:ry1] = True
        cx0, cx1 = max(0, x0 - padding), min(w, x1 + padding)
        if cx1 > cx0:
            is_node_col[cx0:cx1] = True

    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    is_dark = gray < dark_threshold
    row_dark_counts = is_dark.sum(axis=1)
    col_dark_counts = is_dark.sum(axis=0)
    text_rows = row_dark_counts >= dark_count_threshold
    is_node_row |= text_rows
    is_node_col |= (col_dark_counts >= dark_count_threshold)

    # Row -> col extension: protect the horizontal span of every text-dense
    # row so column compression cannot chop the title or edge labels.
    # Vectorised version: leftmost / rightmost dark pixel per row.
    if text_rows.any():
        any_dark = is_dark.any(axis=1)
        # leftmost dark column per row
        first = np.argmax(is_dark, axis=1)
        last = w - 1 - np.argmax(is_dark[:, ::-1], axis=1)
        protect = text_rows & any_dark
        for y in np.nonzero(protect)[0]:
            is_node_col[first[y]:last[y] + 1] = True
    return arr, is_node_row, is_node_col


def _apply_band_compression(
    arr,
    is_node_row,
    is_node_col,
    min_band: int,
    target_band: int,
    min_target_band: int = 2,
) -> Image.Image:
    """Apply one pass of band compression with cached masks. Compares
    contiguous unprotected runs against ``min_band`` and subsamples them
    down to at most max(``min_target_band``, ``target_band``) indices."""
    eff_target = max(min_target_band, target_band)
    new_rows = _compress_axis(is_node_row, min_band, eff_target)
    if new_rows:
        arr = arr[new_rows]
    new_cols = _compress_axis(is_node_col, min_band, eff_target)
    if new_cols:
        arr = arr[:, new_cols]
    return Image.fromarray(arr)


def _compress_non_node_bands(
    img: Image.Image,
    bboxes: List[Tuple[int, int, int, int]],
    *,
    padding: int = 8,
    min_band: int = 20,
    target_band: int = 10,
    dark_threshold: int = 180,
    dark_count_threshold: int = 20,
    min_target_band: int = 2,
) -> Image.Image:
    """Compress contiguous rows / cols that are not part of any node or text.
    Convenience wrapper for callers that only want a single compression pass.
    For multi-pass compression use _build_protection_masks +
    _apply_band_compression and you'll save the mask build cost."""
    if not bboxes:
        return img
    arr, is_node_row, is_node_col = _build_protection_masks(
        img, bboxes, padding=padding,
        dark_threshold=dark_threshold,
        dark_count_threshold=dark_count_threshold,
    )
    if is_node_row is None:
        return img
    return _apply_band_compression(arr, is_node_row, is_node_col,
                                   min_band, target_band, min_target_band)







def _adaptive_compress(
    img: Image.Image,
    bboxes: List[Tuple[int, int, int, int]],
    max_pixels: int,
) -> Image.Image:
    """Iteratively compress whitespace bands harder until the image fits.

    Builds the row/col protection masks ONCE for this image and then
    runs every schedule pass against those cached masks. As a result
    every protected pixel keeps its original resolution; only pure
    whitespace bands get shorter. Text never blurs.

    The schedule starts at a gentle setting and ratchets towards the
    most aggressive (which still keeps a 2-pixel floor for edge lines).
    Returns the first compressed image that fits below ``max_pixels``,
    or the most-compressed result if even max compression is not enough.
    """
    if not bboxes:
        return img
    arr, is_node_row, is_node_col = _build_protection_masks(img, bboxes)
    if is_node_row is None:
        return img
    total_px = img.size[0] * img.size[1]
    if total_px <= max_pixels:
        # Under the cap, but still squeeze out large whitespace gaps so
        # they don't waste vision tokens. Use a single gentle pass that
        # only collapses bands wider than 20px (edge lines stay intact).
        # Skip entirely for small images where there is little to gain.
        if total_px <= max_pixels * 0.6:
            return img
        gentle = _apply_band_compression(arr, is_node_row, is_node_col,
                                         20, 10)
        return gentle
    schedule = [
        (20, 10),  # gentle
        (14,  6),  # moderate
        (10,  4),  # strong
        ( 8,  3),  # very strong
        ( 6,  2),  # maximum (edge lines still keep 2 pixels)
    ]
    cur = img
    for min_band, target_band in schedule:
        cur = _apply_band_compression(arr, is_node_row, is_node_col,
                                      min_band, target_band)
        if cur.size[0] * cur.size[1] <= max_pixels:
            return cur
    return cur


def _cap_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """Uniformly downscale ``img`` so its pixel count <= ``max_pixels``.

    This is the safety net for extremely complex task graphs whose 70+ nodes
    produce 5K-35K Qwen2.5-VL vision tokens per image even after node-aware
    band compression. We use LANCZOS so node text degrades gracefully; for
    the worst-case 35K-token graph this scales by 0.21x, taking ~18px text
    down to ~4px which is at the lower bound of VLM legibility but is the
    necessary trade-off for keeping training within memory.
    """
    w, h = img.size
    px = w * h
    if px <= max_pixels:
        return img
    scale = (max_pixels / px) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    # Guard against rounding/product drift above max_pixels.
    while new_w * new_h > max_pixels and (new_w > 1 or new_h > 1):
        if new_w >= new_h and new_w > 1:
            new_w -= 1
        elif new_h > 1:
            new_h -= 1
        else:
            break
    return img.resize((new_w, new_h), Image.LANCZOS)

# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def render_text_subgraph(
    graph: Dict[str, Dict],
    new_children: Dict[str, List[str]],
    new_parents: Dict[str, List[str]],
    new_sub: Dict[str, str],
    collapse_info: Dict[str, Dict],
    task_name: str,
) -> str:
    """Render subgraph as structured plain text."""
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


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def _prune_dead_done_branches(
    new_sub: Dict[str, str],
    new_ch: Dict[str, List[str]],
    new_pa: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Remove done/wrapper nodes whose subtrees contain no frontier/future/current nodes.

    A done branch that leads nowhere actionable wastes visual space and tokens.
    We keep a done/wrapper node only if it is an ancestor of at least one
    frontier, future, or current node in the rendered subgraph.
    """
    from collections import deque as _dq

    keep_statuses = {"frontier", "future", "current"}
    seeds = [n for n, s in new_sub.items() if s in keep_statuses]

    # Walk backward from seeds to mark all required ancestors
    needed: Set[str] = set(seeds)
    q = _dq(seeds)
    while q:
        n = q.popleft()
        for p in new_pa.get(n, []):
            if p in new_sub and p not in needed:
                needed.add(p)
                q.append(p)

    # Prune nodes not in needed set
    pruned = {n: s for n, s in new_sub.items() if n in needed}
    pruned_ch = {n: [c for c in new_ch.get(n, []) if c in pruned] for n in pruned}
    pruned_pa = {n: [p for p in new_pa.get(n, []) if p in pruned] for n in pruned}

    return pruned, pruned_ch, pruned_pa


def _ensure_connectivity(
    sub: Dict[str, str],
    sub_ch: Dict[str, List[str]],
    sub_pa: Dict[str, List[str]],
    full_children: Dict[str, List[str]],
    full_parents: Dict[str, List[str]],
    current_node: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Reconnect orphan components that contain frontier/future nodes.

    After dead-done-branch pruning, some frontier/future clusters may become
    disconnected from the current node because intermediate bridge wrappers
    were missing from the original subgraph extraction.  We find the shortest
    path in the *full* task graph from the main component (containing current)
    to each orphan component and splice in the missing bridge nodes.
    """
    from collections import deque as _dq

    if current_node is None or current_node not in sub:
        return sub, sub_ch, sub_pa

    # Find connected components (undirected within sub)
    adj: Dict[str, Set[str]] = {n: set() for n in sub}
    for n in sub:
        for c in sub_ch.get(n, []):
            if c in sub:
                adj[n].add(c)
                adj[c].add(n)

    visited: Set[str] = set()
    components: List[Set[str]] = []
    for start in sub:
        if start in visited:
            continue
        comp: Set[str] = set()
        q = _dq([start])
        while q:
            n = q.popleft()
            if n in comp:
                continue
            comp.add(n)
            for nb in adj[n]:
                if nb not in comp:
                    q.append(nb)
        visited |= comp
        components.append(comp)

    if len(components) <= 1:
        return sub, sub_ch, sub_pa

    # Identify the main component (contains current_node)
    main_comp = next(c for c in components if current_node in c)
    keep_statuses = {"frontier", "future", "current"}

    # Make mutable copies
    sub = dict(sub)
    sub_ch = {n: list(sub_ch.get(n, [])) for n in sub}
    sub_pa = {n: list(sub_pa.get(n, [])) for n in sub}

    for comp in components:
        if comp is main_comp:
            continue
        has_actionable = any(sub.get(n) in keep_statuses for n in comp)
        if not has_actionable:
            continue

        # BFS backward from orphan comp nodes through full graph
        # to find shortest path to any node in main_comp.
        found_path = None
        bfs_visited: Dict[str, List[str]] = {}
        q = _dq()
        for n in comp:
            q.append(n)
            bfs_visited[n] = [n]
        while q and found_path is None:
            n = q.popleft()
            for p in full_parents.get(n, []):
                if p in bfs_visited:
                    continue
                path = bfs_visited[n] + [p]
                bfs_visited[p] = path
                if p in main_comp:
                    found_path = list(reversed(path))
                    break
                if len(path) <= 8:
                    q.append(p)

        if found_path is None:
            continue

        # Splice bridge nodes into the sub
        for i, nid in enumerate(found_path):
            if nid not in sub:
                sub[nid] = "wrapper"
                sub_ch[nid] = []
                sub_pa[nid] = []
            if i + 1 < len(found_path):
                nxt = found_path[i + 1]
                if nxt not in sub_ch.get(nid, []):
                    sub_ch.setdefault(nid, []).append(nxt)
                if nid not in sub_pa.get(nxt, []):
                    sub_pa.setdefault(nxt, []).append(nid)

        main_comp.update(n for n in found_path)

    return sub, sub_ch, sub_pa


def build_subgraph_for_step(
    taxonomy: Dict[str, Dict[str, Dict]],
    task_name: str,
    completed_steps: List[str],
    current_step: str,
    mode: str = "image",
    max_depth: int = DEFAULT_FORWARD_DEPTH,
    max_back_depth: int = 1,
    include_all_done: bool = False,
    render_mode: str = "default",
    text_format: str = "legacy",
    future_steps: Optional[List[str]] = None,
    annotation_path: Optional[str] = None,
    tg_cache: Optional[Dict[str, Any]] = None,
    text_omit_legal_pool: bool = False,
) -> Optional[Any]:
    """Build subgraph for a given task + progress state.

    Args:
        taxonomy: {task_name: {node_id_str: node_info}}
        task_name: Predicted task name from Phase 1.
        completed_steps: List of completed step name strings.
        current_step: Predicted current step from Phase 1.
        mode: "image" (PIL.Image) or "text" (str).
        max_depth: Retained for API compatibility; subgraph action nodes are
            wave-1 legal-now only (not a depth-5 forward horizon).
        max_back_depth: Backward parent expansion depth. Default 1 makes
            every visible node's direct parents (gates) visible.
            The model can read blocked-by from node colors alone
            (a future node whose parent is gray rather than green is
            blocked by that parent).
        include_all_done: When False (default) only done nodes that
            actually gate a visible non-done node are shown. When True,
            every node in ``completed_steps`` is added (legacy behaviour).
        render_mode: ``default`` (all legal → frontier) or
            ``cross_thread_frontier`` (only cross-thread legal → frontier;
            uses ``apa_next_action`` thread + legality helpers).
        text_format: When ``mode="text"``: ``legacy`` (dependency list),
            ``compact_yaml``, or ``structured_json`` (no gold field).
        future_steps: GT human future steps (for ``get_legal_robot_actions``
            human_immediate window; same as training label path).
        annotation_path: Path to ``all_annotations.json`` for TaskGraphManager.
        tg_cache: Optional shared TaskGraphManager cache dict.

    Returns:
        PIL.Image for image mode, str for text mode, or None.
    """
    # ── Disk cache (PROACT_SUBGRAPH_CACHE=/path enables it) ────────────
    import os as _os, hashlib as _hl
    _sgcache = _os.environ.get('PROACT_SUBGRAPH_CACHE', '')
    _cache_path = None
    _render_mode = str(render_mode or "default").strip().lower()
    if _render_mode not in _RENDER_MODES:
        _render_mode = "default"

    if _sgcache and mode == 'image':
        from pathlib import Path as _Path
        _render_style_for_key = _os.environ.get('PROACT_RENDER_STYLE', 'normal')
        _key = '|'.join([task_name] + list(completed_steps) + [current_step,
                str(max_depth), str(max_back_depth), str(include_all_done),
                _render_style_for_key, _render_mode])
        _cache_path = _Path(_sgcache) / (_hl.md5(_key.encode('utf-8')).hexdigest() + '.png')
        if _cache_path.exists():
            try:
                return Image.open(_cache_path).convert('RGB')
            except Exception:
                _cache_path = None
    # ────────────────────────────────────────────────────────────────────
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
        max_depth=max_depth, max_back_depth=max_back_depth,
        include_all_done=include_all_done,
        completed_step_names=list(completed_steps),
    )

    if not sub:
        return None

    new_sub, new_ch, new_pa, collapse_info = collapse_done_phases(
        sub, graph, children, parents,
    )

    new_sub, new_ch, new_pa, collapse_info = collapse_fan_in_done_parents(
        new_sub, new_ch, new_pa, collapse_info,
    )

    new_sub, new_ch, new_pa = _prune_dead_done_branches(new_sub, new_ch, new_pa)

    new_sub, new_ch, new_pa = _ensure_connectivity(
        new_sub, new_ch, new_pa, children, parents, current_node,
    )

    thread_map_for_render: Optional[Dict[str, str]] = None
    if _render_mode == "cross_thread_frontier":
        new_sub, thread_map_for_render = apply_cross_thread_frontier_filter(
            new_sub,
            graph,
            task_name=task_name,
            completed_steps=list(completed_steps),
            current_step=current_step,
            future_steps=list(future_steps or []),
            annotation_path=annotation_path,
            tg_cache=tg_cache,
        )
    elif _render_mode == "legal_only":
        _drop = [nid for nid, st in new_sub.items() if st == "future"]
        for _nid in _drop:
            new_sub.pop(_nid, None)
        new_ch = {
            n: [c for c in ch if c in new_sub]
            for n, ch in new_ch.items() if n in new_sub
        }
        new_pa = {
            n: [p for p in pa if p in new_sub]
            for n, pa in new_pa.items() if n in new_sub
        }

    back_edges = _find_back_edges(new_sub, new_ch)

    if mode == "image":
        _render_style = _os.environ.get('PROACT_RENDER_STYLE', 'normal')
        _color_scheme = DECISION_STATUS_COLORS if _render_style == 'decision' else None
        _img = render_image_subgraph(
            graph, new_ch, new_pa, new_sub, collapse_info, task_name,
            back_edges=back_edges, color_scheme=_color_scheme,
            thread_map=thread_map_for_render,
        )
        if _img is not None and _cache_path is not None:
            try:
                _cache_path.parent.mkdir(parents=True, exist_ok=True)
                _tmp = _cache_path.with_suffix(".tmp")
                _img.save(_tmp, format="PNG")
                _tmp.rename(_cache_path)
            except Exception:
                pass
        return _img
    elif mode == "text":
        from train.graph_add.tool.graph_text_serializer import render_subgraph_text

        return render_subgraph_text(
            text_format=text_format,
            task_name=task_name,
            current_step=current_step,
            completed_steps=list(completed_steps),
            graph=graph,
            new_sub=new_sub,
            new_ch=new_ch,
            new_pa=new_pa,
            collapse_info=collapse_info,
            thread_map=thread_map_for_render,
            future_steps=list(future_steps or []),
            annotation_path=annotation_path,
            tg_cache=tg_cache,
            omit_legal_pool=text_omit_legal_pool,
        )
    else:
        raise ValueError(f"Unknown subgraph mode: {mode}")