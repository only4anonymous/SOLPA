"""Eval-aligned next_action relabeling for APA / SS / metric-aligned training (W2i).

Uses TaskGraphManager + EntropyPlanner (same stack as onestep_probe_scorer /
eval_onestep_end2end) so training labels match E-w2 metric definitions when
``onestep_action_selector=model``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from train.graph_add.tool.start_node_filter import filter_task_root_starts_from_names

DEFAULT_WAIT_ACTION = "Wait / None"
IMMEDIATE_M = 3  # match onestep_probe_scorer._select_entropy_action default

_LABEL_MODES = frozenset({"teacher", "apa_parallel", "ss_aligned", "metric_aligned"})


def _proact_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_task_planner():
    tp_path = _proact_root() / "test/onestep_planning/task_planner.py"
    if not tp_path.exists():
        raise FileNotFoundError(f"task_planner not found: {tp_path}")
    spec = importlib.util.spec_from_file_location("proact_task_planner_apa", tp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {tp_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["proact_task_planner_apa"] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_mgr(
    task_name: str,
    annotation_path: str,
    tg_cache: Dict[str, Any],
):
    task_name = (task_name or "").strip()
    if not task_name or not annotation_path:
        return None
    path = Path(annotation_path)
    if not path.exists():
        return None
    if task_name not in tg_cache:
        tp = _load_task_planner()
        tg_cache[task_name] = tp.TaskGraphManager(str(path), task_name)
    return tg_cache[task_name]


def _thread_id(mgr, action: str) -> str:
    return (getattr(mgr, "thread_map", None) or {}).get(action, "serial_main")


def _is_cross_thread(mgr, action: str, human_now: str) -> bool:
    if not action or not human_now:
        return False
    return _thread_id(mgr, action) != _thread_id(mgr, human_now)


def _legal_now(
    mgr,
    completed: List[str],
    future_steps: List[str],
    *,
    immediate_m: int = IMMEDIATE_M,
) -> Set[str]:
    human_immediate = list(future_steps or [])[: max(0, immediate_m)]
    try:
        legal = mgr.get_legal_robot_actions(list(completed), human_immediate) or []
    except TypeError:
        legal = mgr.get_legal_robot_actions(list(completed)) or []
    legal = filter_task_root_starts_from_names(mgr, legal)
    out: Set[str] = set()
    for a in legal:
        s = str(a or "").strip()
        if s and s.lower() != "terminate":
            out.add(s)
    return out


def _immediate_saved_ok(
    mgr,
    action: str,
    completed: List[str],
    future_steps: List[str],
) -> bool:
    if not action:
        return False
    remaining = (
        mgr.get_remaining_steps(list(completed))
        if hasattr(mgr, "get_remaining_steps")
        else []
    )
    in_remaining = action in remaining if remaining else action in (future_steps or [])
    not_done = action not in (completed or [])
    return bool(in_remaining and not_done)


def _min_entropy_action(
    mgr,
    completed: List[str],
    pool: List[str],
    *,
    cand_order: Optional[List[str]] = None,
) -> str:
    if not pool:
        return DEFAULT_WAIT_ACTION
    tp = _load_task_planner()
    planner = tp.EntropyPlanner(mgr)
    order = list(cand_order or pool)
    best = pool[0]
    best_ent = float("inf")
    best_pos = 10**9
    for i, action in enumerate(pool):
        try:
            ent = float(
                planner.calculate_entropy(
                    robot_history=[],
                    candidate_action=action,
                    human_history=list(completed),
                )
            )
        except Exception:
            ent = float("inf")
        pos = order.index(action) if action in order else 10**9
        if (ent < best_ent - 1e-9) or (abs(ent - best_ent) <= 1e-9 and pos < best_pos):
            best_ent = ent
            best = action
            best_pos = pos
    return best if best_ent < float("inf") else DEFAULT_WAIT_ACTION



TERMINATE_STEP = "Terminate"
_FUTURE_STEPS_TARGETS = frozenset({"horizon", "remaining"})


def resolve_future_steps_label(
    mode: str,
    *,
    task_name: str,
    completed_steps: List[str],
    horizon_future_steps: List[str],
    annotation_path: str,
    tg_cache: Dict[str, Any],
    is_trigger: bool = True,
) -> List[str]:
    """Return future_steps CE / generation training labels."""
    if not is_trigger:
        return []
    mode = (mode or "horizon").strip().lower()
    if mode not in _FUTURE_STEPS_TARGETS:
        mode = "horizon"

    horizon = [str(x).strip() for x in (horizon_future_steps or []) if str(x).strip()]
    if mode == "horizon":
        return horizon

    mgr = _get_mgr(task_name, annotation_path, tg_cache)
    if mgr is None:
        return horizon

    completed = list(completed_steps or [])
    remaining: List[str] = []
    if hasattr(mgr, "get_remaining_steps"):
        try:
            remaining = [str(x).strip() for x in (mgr.get_remaining_steps(completed) or []) if str(x).strip()]
        except Exception:
            remaining = []
    if not remaining:
        remaining = list(horizon)

    steps: List[str] = []
    for a in remaining:
        if a.lower() == "terminate":
            continue
        steps.append(a)
    if not steps:
        return [TERMINATE_STEP]
    if steps[-1].lower() != "terminate":
        steps = list(steps) + [TERMINATE_STEP]
    return steps


def resolve_next_action_label(
    mode: str,
    *,
    task_name: str,
    completed_steps: List[str],
    current_step: str,
    future_steps: List[str],
    teacher_action: str,
    annotation_path: str,
    tg_cache: Dict[str, Any],
    wait_token: str = DEFAULT_WAIT_ACTION,
    is_trigger: bool = True,
) -> str:
    """Return the next_action training label for the given mode."""
    if not is_trigger:
        return wait_token

    mode = (mode or "teacher").strip().lower()
    if mode not in _LABEL_MODES:
        mode = "teacher"
    if mode == "teacher":
        return (teacher_action or "").strip() or wait_token

    mgr = _get_mgr(task_name, annotation_path, tg_cache)
    if mgr is None:
        return (teacher_action or "").strip() or wait_token

    completed = list(completed_steps or [])
    human_now = (current_step or "").strip()
    future = list(future_steps or [])
    legal = _legal_now(mgr, completed, future)
    legal.discard(human_now)

    def _ss_candidates(source: List[str]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for a in source:
            s = str(a or "").strip()
            if not s or s in seen or s not in legal:
                continue
            if not _immediate_saved_ok(mgr, s, completed, future):
                continue
            seen.add(s)
            out.append(s)
        return out

    def _apa_candidates(source: List[str]) -> List[str]:
        return [a for a in source if _is_cross_thread(mgr, a, human_now)]

    if mode == "ss_aligned":
        cand = _ss_candidates(future)
        if not cand:
            cand = _ss_candidates(sorted(legal))
        return _min_entropy_action(mgr, completed, cand, cand_order=cand) if cand else wait_token

    if mode == "apa_parallel":
        cross = _apa_candidates(sorted(legal))
        return _min_entropy_action(mgr, completed, cross, cand_order=cross) if cross else wait_token

    if mode == "saved_cross_strict":
        cross = _apa_candidates(sorted(legal))
        saved_cross = _ss_candidates(cross)
        if saved_cross:
            return _min_entropy_action(mgr, completed, saved_cross, cand_order=saved_cross)
        return wait_token   # NO fallback to any cross-thread -> Wait when nothing is saved

    if mode == "metric_aligned":
        cross = _apa_candidates(sorted(legal))
        saved_cross = _ss_candidates(cross)
        if saved_cross:
            return _min_entropy_action(mgr, completed, saved_cross, cand_order=saved_cross)
        if cross:
            return _min_entropy_action(mgr, completed, cross, cand_order=cross)
        return wait_token

    return (teacher_action or "").strip() or wait_token


# Back-compat alias from design doc
resolve_apa_next_action = resolve_next_action_label
