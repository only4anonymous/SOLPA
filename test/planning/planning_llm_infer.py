#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 OpenAI 兼容接口（Azure/Gemini 也可）对 planning_states.jsonl 逐状态生成机器人动作。

依赖：
- pip install openai
- API 配置参考 test/l2/run_eval_l2_gpt4o.sh / run_eval_l2_gemini.sh

输出 JSONL（逐状态）：
{
  "state_id": "...",
  "video_id": "...",
  "t": 0,
  "model": "...",
  "action": "...",
  "reason": "...",
  "confidence": 0.7,
  "candidate_actions": [...],
  "raw_response": "..."
}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_PROMPT = """You are a robot action planner collaborating with a human on a procedural task.
Choose exactly ONE next robot action from the provided candidate list.

Hard constraints:
1) You must output a JSON object with key "action".
2) The value must be exactly one string from CANDIDATE_ACTIONS, or "Wait / None" if the list is empty.
3) Prefer actions that are parallel to the human's current thread (i.e., from a different execution path/branch in the task graph), when possible. Thread (thread): an independent sub-process branch in the task graph induced by the same mid-level structural node; different threads have no shared nodes."""

USER_PROMPT_TEMPLATE = """TASK: {task_name}

TASK_GRAPH (compact):
{task_graph}

COMPLETED_STEPS (world state):
{completed}

HUMAN_IMMEDIATE_NEXT (reference only):
{human_immediate}

HUMAN_FUTURE_HORIZON (reference, may be skipped):
{human_future}

CANDIDATE_ACTIONS:
{candidates}

Return JSON only:
{{"action": "...", "reason": "...", "confidence": 0-1}}"""


# -----------------------------
# Human simulation helpers (keep semantics aligned with planning_eval.py)
# -----------------------------
def _is_real_action_node(graph, name: str) -> bool:
    """
    Keep consistent with planning_eval.py:
    - exclude midlevel/structural nodes
    - exclude the terminal state node "Terminate"
    """
    s = str(name or "").strip()
    if not s or s == "Wait / None":
        return False
    if s.lower() == "terminate":
        return False
    node = (getattr(graph, "name2node", None) or {}).get(s)
    if not node:
        return False
    try:
        if bool(node.get("is_midlevel", False)):
            return False
    except Exception:
        pass
    return True


def _terminate_allowed_now(*, graph, remaining: List[str], completed: List[str]) -> bool:
    """
    Hard rule (same as planning_eval.py):
    - "Terminate" can ONLY be executed by the human.
    - It can ONLY be executed when all remaining real action steps are already completed.
    """
    for s in (remaining or []):
        if str(s or "").strip().lower() == "terminate":
            continue
        if _is_real_action_node(graph, str(s)) and (str(s) not in completed):
            return False
    return True


def _human_choose_and_apply(
    *,
    graph,
    completed: List[str],
    remaining: List[str],
    human_hist: List[str],
    robot_hist: List[str],
    in_progress: Optional[str],
    human_min_entropy: bool,
    no_switch: bool,
) -> None:
    """
    Apply ONE human action, matching planning_eval.py semantics as closely as possible.
    Mutates completed/remaining/human_hist in place.
    """

    def _enabled(step_name: str) -> bool:
        step_name = str(step_name or "").strip()
        if not step_name:
            return False
        if step_name in completed:
            return False
        if in_progress and step_name == in_progress:
            return False
        # Hard terminal rule
        if step_name.lower() == "terminate":
            return _terminate_allowed_now(graph=graph, remaining=remaining, completed=completed)
        # Human GT head is always allowed (GT/taxonomy mismatch guard).
        if remaining and step_name == str(remaining[0]):
            return True
        node = (getattr(graph, "name2node", None) or {}).get(step_name)
        cond = node.get("activation_condition", "TRUE") if node else "TRUE"
        try:
            return bool(graph.check_condition(cond, completed))
        except Exception:
            return False

    if not remaining:
        return

    # Precedence: no_switch overrides human_min_entropy.
    if bool(no_switch):
        head = str(remaining[0])
        human_choice = head if _enabled(head) else None
    elif bool(human_min_entropy):
        # 1) Prefer head if enabled.
        head = str(remaining[0])
        human_choice = head if _enabled(head) else None
        # 2) Otherwise choose an enabled step in a thread that robot touched least (heuristic).
        if human_choice is None:
            robot_counts: Dict[str, int] = {}
            for a in (robot_hist or []):
                if not _is_real_action_node(graph, a):
                    continue
                tid = (getattr(graph, "thread_map", None) or {}).get(str(a)) or "serial_main"
                robot_counts[str(tid)] = robot_counts.get(str(tid), 0) + 1
            best = None
            best_key = None
            for j, step in enumerate(remaining):
                s = str(step)
                if not _enabled(s):
                    continue
                tid = (getattr(graph, "thread_map", None) or {}).get(s) or "serial_main"
                key = (robot_counts.get(str(tid), 0), j)
                if best_key is None or key < best_key:
                    best_key = key
                    best = s
            human_choice = best
    else:
        # default: earliest-enabled
        human_choice = None
        for step in (remaining or []):
            s = str(step)
            if _enabled(s):
                human_choice = s
                break

    if human_choice is None:
        return

    if human_choice not in completed:
        completed.append(human_choice)
    human_hist.append(human_choice)
    if human_choice in remaining:
        try:
            remaining.remove(human_choice)
        except ValueError:
            pass


# -----------------------------
# API helpers
# -----------------------------
def _normalize_api_base(api_base: str) -> str:
    if not api_base:
        return api_base
    base = api_base.rstrip("/")
    suffix = "/chat/completions"
    if base.lower().endswith(suffix):
        base = base[: -len(suffix)]
    return base


def _safe_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """兼容 markdown 代码块 / 部分 JSON 的简易解析。"""
    if not text:
        return None
    text = text.strip()
    import re

    pat = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    m = re.search(pat, text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    # 粗略捕获第一个 {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _extract_action_from_raw(raw: str, candidates: List[str]) -> Tuple[str, str, Optional[float]]:
    """
    Robustly extract (action, reason, confidence) from raw LLM response.
    Guarantees returned action is never empty:
      - prefer valid JSON
      - fallback to regex "action": "..."
      - fallback to picking a candidate mentioned in raw
      - else "Wait / None"
    """
    import re

    raw = (raw or "").strip()
    candidates = [c for c in (candidates or []) if isinstance(c, str) and c.strip()]
    cand_set = set(candidates)

    parsed = _safe_extract_json(raw) or {}
    action = str(parsed.get("action", "") or "").strip()
    reason = str(parsed.get("reason", "") or "").strip()
    conf_val: Optional[float] = None
    try:
        conf_val = float(parsed.get("confidence")) if parsed.get("confidence") is not None else None
    except Exception:
        conf_val = None

    # If JSON parsing failed (often due to truncated code blocks), try regex for action/reason fields.
    if not action:
        m = re.search(r"(?i)[\"']action[\"']\s*:\s*[\"']([^\"']+)[\"']", raw)
        if m:
            action = m.group(1).strip()
    if not reason:
        m = re.search(r"(?i)[\"']reason[\"']\s*:\s*[\"']([^\"']+)[\"']", raw)
        if m:
            reason = m.group(1).strip()

    # Normalize common waits
    if action.strip().lower() in {"wait", "none", "wait/none", "wait / none"}:
        action = "Wait / None"

    # If action still empty, try to pick a candidate that appears in raw.
    if not action:
        raw_low = raw.lower()
        best = None
        best_pos = None
        for c in candidates:
            pos = raw_low.find(c.lower())
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best = c
        if best:
            action = best
            if not reason:
                reason = "picked_from_raw_response"

    # Final guard: never allow empty action
    if not action:
        action = "Wait / None"
        if not reason:
            reason = "empty_action_fallback"

    # Enforce candidate membership (except Wait / None)
    if action != "Wait / None" and action not in cand_set:
        # Do NOT auto-rewrite to some other candidate (it makes action/reason inconsistent).
        # If the model output is not in candidates, we must wait.
        reason = (reason + " | " if reason else "") + "action_not_in_candidates"
        action = "Wait / None"

    return action, reason, conf_val


def _http_post_json(url: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        try:
            return json.loads(body)
        except Exception:
            # unify error shape to simplify callers
            return {"error": {"message": body, "code": e.code}}
    except Exception as e:
        return {"error": {"message": str(e), "code": None}}


def _build_client(api_base: str, api_key: str, api_version: str, model: str, transport: str = "sdk"):
    api_base = _normalize_api_base(api_base)
    use_azure = "azure.com" in (api_base or "").lower()
    is_gemini = (
        "aiplatform.googleapis.com" in (api_base or "")
        or api_key == "GEMINI_GCLOUD"
        or os.getenv("USE_GEMINI", "").lower() in ("1", "true", "yes")
        or model.startswith("google/")
    )
    transport = (transport or "sdk").strip().lower()
    if transport not in {"sdk", "http"}:
        raise ValueError(f"invalid transport: {transport} (expected sdk|http)")

    # HTTP mode: use direct OpenAI-compatible REST; no SDK required.
    if transport == "http":
        return {
            "_transport": "http",
            "_api_base": api_base,
            "_api_key": api_key,
            "_is_gemini": bool(is_gemini),
            "_api_version": api_version,
        }

    if use_azure:
        from openai import AzureOpenAI

        key = api_key or os.getenv("AZURE_GPT4O_API_KEY_02") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        client = AzureOpenAI(api_key=key, api_version=api_version or "2024-02-01", azure_endpoint=api_base)
        client._is_gemini = False
        return client

    if is_gemini:
        from openai import OpenAI

        gcloud_path = os.getenv("GCLOUD_PATH") or "/home/moezhu/proactive_agent/google-cloud-sdk/bin/gcloud"

        def refresh_token() -> str:
            try:
                out = subprocess.run(
                    [gcloud_path, "auth", "print-access-token"], capture_output=True, text=True, check=True
                )
                return out.stdout.strip()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"gcloud 获取 token 失败: {e}")

        token = refresh_token()
        client = OpenAI(api_key=token, base_url=api_base)
        client._refresh_token_fn = refresh_token
        client._api_base = api_base
        client._is_gemini = True
        return client

    from openai import OpenAI

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or ""
    client = OpenAI(api_key=key, base_url=api_base)
    client._is_gemini = False
    return client


def _call_chat(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: Optional[float] = None,
) -> str:
    def _maybe_adjust_max_tokens_from_ctx_error(msg: str, requested: int) -> Optional[int]:
        """
        Parse typical OpenAI-compatible context-length errors and return a smaller max_tokens.
        Example msg:
          "... maximum context length is 8192 tokens and your request has 7358 input tokens (1024 > 8192 - 7358)."
        """
        if not msg:
            return None
        import re

        m = re.search(r"maximum context length is\s+(\d+)\s+tokens.*request has\s+(\d+)\s+input tokens", msg, re.IGNORECASE)
        if not m:
            return None
        try:
            max_ctx = int(m.group(1))
            in_tok = int(m.group(2))
        except Exception:
            return None
        # Leave a small safety margin for formatting / tokenizer mismatch.
        budget = max_ctx - in_tok - 16
        if budget <= 0:
            return 1
        return min(int(requested), int(budget))

    # HTTP transport: OpenAI-compatible /chat/completions
    if isinstance(client, dict) and client.get("_transport") == "http":
        api_base = str(client.get("_api_base") or "").rstrip("/")
        api_key = str(client.get("_api_key") or "")
        url = f"{api_base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        j = _http_post_json(url=url, api_key=api_key, payload=payload)
        if "error" in j:
            msg = str(j.get("error", {}).get("message", j.get("error")) or "")
            code = j.get("error", {}).get("code")
            # Retry once with a smaller max_tokens if we hit context budget error.
            if int(code or 0) == 400:
                new_max = _maybe_adjust_max_tokens_from_ctx_error(msg, int(payload.get("max_tokens", max_tokens)))
                if new_max is not None and new_max < int(payload.get("max_tokens", max_tokens)):
                    payload["max_tokens"] = int(new_max)
                    j2 = _http_post_json(url=url, api_key=api_key, payload=payload)
                    if "error" not in j2:
                        try:
                            return str(j2["choices"][0]["message"]["content"] or "")
                        except Exception:
                            return json.dumps(j2, ensure_ascii=False)
                    msg2 = str(j2.get("error", {}).get("message", j2.get("error")) or "")
                    code2 = j2.get("error", {}).get("code")
                    raise RuntimeError(f"HTTP chat/completions failed after max_tokens retry (code={code2}): {msg2}")
            raise RuntimeError(f"HTTP chat/completions failed (code={code}): {msg}")
        try:
            return str(j["choices"][0]["message"]["content"] or "")
        except Exception:
            return json.dumps(j, ensure_ascii=False)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    # GPT-5.x / o-series (Azure) reject max_tokens (need max_completion_tokens)
    # and reject non-default temperature/top_p/presence_penalty entirely.
    model_l = str(model or "").lower()
    is_strict_reasoning_model = (
        model_l.startswith("gpt-5")
        or model_l.startswith("o1")
        or model_l.startswith("o3")
        or model_l.startswith("o4")
    )
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if is_strict_reasoning_model:
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
        params["temperature"] = temperature
        params["top_p"] = top_p
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
    resp = None
    last_err = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**params)
            # time.sleep(3)
            break
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # 400 / context budget: reduce max_tokens and retry once.
            if "400" in err_str and ("maximum context length" in err_str or "max_tokens" in err_str or "max_completion_tokens" in err_str):
                new_max = _maybe_adjust_max_tokens_from_ctx_error(str(e), int(params.get("max_tokens", max_tokens)))
                if new_max is not None and new_max < int(params.get("max_tokens", max_tokens)):
                    params["max_tokens"] = int(new_max)
                    print(f"[retry] context budget error, reducing max_tokens -> {new_max}")
                    time.sleep(0.2)
                    continue
            # 401: 尝试刷新 token（Azure/Gemini 会走这里）
            if ("401" in err_str or "unauthorized" in err_str) and hasattr(client, "_refresh_token_fn"):
                print("[retry] 401 detected, refreshing token...")
                try:
                    new_token = client._refresh_token_fn()
                    from openai import OpenAI

                    client = OpenAI(api_key=new_token, base_url=getattr(client, "_api_base", None))
                    resp = client.chat.completions.create(**params)
                    # time.sleep(3)
                    break
                except Exception as e2:
                    last_err = e2
                    continue
            # 429 / rate limit: 退避重试（从 3s 起递增）
            if "429" in err_str or "rate limit" in err_str or "ratelimit" in err_str:
                wait = 1 * (attempt + 1)
                print(f"[retry] 429 detected, sleep {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                continue
            # 其他错误直接抛出
            raise e
    if resp is None:
        raise last_err if last_err else RuntimeError("chat.completions failed with unknown error")
    # 兼容不同 SDK / 兼容端点的返回结构
    # 优先使用 resp.choices[0].message.content；若不存在，尝试字典形式；若依然失败则退化为 str(resp)
    try:
        choice = resp.choices[0]
        return (choice.message.content or "").strip()
    except Exception:
        if isinstance(resp, dict):
            try:
                ch = resp.get("choices", [{}])[0]
                msg = ch.get("message", {}) if isinstance(ch, dict) else {}
                content = msg.get("content", "")
                if content:
                    return str(content).strip()
            except Exception:
                pass
        try:
            return str(resp).strip()
        except Exception:
            return ""


# -----------------------------
# 主流程
# -----------------------------
def load_states(path: Path, max_states: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line)
            except Exception:
                continue
            out.append(s)
            if max_states is not None and len(out) >= max_states:
                break
    return out


def _states_one_per_video(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only one (earliest-t) state per video_id.

    Our dynamic simulation uses the first state's (completed_steps, human_remaining_gt)
    to initialize per-video state; subsequent per-t states are NOT used to reconstruct
    world state. This option mainly reduces redundant input and avoids confusing progress
    totals that look like "all input states will be used".
    """
    by_vid: Dict[str, List[Dict[str, Any]]] = {}
    for s in states:
        vid = str(s.get("video_id", "")).strip()
        if not vid:
            continue
        by_vid.setdefault(vid, []).append(s)

    out: List[Dict[str, Any]] = []

    def _t(x: Dict[str, Any]) -> int:
        try:
            return int(x.get("t", 0))
        except Exception:
            return 0

    for _, vstates in by_vid.items():
        vstates.sort(key=_t)
        out.append(vstates[0])

    # stable order for reproducibility
    out.sort(key=lambda x: str(x.get("video_id", "")))
    return out


def infer_actions(
    states: List[Dict[str, Any]],
    client,
    model: str,
    out_path: Path,
    immediate_M: int,
    horizon: int,
    candidate_mode: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    presence_penalty: Optional[float],
    annotation_path: Path,
    human_min_entropy: bool = False,
    no_switch: bool = False,
    states_one_per_video: bool = False,
    max_decisions_per_video: Optional[int] = None,
) -> None:
    # 动态仿真式推理（与 planning_eval_new.py 对齐）
    # - 维护 completed / remaining / robot_history / human_history
    # - t 使用动态 t_idx（与 decisions 评测一致）
    # - candidates 支持 future5 与 all-remaining 两种模式
    from task_planner import TaskGraphManager

    graph_cache: Dict[str, TaskGraphManager] = {}

    if states_one_per_video:
        states = _states_one_per_video(states)

    def _get_graph_text(task_name: str) -> str:
        if not task_name:
            return "(no task graph)"
        if task_name not in graph_cache:
            graph_cache[task_name] = TaskGraphManager(str(annotation_path), task_name)
        g = graph_cache[task_name]
        lines = []
        for nid, node in g.id2node.items():
            name = node.get("name", "")
            cond = str(node.get("activation_condition", "") or "").strip()
            # Keep it compact to avoid blowing up the prompt; still preserve AND/OR structure.
            if len(cond) > 160:
                cond = cond[:160] + "...(truncated)"
            parents = node.get("parent_id")
            if parents is None:
                parents = []
            if not isinstance(parents, list):
                parents = [parents]
            pn = [g.id2node.get(str(p), {}).get("name", "") for p in parents]
            leaf = bool(node.get("is_leafnode", False))
            mid = bool(node.get("is_midlevel", False))
            cat = node.get("midlevel_category", "")
            lines.append(f"id:{nid} name:{name} p:{parents} pn:{pn} leaf:{leaf} mid:{mid} cat:{cat} cond:{cond}")
        return "\n".join(lines[:400])  # 截断以避免超长

    # 断点续跑：读取已有输出，跳过已处理的 state
    processed_ids: set[str] = set()
    processed_actions: Dict[str, str] = {}
    # 统计每个视频已处理的 state 数量（用于按视频跳过）
    processed_videos: Dict[str, set[str]] = {}
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        obj = json.loads(line)
                        sid = obj.get("state_id")
                        vid = obj.get("video_id")
                        if sid:
                            processed_ids.add(str(sid))
                            if vid:
                                vid_str = str(vid).strip()
                                if vid_str not in processed_videos:
                                    processed_videos[vid_str] = set()
                                processed_videos[vid_str].add(str(sid))
                            act = str(obj.get("action", "") or "").strip()
                            if act:
                                processed_actions[str(sid)] = act
                    except Exception:
                        continue
            print(f"[resume] 已存在输出 {len(processed_ids)} 条 state，涉及 {len(processed_videos)} 个视频")
        except Exception as e:
            print(f"[resume] 读取已有输出失败，忽略续跑: {e}")

    # 分组
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for s in states:
        vid = str(s.get("video_id", "")).strip()
        if not vid:
            continue
        by_video.setdefault(vid, []).append(s)
    for vid in by_video:
        by_video[vid].sort(key=lambda x: int(x.get("t", 0)))

    def _state_id(video_id: str, t_idx: int) -> str:
        return f"{video_id}:{t_idx}"

    candidate_mode = (candidate_mode or "future5").strip().lower()
    if candidate_mode not in {"future5", "remaining"}:
        raise ValueError(f"invalid candidate_mode: {candidate_mode}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as wf:
        # 全局改为按“视频”进度条，避免状态条显示为 state 数量导致误解
        progress_by_video = True
        pbar = tqdm(total=len(by_video), desc="LLM inference (videos)", leave=True)

        decisions_total = 0
        videos_done = 0

        for vid, vstates in by_video.items():
            # 检查该视频是否已完全处理：如果该视频的所有 state 都已处理，跳过整个视频
            if vid in processed_videos:
                # 检查该视频是否还有未处理的 state
                # 方法：模拟运行一次，检查是否有任何 state 不在 processed_ids 中
                base = vstates[0]
                task_name = str(base.get("task_name", "")).strip()
                if not task_name:
                    continue
                if task_name not in graph_cache:
                    graph_cache[task_name] = TaskGraphManager(str(annotation_path), task_name)
                graph = graph_cache[task_name]
                
                completed: List[str] = list(base.get("completed_steps", []))
                remaining: List[str] = list(base.get("human_remaining_gt", []))
                t_idx = len(completed)
                
                # 快速检查：遍历该视频所有可能的 state，看是否都已处理
                video_fully_processed = True
                max_check_ticks = min(max(len(remaining) * 6, 100), 200)  # 限制检查范围
                check_ticks = 0
                check_completed = list(completed)
                check_remaining = list(remaining)
                check_t_idx = t_idx
                
                while check_remaining and check_ticks < max_check_ticks:
                    while check_remaining and check_remaining[0] in check_completed:
                        check_remaining.pop(0)
                        check_t_idx += 1
                    if not check_remaining:
                        break
                    
                    check_sid = _state_id(vid, check_t_idx)
                    if check_sid not in processed_ids:
                        video_fully_processed = False
                        break
                    
                    # 模拟推进一步
                    if check_remaining:
                        head = check_remaining[0]
                        if head not in check_completed:
                            check_completed.append(head)
                        while check_remaining and check_remaining[0] in check_completed:
                            check_remaining.pop(0)
                            check_t_idx += 1
                    check_ticks += 1
                
                if video_fully_processed:
                    # 该视频已完全处理，跳过
                    videos_done += 1
                    pbar.set_postfix_str(f"skip={vid} (fully processed) videos_done={videos_done}")
                    pbar.update(1)
                    print(f"[video_skip] {vid} (已完全处理，跳过)")
                    continue
            
            decisions_this_video = 0
            base = vstates[0]
            task_name = str(base.get("task_name", "")).strip()
            if not task_name:
                continue
            if task_name not in graph_cache:
                graph_cache[task_name] = TaskGraphManager(str(annotation_path), task_name)
            graph = graph_cache[task_name]

            completed: List[str] = list(base.get("completed_steps", []))
            remaining: List[str] = list(base.get("human_remaining_gt", []))
            human_hist: List[str] = list(completed)
            robot_hist: List[str] = []
            # t_idx 与 planning_eval_new.py 保持一致：仅在"从 remaining 头部移除完成步骤"时递增
            t_idx = len(completed)

            max_ticks = max(len(remaining) * 6, 100)
            ticks = 0
            while remaining and ticks < max_ticks:
                if isinstance(max_decisions_per_video, int) and max_decisions_per_video > 0:
                    if decisions_this_video >= max_decisions_per_video:
                        break
                while remaining and remaining[0] in completed:
                    remaining.pop(0)
                    t_idx += 1
                if not remaining:
                    break

                sid = _state_id(vid, t_idx)
                if sid in processed_ids:
                    pbar.set_postfix_str(
                        f"cur={vid}:{t_idx} decisions_this_video={decisions_this_video} decisions_total={decisions_total}"
                    )
                    # 续跑：先复现已保存的机器人动作，再推进人类一步（保证状态一致）
                    prev_action_eff = processed_actions.get(sid, "Wait / None")
                    in_progress_eff: Optional[str] = None
                    if prev_action_eff and prev_action_eff != "Wait / None":
                        node = graph.name2node.get(prev_action_eff)
                        cond = node.get("activation_condition", "TRUE") if node else "TRUE"
                        try:
                            cond_ok = graph.check_condition(cond, completed)
                        except Exception:
                            cond_ok = False
                        # NOTE: Robot does NOT force-add human_next to legal set.
                        # Only check if condition is met; if not, robot must wait.
                        if cond_ok:
                            in_progress_eff = str(prev_action_eff)
                            robot_hist.append(prev_action_eff)
                            if prev_action_eff not in completed:
                                completed.append(prev_action_eff)
                            if prev_action_eff in remaining:
                                try:
                                    remaining.remove(prev_action_eff)
                                except ValueError:
                                    pass

                    # 若机器人提前完成了 head，先弹出 head（推进 t_idx）
                    while remaining and remaining[0] in completed:
                        remaining.pop(0)
                        t_idx += 1

                    # human acts (align with planning_eval.py's human mode)
                    _human_choose_and_apply(
                        graph=graph,
                        completed=completed,
                        remaining=remaining,
                        human_hist=human_hist,
                        robot_hist=robot_hist,
                        in_progress=in_progress_eff,
                        human_min_entropy=bool(human_min_entropy),
                        no_switch=bool(no_switch),
                    )
                    # pop head-completed steps, advance t_idx
                    while remaining and remaining[0] in completed:
                        remaining.pop(0)
                        t_idx += 1

                    ticks += 1
                    continue

                future_window = remaining[: max(1, horizon)] if horizon > 0 else []
                human_immediate = future_window[:immediate_M]
                # Calculate legal robot actions (robot does NOT force-add human_next to legal set)
                legal_raw = graph.get_legal_robot_actions(completed, human_immediate)
                legal = [a for a in (legal_raw or []) if _is_real_action_node(graph, a)]
                # candidates: predicted_src ∩ legal (same as planning_eval.py)
                predicted_src0 = remaining if candidate_mode == "remaining" else future_window
                predicted_src = [a for a in (predicted_src0 or []) if _is_real_action_node(graph, a)]
                legal_set = set(legal)
                candidates = [a for a in predicted_src if a in legal_set]

                if not candidates:
                    rec = {
                        "state_id": sid,
                        "video_id": vid,
                        "t": t_idx,
                        "task_name": task_name,
                        "model": model,
                        "candidate_mode": candidate_mode,
                        "horizon": horizon,
                        "action": "Wait / None",
                        "reason": "no candidates",
                        "confidence": None,
                        "candidate_actions": candidates,
                        "raw_response": "",
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wf.flush()
                    processed_ids.add(sid)
                    decisions_this_video += 1
                    decisions_total += 1
                    pbar.set_postfix_str(
                        f"cur={vid}:{t_idx} decisions_this_video={decisions_this_video} decisions_total={decisions_total}"
                    )

                    _human_choose_and_apply(
                        graph=graph,
                        completed=completed,
                        remaining=remaining,
                        human_hist=human_hist,
                        robot_hist=robot_hist,
                        in_progress=None,
                        human_min_entropy=bool(human_min_entropy),
                        no_switch=bool(no_switch),
                    )
                    while remaining and remaining[0] in completed:
                        remaining.pop(0)
                        t_idx += 1

                    ticks += 1
                    continue

                task_graph_text = _get_graph_text(task_name)
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    task_name=task_name,
                    task_graph=task_graph_text,
                    completed=", ".join(completed) if completed else "(none)",
                    human_immediate=", ".join(human_immediate) if human_immediate else "(none)",
                    human_future=", ".join(future_window) if future_window else "(none)",
                    candidates=", ".join(candidates),
                )

                # 网络/服务端偶发错误：重试几次；仍失败则写入一个占位决策并继续，避免整轮任务被中断
                raw = ""
                last_err: str | None = None
                for attempt in range(1, 4):
                    try:
                        raw = _call_chat(
                            client=client,
                            model=model,
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=user_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                        )
                        if raw and raw.strip():
                            break
                    except Exception as e:
                        last_err = str(e)
                        print(f"[ERROR] video={vid}, t={t_idx}, attempt={attempt}/3, err={e}")
                        time.sleep(min(10, 2 * attempt))

                if not raw.strip():
                    # 记录一个可恢复的占位输出（action=Wait / None），并继续推进
                    reason = f"llm_connection_error: {last_err}" if last_err else "llm_empty_response"
                    rec = {
                        "state_id": sid,
                        "video_id": vid,
                        "t": t_idx,
                        "task_name": task_name,
                        "model": model,
                        "candidate_mode": candidate_mode,
                        "horizon": horizon,
                        "action": "Wait / None",
                        "reason": reason,
                        "confidence": None,
                        "candidate_actions": candidates,
                        "raw_response": "",
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wf.flush()
                    processed_ids.add(sid)
                    decisions_this_video += 1
                    decisions_total += 1
                    pbar.set_postfix_str(
                        f"cur={vid}:{t_idx} decisions_this_video={decisions_this_video} decisions_total={decisions_total}"
                    )

                    _human_choose_and_apply(
                        graph=graph,
                        completed=completed,
                        remaining=remaining,
                        human_hist=human_hist,
                        robot_hist=robot_hist,
                        in_progress=None,
                        human_min_entropy=bool(human_min_entropy),
                        no_switch=bool(no_switch),
                    )
                    while remaining and remaining[0] in completed:
                        remaining.pop(0)
                        t_idx += 1

                    ticks += 1
                    continue

                action, reason, conf_val = _extract_action_from_raw(raw, candidates)

                # prereq check: if not executable, robot must wait (and this wait is what we record as action)
                # NOTE: Robot does NOT force-add human_next to legal set.
                # If action is not in legal set (or condition not met), robot must wait.
                if action != "Wait / None":
                    node = graph.name2node.get(action)
                    cond = node.get("activation_condition", "TRUE") if node else "TRUE"
                    try:
                        cond_ok = graph.check_condition(cond, completed)
                    except Exception:
                        cond_ok = False
                    # Check if action is in legal set (candidates are already filtered by legal, but double-check)
                    # Since candidates = predicted_src ∩ legal_set, if action is in candidates, it's in legal_set
                    # But we still check condition explicitly
                    if not cond_ok:
                        reason = (reason + " | " if reason else "") + "prereq_not_met"
                        action = "Wait / None"

                rec = {
                    "state_id": sid,
                    "video_id": vid,
                    "t": t_idx,
                    "task_name": task_name,
                    "model": model,
                    "candidate_mode": candidate_mode,
                    "horizon": horizon,
                    "action": action,
                    "reason": reason,
                    "confidence": conf_val,
                    "candidate_actions": candidates,
                    "raw_response": raw,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wf.flush()
                processed_ids.add(sid)
                decisions_this_video += 1
                decisions_total += 1
                pbar.set_postfix_str(
                    f"cur={vid}:{t_idx} decisions_this_video={decisions_this_video} decisions_total={decisions_total}"
                )

                # apply robot action if any
                if action != "Wait / None":
                    robot_hist.append(action)
                    if action not in completed:
                        completed.append(action)
                    if action in remaining:
                        try:
                            remaining.remove(action)
                        except ValueError:
                            pass

                # if robot completed head, pop it first
                while remaining and remaining[0] in completed:
                    remaining.pop(0)
                    t_idx += 1

                # human acts (align with planning_eval.py's human mode)
                _human_choose_and_apply(
                    graph=graph,
                    completed=completed,
                    remaining=remaining,
                    human_hist=human_hist,
                    robot_hist=robot_hist,
                    in_progress=(None if action == "Wait / None" else str(action)),
                    human_min_entropy=bool(human_min_entropy),
                    no_switch=bool(no_switch),
                )
                while remaining and remaining[0] in completed:
                    remaining.pop(0)
                    t_idx += 1

                ticks += 1

            videos_done += 1
            pbar.set_postfix_str(f"done={vid} decisions_this_video={decisions_this_video} videos_done={videos_done}")
            pbar.update(1)
            # lightweight proof: print one line per finished video (helps users verify +1 is per video)
            print(f"[video_done] {vid} decisions={decisions_this_video}")
        pbar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 生成机器人动作 (planning_states)")
    parser.add_argument(
        "--states",
        type=Path,
        default=Path("/home/moezhu/proactive_agent/test/planning/planning_states.jsonl"),
        help="planning_states.jsonl 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/moezhu/proactive_agent/test/planning/llm_actions.jsonl"),
        help="输出 JSONL 路径",
    )
    parser.add_argument("--model", type=str, required=True, help="模型名称，如 gpt-4o / google/gemini-2.5-flash")
    parser.add_argument("--api_base", type=str, default=os.getenv("API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", ""))
    parser.add_argument("--api_version", type=str, default=os.getenv("API_VERSION", "2024-02-01"))
    parser.add_argument(
        "--transport",
        type=str,
        default=os.getenv("PLANNING_API_TRANSPORT", "sdk"),
        choices=["sdk", "http"],
        help="API 调用方式：sdk=openai SDK（默认）；http=直接 HTTP 请求 /v1/chat/completions",
    )
    parser.add_argument("--immediate_M", type=int, default=1, help="禁止抢活的前 M 步")
    parser.add_argument("--horizon", type=int, default=5, help="候选窗口大小（candidate_mode=future5 时生效）")
    parser.add_argument(
        "--candidate_mode",
        type=str,
        default="future5",
        choices=["future5", "remaining"],
        help="候选集模式：future5=仅未来窗口；remaining=所有剩余步骤",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--max_states", type=int, default=None, help="仅处理前 N 条状态（调试）")
    parser.add_argument(
        "--max_decisions_per_video",
        type=int,
        default=None,
        help="调试：每个视频最多调用 LLM 决策次数（不改变默认行为；None 表示不限）",
    )
    parser.add_argument(
        "--states_one_per_video",
        action="store_true",
        help="只保留每个 video_id 最早的一条 state 用于初始化（动态仿真仍会按 tick 推进并请求 LLM）",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("/home/moezhu/video_annotation_project/data/annotations/all_annotations.json"),
        help="taxonomy/annotation 路径（供 LLM 提供简化 task graph）",
    )
    # --- Compatibility flags (accepted but ignored) ---
    # Some repo scripts historically pass these flags to the inference script.
    # They are evaluation-only controls in planning_eval.py, but we accept them here
    # to avoid "unrecognized arguments" failures.
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="兼容参数（忽略）：旧脚本的 prompt mode 开关；推理脚本当前不使用该参数。",
    )
    parser.add_argument(
        "--human_min_entropy",
        action="store_true",
        help="兼容参数（忽略）：评测阶段的人类模式开关；仅 planning_eval.py 使用。",
    )
    parser.add_argument(
        "--no_switch",
        action="store_true",
        help="兼容参数（忽略）：评测阶段的人类模式开关；仅 planning_eval.py 使用。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = _build_client(
        api_base=args.api_base,
        api_key=args.api_key,
        api_version=args.api_version,
        model=args.model,
        transport=getattr(args, "transport", "sdk"),
    )
    states = load_states(args.states, max_states=args.max_states)
    print(f"[load] {len(states)} states from {args.states}")
    if bool(getattr(args, "states_one_per_video", False)):
        # show post-compression count to avoid confusion
        try:
            states = _states_one_per_video(states)
            print(f"[states_one_per_video] keep {len(states)} videos (1 state per video) for initialization")
        except Exception as e:
            print(f"[states_one_per_video] failed to compress states: {e}")
    infer_actions(
        states=states,
        client=client,
        model=args.model,
        out_path=args.output,
        immediate_M=args.immediate_M,
        horizon=args.horizon,
        candidate_mode=args.candidate_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        annotation_path=args.annotation,
        human_min_entropy=bool(getattr(args, "human_min_entropy", False)),
        no_switch=bool(getattr(args, "no_switch", False)),
        # already compressed above if enabled; keep this false to avoid double work
        states_one_per_video=False,
        max_decisions_per_video=getattr(args, "max_decisions_per_video", None),
    )
    print(f"[done] 写入 {args.output}")


if __name__ == "__main__":
    main()

