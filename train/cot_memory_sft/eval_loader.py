"""DLV2 unified checkpoint loader for dedicated E-w2 eval.

Single load path — never double-load LoRA (default + stage) or skip
modules_to_save sync.  See refine-logs/EVAL_CONTRACT.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from train.cot_sft_v2.train_cot_sft_two_stage import _get_env_silence_and_rank


def unwrap_model(model: Any) -> Any:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


def _canonicalise_peft_key(key: str) -> str:
    """Strip PEFT adapter-name segments for cross-version key matching."""
    k = str(key or "")
    for slot in ("default", "w1", "w2"):
        k = k.replace(f".modules_to_save.{slot}.", ".")
    k = k.replace(".modules_to_save.", ".")
    for slot in ("default", "w1", "w2"):
        k = k.replace(f".lora_A.{slot}.", ".lora_A.")
        k = k.replace(f".lora_B.{slot}.", ".lora_B.")
    return k


def _is_bare_head_alias(key: str) -> bool:
    """PEFT may emit bare embed/lm_head .weight keys with no modules_to_save slot.

    These never exist in the live model state dict (only .modules_to_save.* /
    .original_module.* do).  Head tensors are loaded via load_modules_to_save_for_eval.
    """
    k = str(key or "")
    if ".modules_to_save." in k or ".original_module." in k:
        return False
    return k.endswith(".lm_head.weight") or k.endswith(".embed_tokens.weight")


def remap_ckpt_state_for_stage(
    raw_state: Dict[str, torch.Tensor], stage_name: str
) -> Dict[str, torch.Tensor]:
    """PEFT dual-adapter saves undecorated lora_A/B keys; remap to .lora_A.{stage}."""
    out: Dict[str, torch.Tensor] = {}
    stage = str(stage_name or "").strip()
    if not stage:
        return dict(raw_state)
    for key, val in raw_state.items():
        if _is_bare_head_alias(key):
            continue
        nk = key
        if ".lora_A." in key and f".lora_A.{stage}." not in key:
            if ".lora_A.default." not in key and ".lora_A.w1." not in key and ".lora_A.w2." not in key:
                nk = key.replace(".lora_A.", f".lora_A.{stage}.", 1)
        elif ".lora_B." in key and f".lora_B.{stage}." not in key:
            if ".lora_B.default." not in key and ".lora_B.w1." not in key and ".lora_B.w2." not in key:
                nk = key.replace(".lora_B.", f".lora_B.{stage}.", 1)
        elif f".modules_to_save.default." in key and stage != "default":
            # Raw-init w2 saves may store trained embed/lm_head under default slot.
            nk = key.replace(".modules_to_save.default.", f".modules_to_save.{stage}.", 1)
        elif ".modules_to_save." in key and f".modules_to_save.{stage}." not in key:
            if ".modules_to_save.default." not in key and ".modules_to_save.w1." not in key:
                nk = key.replace(".modules_to_save.", f".modules_to_save.{stage}.", 1)
        out[nk] = val
    return out


def _ckpt_head_priority(key: str, stage: str) -> int:
    """Lower = preferred when multiple ckpt keys map to the same head tensor."""
    if f".modules_to_save.{stage}." in key:
        return 0
    if ".modules_to_save.default." in key:
        return 1
    if ".modules_to_save." in key:
        return 2
    if ".original_module." in key:
        return 3
    return 4


def _copy_tensor_rows(
    model_tensor: torch.Tensor, ckpt_tensor: torch.Tensor
) -> Tuple[bool, Optional[int]]:
    """Shape-checked copy with partial row heal for resized vocab."""
    if model_tensor.shape == ckpt_tensor.shape:
        with torch.no_grad():
            model_tensor.copy_(ckpt_tensor.to(model_tensor.dtype))
        return True, None
    if (
        len(model_tensor.shape) == 2
        and len(ckpt_tensor.shape) == 2
        and model_tensor.shape[1] == ckpt_tensor.shape[1]
    ):
        n_rows = min(model_tensor.shape[0], ckpt_tensor.shape[0])
        with torch.no_grad():
            model_tensor[:n_rows].copy_(ckpt_tensor[:n_rows].to(model_tensor.dtype))
        return True, n_rows
    return False, None


def _max_lora_b_magnitude(path: Path) -> float:
    """Cheap signal of how trained an adapter file is: max mean|lora_B| over keys.

    ``lora_B`` is zero-initialised; a trained adapter has nonzero ``lora_B`` while
    an untrained (raw-init ``default``) adapter is exactly zero.  Reads only the
    small ``lora_B`` slices via safetensors lazy access.
    """
    from safetensors import safe_open

    best = 0.0
    try:
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if ".lora_B" in k:
                    v = f.get_tensor(k).float().abs().mean().item()
                    if v > best:
                        best = v
    except Exception:
        return 0.0
    return best


def resolve_stage_adapter_file(ckpt_dir: str, stage_name: str) -> Path:
    """Return the adapter file whose LoRA was actually trained for this stage.

    PEFT ``save_pretrained`` on a multi-adapter model writes the trained stage
    adapter to ``{ckpt}/{stage}/adapter_model.safetensors`` and the frozen
    ``default`` adapter to ``{ckpt}/adapter_model.safetensors``.  At eval only the
    ``{stage}`` adapter is active (``set_active_adapter_trainable``), so the file
    loaded into that slot must hold the trained weights.

    Which file that is depends on the run:
      * raw-init (E3/HG): root ``default`` is zero-init (untrained ``lora_B``);
        the trained adapter is in the ``{stage}`` subfolder.
      * warm-init (W2k from B2b): the strong trained adapter is the root file;
        the ``{stage}`` subfolder is only a minor delta.

    Pick the file with the larger ``|lora_B|`` so the active slot gets the
    actually-trained weights in BOTH cases (matches legacy root behaviour for
    warm-start, fixes raw-init).  Falls back to root for legacy single-adapter saves.
    """
    stage = str(stage_name or "").strip()
    root = Path(ckpt_dir) / "adapter_model.safetensors"
    if not stage or stage == "default":
        return root
    sub = Path(ckpt_dir) / stage / "adapter_model.safetensors"
    if not sub.exists():
        return root
    if not root.exists():
        return sub
    return sub if _max_lora_b_magnitude(sub) >= _max_lora_b_magnitude(root) else root


def _adapter_file_candidates(ckpt_dir: str, stage_name: str) -> List[Path]:
    """LoRA primary file plus root/subfolder adapters for head weight merge."""
    stage = str(stage_name or "w2").strip()
    ckpt = Path(ckpt_dir)
    primary = resolve_stage_adapter_file(ckpt_dir, stage)
    out: List[Path] = []
    for p in (primary, ckpt / "adapter_model.safetensors", ckpt / stage / "adapter_model.safetensors"):
        if p.exists() and p not in out:
            out.append(p)
    return out


def load_modules_to_save_for_eval(
    model: Any,
    ckpt_dir: str,
    stage_name: str = "w2",
) -> Dict[str, Any]:
    """Load embed_tokens/lm_head via canonical key match (fixes unmapped=2 on raw-init).

    PEFT checkpoints may store head weights as ``modules_to_save.{stage}``,
    ``modules_to_save.default``, undecorated ``modules_to_save``, or bare
    ``original_module`` keys.  Raw-init saves may put trained heads in the
    root ``modules_to_save.{stage}`` while the ``{stage}/`` subfolder uses
    ``modules_to_save.default`` — scan all adapter files and pick best priority.
    """
    stage = str(stage_name or "w2").strip()
    paths = _adapter_file_candidates(ckpt_dir, stage)
    if not paths:
        raise FileNotFoundError(f"Adapter weights not found under {ckpt_dir}")

    target = unwrap_model(model)
    target_sd = target.state_dict()

    stage_mid = f".modules_to_save.{stage}."
    default_mid = ".modules_to_save.default."
    canon_to_stage: Dict[str, str] = {}
    canon_to_default: Dict[str, str] = {}
    for mk in target_sd:
        if "embed_tokens" not in mk and "lm_head" not in mk:
            continue
        if ".modules_to_save." not in mk:
            continue
        canon = _canonicalise_peft_key(mk)
        if stage_mid in mk:
            canon_to_stage.setdefault(canon, mk)
        if default_mid in mk:
            canon_to_default.setdefault(canon, mk)

    best: Dict[str, Tuple[int, str, torch.Tensor]] = {}
    for path in paths:
        raw_state = load_file(str(path))
        for ck_key, cv in raw_state.items():
            # A1 patch: allow bare v1 head keys (e.g. base_model.model.lm_head.weight)
            # so v1-native W1 heads can fill the stage modules_to_save slot.
            if "embed_tokens" not in ck_key and "lm_head" not in ck_key:
                continue
            canon = _canonicalise_peft_key(ck_key)
            if "embed_tokens" not in canon and "lm_head" not in canon:
                continue
            pri = _ckpt_head_priority(ck_key, stage)
            if canon not in best or pri < best[canon][0]:
                best[canon] = (pri, ck_key, cv)

    loaded_stage = 0
    loaded_default = 0
    partial_healed: List[str] = []
    head_sources: List[str] = []
    for canon, (_, ck_key, cv) in best.items():
        for slot, slot_map in (("stage", canon_to_stage), ("default", canon_to_default)):
            mk = slot_map.get(canon)
            if mk is None:
                continue
            ok, n_rows = _copy_tensor_rows(target_sd[mk], cv)
            if not ok:
                continue
            if slot == "stage":
                loaded_stage += 1
            else:
                loaded_default += 1
            head_sources.append(ck_key)
            if n_rows is not None:
                partial_healed.append(f"{ck_key}:{n_rows}rows")

    report = {
        "head_keys_loaded_stage": loaded_stage,
        "head_keys_loaded_default": loaded_default,
        "partial_healed": partial_healed,
        "head_ckpt_keys": sorted(set(head_sources)),
    }
    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main and (loaded_stage or loaded_default):
        print(
            f"[eval_loader] modules_to_save canonical load stage={stage} "
            f"stage={loaded_stage} default={loaded_default} "
            f"partial={len(partial_healed)} sources={len(paths)}",
            flush=True,
        )
    return report


def sync_modules_to_save_default_from_stage(model: Any, stage_name: str) -> None:
    """Copy modules_to_save.{stage} → modules_to_save.default for embed/lm_head."""
    stage = str(stage_name or "").strip()
    if not stage or stage == "default":
        return
    target = unwrap_model(model)
    sd = target.state_dict()
    copied: List[str] = []
    for key, val in sd.items():
        stage_mid = f".modules_to_save.{stage}."
        if stage_mid not in key:
            continue
        default_key = key.replace(stage_mid, ".modules_to_save.default.", 1)
        if default_key not in sd or sd[default_key].shape != val.shape:
            continue
        with torch.no_grad():
            sd[default_key].copy_(val.to(sd[default_key].dtype))
        copied.append(default_key.rsplit(".", 2)[-2])
    silence, is_main, _, _ = _get_env_silence_and_rank()
    if copied and not silence and is_main:
        print(
            f"[eval_loader] synced modules_to_save.{stage} → default "
            f"for {sorted(set(copied))}",
            flush=True,
        )


def _build_canon_to_model(target_sd: Dict[str, torch.Tensor]) -> Dict[str, str]:
    canon_to_model: Dict[str, str] = {}
    for mk in target_sd:
        canon_to_model.setdefault(_canonicalise_peft_key(mk), mk)
    return canon_to_model


def load_stage_adapter_weights(
    model: Any,
    ckpt_dir: str,
    stage_name: str,
    *,
    permissive: bool = True,
) -> int:
    """Load adapter_model.safetensors into a named PEFT stage (no default adapter).

    Prefers the per-stage subfolder (``{ckpt}/{stage}/``) which holds the
    actually-trained stage adapter; root holds the frozen ``default`` adapter
    (untrained / zero ``lora_B`` for raw-init).  See ``resolve_stage_adapter_file``.
    """
    path = resolve_stage_adapter_file(ckpt_dir, stage_name)
    if not path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {path}")
    raw_state = load_file(str(path))
    remapped = remap_ckpt_state_for_stage(raw_state, stage_name)
    target = unwrap_model(model)
    target_sd = target.state_dict()
    canon_to_model = _build_canon_to_model(target_sd)
    loaded, unmapped = 0, []
    for ck, cv in remapped.items():
        mk = ck if ck in target_sd else canon_to_model.get(_canonicalise_peft_key(ck))
        if mk is None:
            unmapped.append(ck)
            continue
        if target_sd[mk].shape != cv.shape:
            unmapped.append(ck)
            continue
        with torch.no_grad():
            target_sd[mk].copy_(cv.to(target_sd[mk].dtype))
        loaded += 1
    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main:
        print(
            f"[eval_loader] stage={stage_name} from {path} "
            f"loaded={loaded}/{len(remapped)} unmapped={len(unmapped)}",
            flush=True,
        )
        if unmapped:
            print(
                f"[eval_loader] unmapped keys: {unmapped}",
                flush=True,
            )
    if unmapped and not permissive:
        raise RuntimeError(
            f"load_stage_adapter_weights unmapped={len(unmapped)} stage={stage_name} "
            f"keys={unmapped[:5]}"
        )
    return loaded


def _active_adapter_names(peft_model: Any) -> List[str]:
    active = getattr(peft_model, "active_adapters", None)
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    try:
        return [str(x) for x in active]
    except TypeError:
        return [str(active)]


def _pick_language_lora_b_key(keys: List[str]) -> Optional[str]:
    """Prefer language-model LoRA keys over visual-tower keys for sanity checks."""
    lang = [
        k
        for k in keys
        if ".lora_B" in k and ".language_model." in k and ".visual." not in k
    ]
    if lang:
        return sorted(lang)[0]
    lora = [k for k in keys if ".lora_B" in k]
    return sorted(lora)[0] if lora else None


def verify_adapter_weights_match(
    model: Any,
    ckpt_path: str,
    stage: str = "w2",
    *,
    atol: float = 1e-5,
    rel_tol: float = 0.02,
) -> Dict[str, Any]:
    """Compare a language-model lora_B tensor between checkpoint and loaded model."""
    stage = str(stage or "w2").strip()
    path = resolve_stage_adapter_file(ckpt_path, stage)
    if not path.exists():
        return {"ok": False, "error": f"missing {path}"}

    raw_state = load_file(str(path))
    remapped = remap_ckpt_state_for_stage(raw_state, stage)
    ck_key = _pick_language_lora_b_key(list(remapped.keys()))
    if ck_key is None:
        return {"ok": False, "error": "no lora_B in checkpoint", "adapter_file": str(path)}

    ckpt_norm = float(remapped[ck_key].float().abs().mean().item())
    target_sd = unwrap_model(model).state_dict()
    canon_to_model = _build_canon_to_model(target_sd)
    model_key = ck_key if ck_key in target_sd else canon_to_model.get(
        _canonicalise_peft_key(ck_key)
    )
    model_norm: Optional[float] = None
    if model_key and model_key in target_sd:
        model_norm = float(target_sd[model_key].float().abs().mean().item())

    if model_norm is None:
        ok = False
    elif ckpt_norm == 0.0:
        ok = model_norm == 0.0
    else:
        ok = abs(model_norm - ckpt_norm) <= max(atol, rel_tol * ckpt_norm)

    return {
        "ok": ok,
        "ckpt_lora_b_mean": ckpt_norm,
        "model_lora_b_mean": model_norm,
        "ckpt_key": ck_key,
        "model_key": model_key,
        "adapter_file": str(path),
    }


def activate_eval_adapters(model: Any, stage: str = "w2") -> None:
    """Activate PEFT adapters for dedicated eval on unwrapped model (not DDP shell).

    After disk reload, w2 must be re-activated on the unwrapped PEFT model.
    Calling ``set_adapter`` on the DDP wrapper deactivates w2 → empty phase-2.

    Match training / inline on_save forward: ``set_adapter([default, stage])``
    with ``modules_to_save.{stage}`` synced into ``default`` (see
    ``sync_modules_to_save_default_from_stage``).  Fall back to stage-only for
    checkpoints that reject multi-adapter activation.
    """
    stage = str(stage or "w2").strip()
    peft_model = unwrap_model(model)
    chosen: Optional[List[str]] = None
    for candidate in (["default", stage], [stage, "default"], [stage]):
        try:
            peft_model.set_adapter(candidate if len(candidate) > 1 else candidate[0])
            chosen = candidate
            break
        except Exception:
            continue
    active = _active_adapter_names(peft_model)
    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not silence and is_main:
        print(
            f"[eval_loader] activate_eval_adapters unwrapped active={active!r} "
            f"requested={chosen!r} stage={stage}",
            flush=True,
        )


def load_checkpoint_for_eval(
    model: Any,
    ckpt_path: str,
    *,
    stage: str = "w2",
    permissive: bool = True,
) -> Dict[str, Any]:
    """ONE eval load path: base Qwen already wrapped in PEFT.

    Rules (invariant):
      - NEVER call legacy _load_adapter_weights(eval_ckpt) before this for w2 eval.
      - Optional INIT_CKPT (8540 warm-start) must be loaded in main() into default only.
      - Stage weights load into .lora_*.{stage}. slots only.
      - modules_to_save.{stage} copied to .default before generation.
      - Adapters re-activated via unwrapped set_adapter([default, stage]).

    Returns a small report dict for logging.
    """
    ckpt = str(ckpt_path or "").strip()
    if not ckpt:
        raise ValueError("load_checkpoint_for_eval: ckpt_path required")
    stage = str(stage or "w2").strip()

    report: Dict[str, Any] = {
        "ckpt": ckpt,
        "stage": stage,
    }

    n_loaded = load_stage_adapter_weights(model, ckpt, stage, permissive=permissive)
    report["stage_keys_loaded"] = n_loaded
    head_report = load_modules_to_save_for_eval(model, ckpt, stage)
    report.update(head_report)
    sync_modules_to_save_default_from_stage(model, stage)
    activate_eval_adapters(model, stage)
    active = _active_adapter_names(unwrap_model(model))
    report["active_adapters"] = active
    weight_check = verify_adapter_weights_match(model, ckpt, stage)
    report["weight_check"] = weight_check
    silence, is_main, _, _ = _get_env_silence_and_rank()
    if not weight_check.get("ok", False):
        if not silence and is_main:
            print(
                f"[eval_loader] WARN: lora_B norm mismatch after load: {weight_check}",
                flush=True,
            )
    elif not silence and is_main:
        print(
            f"[eval_loader] weight_check ok key={weight_check.get('ckpt_key', '')!r} "
            f"norm={weight_check.get('model_lora_b_mean')}",
            flush=True,
        )
    return report


def parse_future_steps(text: str) -> List[str]:
    m = re.search(
        r"<\|future_steps_start\|>(.*?)<\|future_steps_end\|>",
        text or "",
        flags=re.DOTALL,
    )
    if not m:
        return []
    inner = (m.group(1) or "").strip()
    if not inner or inner.lower() == "terminate":
        return []
    return [s.strip() for s in inner.split(";") if s.strip()]


def parse_next_action(text: str) -> str:
    m = re.search(
        r"<\|next_action_start\|>(.*?)<\|next_action_end\|>",
        text or "",
        flags=re.DOTALL,
    )
    return (m.group(1) or "").strip() if m else ""


def verify_pred_jsonl_smoke(
    pred_path: str | Path,
    *,
    min_trigger_with_future: int = 1,
    min_trigger_with_action: int = 1,
    min_trigger_future_fill_rate: float = 0.95,
) -> Tuple[bool, Dict[str, Any]]:
    """Post-eval smoke: model must emit future_steps + next_action on trigger windows."""
    path = Path(pred_path)
    if not path.exists():
        return False, {"error": f"missing {path}"}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    trig = [r for r in rows if r.get("pred_is_trigger")]
    fut_ok = sum(1 for r in trig if r.get("pred_future_steps"))
    act_ok = sum(
        1
        for r in trig
        if parse_next_action(
            f"<|next_action_start|>{r.get('pred_next_action', '')}<|next_action_end|>"
        )
        or (r.get("pred_next_action") or "").strip()
    )
    fill_rate = (fut_ok / len(trig)) if trig else 1.0
    report = {
        "chunks": len(rows),
        "trigger_windows": len(trig),
        "pred_future_nonempty": fut_ok,
        "pred_action_nonempty": act_ok,
        "pred_future_fill_rate": round(fill_rate, 4),
        "min_trigger_future_fill_rate": min_trigger_future_fill_rate,
    }
    ok = (
        len(rows) > 0
        and (
            not trig
            or (
                fut_ok >= min_trigger_with_future
                and act_ok >= min_trigger_with_action
                and fill_rate >= min_trigger_future_fill_rate
            )
        )
    )
    return ok, report


def verify_output_format_invariant(text: str) -> bool:
    """Canonical output: future_steps block then newline then next_action block."""
    if not text:
        return False
    fs = "<|future_steps_start|>" in text and "<|future_steps_end|>" in text
    na = "<|next_action_start|>" in text and "<|next_action_end|>" in text
    if not (fs and na):
        return False
    # Order: future before next_action
    return text.find("<|future_steps_start|>") < text.find("<|next_action_start|>")
