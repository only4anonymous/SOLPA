from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def _resolve_adapter_model(model):
    inner = getattr(model, "module", None)
    if inner is not None and hasattr(inner, "set_adapter"):
        return inner
    return model


def _is_adapter_param_name(name: str, adapter_name: str) -> bool:
    if not name or not adapter_name:
        return False
    if f".lora_A.{adapter_name}." in name or f".lora_B.{adapter_name}." in name:
        return True
    if f".modules_to_save.{adapter_name}." in name:
        return True
    return False


def collect_adapter_parameter_names(names: Sequence[str], adapter_name: str) -> List[str]:
    return [name for name in names if _is_adapter_param_name(name, adapter_name)]


def collect_adapter_parameters(named_parameters: Iterable[Tuple[str, object]], adapter_name: str) -> List[object]:
    return [param for name, param in named_parameters if _is_adapter_param_name(name, adapter_name)]


def set_active_adapter_trainable(model, adapter_name: str) -> None:
    adapter_model = _resolve_adapter_model(model)
    adapter_model.set_adapter(adapter_name)
    for name, param in model.named_parameters():
        param.requires_grad = _is_adapter_param_name(name, adapter_name)


def set_named_adapters_trainable(model, adapter_names: Sequence[str]) -> None:
    adapter_names = tuple(str(name) for name in adapter_names if str(name))
    for name, param in model.named_parameters():
        param.requires_grad = any(_is_adapter_param_name(name, adapter_name) for adapter_name in adapter_names)


def has_adapter(model, adapter_name: str) -> bool:
    adapter_model = _resolve_adapter_model(model)
    peft_config = getattr(adapter_model, "peft_config", None)
    return isinstance(peft_config, dict) and adapter_name in peft_config


def copy_adapter_weights_(model, source_adapter: str, target_adapter: str) -> int:
    named_parameters = dict(model.named_parameters())
    copied = 0
    for name, param in list(named_parameters.items()):
        if not _is_adapter_param_name(name, source_adapter):
            continue
        target_name = name.replace(f".{source_adapter}.", f".{target_adapter}.")
        target_param = named_parameters.get(target_name)
        if target_param is None:
            continue
        if hasattr(target_param, "data") and hasattr(param, "data"):
            try:
                target_param.data.copy_(param.data)
            except Exception:
                try:
                    target_param.data = param.data.clone()
                except Exception:
                    target_param.data = param.data
            copied += 1
    return copied


def _extract_span(text: str, left: str, right: str) -> str:
    if not text:
        return ""
    start = text.find(left)
    if start == -1:
        return ""
    end = text.find(right, start + len(left))
    if end == -1:
        return ""
    return text[start : end + len(right)]


def extract_two_stage_targets(full_target: str) -> Tuple[str, str]:
    state_parts = []
    decision_parts = []
    for left, right, sink in (
        ("<|trigger_start|>", "<|trigger_end|>", state_parts),
        ("<|task_start|>", "<|task_end|>", state_parts),
        ("<|step_start|>", "<|step_end|>", state_parts),
        ("<|future_steps_start|>", "<|future_steps_end|>", decision_parts),
        ("<|next_action_start|>", "<|next_action_end|>", decision_parts),
    ):
        span = _extract_span(full_target, left, right)
        if span:
            sink.append(span)
    return "\n".join(state_parts), "\n".join(decision_parts)


def mask_labels_by_regions(labels: Sequence[int], regions: Sequence[Tuple[int, int]]) -> List[int]:
    masked = [-100] * len(labels)
    for region in regions:
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            continue
        start, end = int(region[0]), int(region[1])
        start = max(0, start)
        end = min(len(labels), end)
        if end <= start:
            continue
        for idx in range(start, end):
            masked[idx] = int(labels[idx])
    return masked


def build_two_stage_branch_labels(
    batch_labels,
    is_regions: Sequence[Tuple[int, int]],
    task_regions: Sequence[Tuple[int, int]],
    step_regions: Sequence[Tuple[int, int]],
):
    state_labels = batch_labels.new_full(batch_labels.shape, -100)
    decision_labels = batch_labels.clone()
    batch_size = int(batch_labels.size(0))
    seq_len = int(batch_labels.size(1))

    for row_idx in range(batch_size):
        regions = []
        for region_list in (is_regions, task_regions, step_regions):
            region = region_list[row_idx] if row_idx < len(region_list) else None
            if not isinstance(region, (list, tuple)) or len(region) != 2:
                continue
            start, end = int(region[0]), int(region[1])
            start = max(0, start)
            end = min(seq_len, end)
            if end <= start:
                continue
            regions.append((start, end))
            state_labels[row_idx, start:end] = batch_labels[row_idx, start:end]

        if not regions:
            decision_labels[row_idx, :] = -100
            continue

        state_end = max(end for _, end in regions)
        decision_labels[row_idx, :state_end] = -100

    return state_labels, decision_labels
