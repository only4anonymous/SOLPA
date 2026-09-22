"""
Graph QA Dataset for joint training with ProAct E2E.

Lightweight dataset that wraps pre-sampled QA pairs and dynamically renders
subgraph images. Designed to be used as an auxiliary dataset alongside
ProActE2EDataset in the joint training loop.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# Keep graph QA images readable but bounded. Very large rendered graphs can
# expand into thousands of visual tokens and OOM when QA is mixed into training.
MAX_GRAPH_PIXELS = 512 * 28 * 28
LABELS = ["A", "B", "C", "D"]


# === QA_DATASET_OPEN_ENDED ===
OPEN_ENDED_USER_SUFFIX = (
    "Answer in plain text (step name or short phrase only). "
    "Do not reply with A/B/C/D."
)


def _qa_answer_text(sample: dict) -> str:
    if sample.get("answer_text"):
        return str(sample["answer_text"]).strip()
    opts = sample.get("options") or []
    ans_idx = sample.get("answer")
    if isinstance(ans_idx, int) and 0 <= ans_idx < len(opts):
        return str(opts[ans_idx]).strip()
    return ""

GRAPH_LEGEND = (
    "You are analyzing a task progress graph.\n"
    "Solid colored boxes are real action steps:\n"
    "  - Green box      ([done])     = the user has already completed this step\n"
    "  - Yellow box     ([current])  = the step the user is performing right now\n"
    "  - Light-blue box ([frontier]) = currently legal next action (all\n"
    "      prerequisites already satisfied -- can be started now)\n"
    "  - Gray box       ([future])   = still blocked: at least one\n"
    "      prerequisite is not yet done, so it cannot be started now\n"
    "Both frontier (blue) and future (gray) appear in the graph; only\n"
    "  frontier ones are feasible at this moment.\n"
    "Dashed grey boxes (no fill, smaller font) are PHASE CONTAINERS, not\n"
    "  action steps. They group related actions under a phase\n"
    "  (e.g. 'Get Ingredients', 'Preparation Phase', 'Finish Serve').\n"
    "  They are shown only to convey graph structure -- never select a\n"
    "  dashed box as an action.\n"
    "A virtual box labelled 'N done prereqs' summarises N already-done\n"
    "  prerequisites that all converge on the same next step.\n"
    "Arrows show prerequisite dependencies (parent -> child).\n"
    "An edge label (AND / OR) describes how the child node is gated:\n"
    "  - AND: every parent of the child must be done before it can run\n"
    "  - OR : any one parent being done is enough to enable the child\n"
    "Answer the question by selecting the correct option letter."
)


# Text-mode legend (compact_yaml subgraph). Mirrors the e2e/APA text-graph
# decision context: no image, the graph is given as YAML-like text.
GRAPH_LEGEND_TEXT = (
    "You are analyzing a task progress graph given as text.\n"
    "Each node is an action step with a status:\n"
    "  - done     = already completed by the user\n"
    "  - current  = the step the user is performing right now\n"
    "  - frontier = currently legal next action (all prerequisites done)\n"
    "  - future   = still blocked (a prerequisite is not yet done)\n"
    "Only frontier steps are feasible right now. Prerequisite edges are listed\n"
    "per node; AND means all parents must be done, OR means any one parent.\n"
    "Answer in plain text using exact step names from the graph."
)


class GraphQAMiniDataset(Dataset):
    """A small dataset wrapping pre-sampled QA pairs for one epoch."""

    def __init__(
        self,
        qa_samples: List[dict],
        taxonomy: dict,
        processor: Any,
        tokenizer: Any,
        max_depth: int = 5,
        max_back_depth: int = 1,
        include_all_done: bool = False,
        cached_image_dir: Optional[str] = None,
        reasoning_mode: Optional[str] = None,
    ):
        self.samples = qa_samples
        self.taxonomy = taxonomy
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.max_back_depth = max_back_depth
        self.include_all_done = include_all_done
        self.cached_image_dir = cached_image_dir
        # Reasoning traces are useful for ablations, but should not silently
        # become assistant-side teacher-forced context in joint E2E training.
        # Default to pure answer supervision even when the JSON contains a
        # `reasoning` field.
        mode = reasoning_mode or os.environ.get("GRAPH_QA_REASONING_MODE", "ignore")
        mode = str(mode).strip().lower()
        valid_modes = {"ignore", "user_hint", "assistant_answer", "assistant_full"}
        self.reasoning_mode = mode if mode in valid_modes else "ignore"
        # HGQA2: render the QA subgraph as text (compact_yaml) instead of a PNG so
        # the QA forward shares the SAME modality/prompt skeleton as the e2e/APA
        # decision path (no vision tokens). Default "image" keeps full back-compat.
        sg_mode = os.environ.get("GRAPH_QA_SUBGRAPH_MODE", "image").strip().lower()
        self.subgraph_mode = sg_mode if sg_mode in {"image", "text", "none"} else "image"
        self.subgraph_text_format = os.environ.get(
            "GRAPH_QA_SUBGRAPH_TEXT_FORMAT", "compact_yaml"
        ).strip().lower()

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _state_key(task_name, completed_steps, current_step):
        cs = tuple(sorted(completed_steps))
        raw = f"{task_name}|{cs}|{current_step}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_image(self, sample):
        if self.cached_image_dir:
            key = self._state_key(
                sample["task_name"], sample["completed_steps"], sample["current_step"]
            )
            path = os.path.join(self.cached_image_dir, f"{key}.png")
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except Exception:
                    pass

        from train.graph_add.tool.subgraph import build_subgraph_for_step
        try:
            img = build_subgraph_for_step(
                self.taxonomy, sample["task_name"],
                sample["completed_steps"], sample["current_step"],
                mode="image", max_depth=self.max_depth,
                max_back_depth=self.max_back_depth,
                include_all_done=self.include_all_done,
            )
            if img is None:
                return None
            w, h = img.size
            if w * h > MAX_GRAPH_PIXELS:
                scale = math.sqrt(MAX_GRAPH_PIXELS / (w * h))
                img = img.resize(
                    (max(28, int(w * scale)), max(28, int(h * scale))),
                    Image.LANCZOS,
                )
            return img
        except Exception:
            return None

    def _get_subgraph_text(self, sample):
        from train.graph_add.tool.subgraph import build_subgraph_for_step
        try:
            omit = str(os.environ.get("GRAPH_QA_OMIT_LEGAL_POOL", "")).strip().lower() in {"1", "true", "yes"}
            old_omit = os.environ.get("OMIT_LEGAL_POOL_FOR_QA")
            if omit:
                os.environ["OMIT_LEGAL_POOL_FOR_QA"] = "1"
            try:
                txt = build_subgraph_for_step(
                    self.taxonomy, sample["task_name"],
                    sample["completed_steps"], sample["current_step"],
                    mode="text", text_format=self.subgraph_text_format,
                    max_depth=self.max_depth, max_back_depth=self.max_back_depth,
                    include_all_done=self.include_all_done,
                )
            finally:
                if omit:
                    if old_omit is None:
                        os.environ.pop("OMIT_LEGAL_POOL_FOR_QA", None)
                    else:
                        os.environ["OMIT_LEGAL_POOL_FOR_QA"] = old_omit
            if txt and isinstance(txt, str) and txt.strip():
                return txt
            return None
        except Exception:
            return None

    def _build_none_item(self, idx):
        """No-graph QA: question + answer with no subgraph (image or text).

        Used for y4_status_recovery (status/completion questions) and the
        R-Y2-OFF ablation.  The model must answer from video/dialogue context
        alone — no task-graph context is injected.
        """
        n = len(self)
        sample = None
        for attempt in range(min(n, 8)):
            cand = self.samples[(idx + attempt) % n]
            if _qa_answer_text(cand):
                sample = cand
                break
        if sample is None:
            return None
        answer_text = _qa_answer_text(sample)
        if not answer_text:
            return None
        sys_prompt = (
            "You are a task-progress assistant. "
            "Answer based on the video and dialogue history only."
        )
        user_text = f"{sample['question']}\n\n{OPEN_ENDED_USER_SUFFIX}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        enc = self.processor(text=[text], return_tensors="pt", padding=False)
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = torch.full_like(input_ids, -100)
        asst_marker = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        marker_len = len(asst_marker)
        asst_pos = -1
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i:i + marker_len].tolist() == asst_marker:
                asst_pos = i + marker_len
        if asst_pos < 0 or asst_pos >= input_ids.size(0):
            return None
        end_marker = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        end_pos = input_ids.size(0)
        if end_marker:
            end_len = len(end_marker)
            for j in range(asst_pos, input_ids.size(0) - end_len + 1):
                if input_ids[j:j + end_len].tolist() == end_marker:
                    end_pos = j
                    break
        if end_pos <= asst_pos:
            return None
        labels[asst_pos:end_pos] = input_ids[asst_pos:end_pos]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_text_item(self, idx):
        """Text-modality QA: compact_yaml subgraph, no image / pixel_values.

        Only open-ended (answer_text) QA is supported in text mode -- which is
        exactly the APA-aligned recipe (answers in the apa_parallel space)."""
        n = len(self)
        sample = None
        sg_text = None
        for attempt in range(min(n, 8)):
            cand = self.samples[(idx + attempt) % n]
            sg_text = self._get_subgraph_text(cand)
            if sg_text is not None:
                sample = cand
                break
        if sample is None:
            return None

        answer_text = _qa_answer_text(sample)
        if not answer_text:
            return None
        user_text = (
            f"{sample['question']}\n\n[GRAPH]\n{sg_text}\n\n{OPEN_ENDED_USER_SUFFIX}"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": GRAPH_LEGEND_TEXT}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        enc = self.processor(text=[text], return_tensors="pt", padding=False)
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = torch.full_like(input_ids, -100)
        asst_marker = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        marker_len = len(asst_marker)
        asst_pos = -1
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i:i + marker_len].tolist() == asst_marker:
                asst_pos = i + marker_len
        if asst_pos < 0 or asst_pos >= input_ids.size(0):
            return None
        end_marker = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        end_pos = input_ids.size(0)
        if end_marker:
            end_len = len(end_marker)
            for j in range(asst_pos, input_ids.size(0) - end_len + 1):
                if input_ids[j:j + end_len].tolist() == end_marker:
                    end_pos = j
                    break
        if end_pos <= asst_pos:
            return None
        labels[asst_pos:end_pos] = input_ids[asst_pos:end_pos]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx):
        # Sample-level subgraph_mode override: y4_status_recovery items carry
        # "subgraph_mode": "none" so they bypass graph rendering regardless of
        # the dataset-level GRAPH_QA_SUBGRAPH_MODE env setting.
        sample_sg = (self.samples[idx % len(self)].get("subgraph_mode") or "")
        if sample_sg == "none" or self.subgraph_mode == "none":
            return self._build_none_item(idx)
        if self.subgraph_mode == "text":
            return self._build_text_item(idx)
        n = len(self)
        for attempt in range(min(n, 8)):
            sample = self.samples[(idx + attempt) % n]
            img = self._get_image(sample)
            if img is not None:
                break
        else:
            return None

        options = sample.get("options") or []
        reasoning = str(sample.get("reasoning", "") or "").strip()
        reasoning_mode = self.reasoning_mode if reasoning else "ignore"
        if not options:
            answer_text = _qa_answer_text(sample)
            user_text = f"{sample['question']}\n\n{OPEN_ENDED_USER_SUFFIX}"
            assistant_text = answer_text
        else:
            opts_text = "\n".join(
                f"{LABELS[i]}. {opt}" for i, opt in enumerate(options)
            )
            answer_letter = LABELS[sample["answer"]]
            if reasoning_mode == "user_hint":
                user_text = (
                    f"{sample['question']}\n\n{opts_text}\n\n"
                    f"Graph decision certificate (legality / thread / rollout evidence):\n{reasoning}\n\n"
                    "Use the certificate only as supporting evidence. "
                    "Answer with the letter of the correct option."
                )
                assistant_text = answer_letter
            elif reasoning_mode in {"assistant_answer", "assistant_full"}:
                user_text = (
                    f"{sample['question']}\n\n{opts_text}\n\n"
                    "Give one short reason, then answer as: Answer: <letter>."
                )
                assistant_text = f"Reason: {reasoning}\nAnswer: {answer_letter}"
            else:
                user_text = (
                    f"{sample['question']}\n\n{opts_text}\n\n"
                    "Answer with the letter of the correct option."
                )
                assistant_text = answer_letter

        messages = [
            {"role": "system", "content": [{"type": "text", "text": GRAPH_LEGEND}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        enc = self.processor(
            text=[text], images=[img], return_tensors="pt", padding=False,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = torch.full_like(input_ids, -100)
        asst_marker = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        marker_len = len(asst_marker)
        asst_pos = -1
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i:i + marker_len].tolist() == asst_marker:
                asst_pos = i + marker_len
        if asst_pos < 0 or asst_pos >= input_ids.size(0):
            return None
        # open-ended full-token labels
        if not options:
            end_marker = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            end_pos = input_ids.size(0)
            if end_marker:
                end_len = len(end_marker)
                for j in range(asst_pos, input_ids.size(0) - end_len + 1):
                    if input_ids[j : j + end_len].tolist() == end_marker:
                        end_pos = j
                        break
            if end_pos <= asst_pos:
                return None
            labels[asst_pos:end_pos] = input_ids[asst_pos:end_pos]
        elif reasoning_mode == "assistant_full":
            # Reproduction-only mode: train the full teacher rationale.  This is
            # intentionally not the default because it competes with E2E output
            # formatting and can dominate the auxiliary gradient.
            end_marker = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            end_pos = input_ids.size(0)
            if end_marker:
                end_len = len(end_marker)
                for j in range(asst_pos, input_ids.size(0) - end_len + 1):
                    if input_ids[j : j + end_len].tolist() == end_marker:
                        end_pos = j
                        break
            if end_pos <= asst_pos:
                return None
            labels[asst_pos:end_pos] = input_ids[asst_pos:end_pos]
        elif reasoning_mode == "assistant_answer":
            # Legacy answer-only ablation: keep teacher rationale in the
            # assistant context, but only supervise "Answer: <letter>".
            answer_marker = self.tokenizer.encode("Answer:", add_special_tokens=False)
            end_marker = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            end_pos = input_ids.size(0)
            if end_marker:
                end_len = len(end_marker)
                for j in range(asst_pos, input_ids.size(0) - end_len + 1):
                    if input_ids[j : j + end_len].tolist() == end_marker:
                        end_pos = j
                        break
            ans_start = -1
            if answer_marker:
                mk_len = len(answer_marker)
                for j in range(asst_pos, end_pos - mk_len + 1):
                    if input_ids[j : j + mk_len].tolist() == answer_marker:
                        ans_start = j
                        break
            if ans_start < 0 or end_pos <= ans_start:
                return None
            labels[ans_start:end_pos] = input_ids[ans_start:end_pos]
        else:
            # Default and user_hint modes: train the same single-letter target
            # as regular QA, so reasoning data cannot distort the answer format.
            labels[asst_pos] = input_ids[asst_pos]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        pv = enc.get("pixel_values")
        if pv is not None:
            result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv
        igt = enc.get("image_grid_thw")
        if igt is not None:
            result["image_grid_thw"] = igt.squeeze(0) if igt.dim() > 1 else igt

        return result


def qa_collate_fn(batch, pad_token_id=0):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        labels[i, :L] = b["labels"]

    result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.cat([
            b["image_grid_thw"].unsqueeze(0) if b["image_grid_thw"].dim() == 1
            else b["image_grid_thw"]
            for b in batch
        ], dim=0)

    return result
