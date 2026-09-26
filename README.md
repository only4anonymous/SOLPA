# SOLPA

Two-stage Perceiver (W1) + Planner (W2) training and evaluation code for
graph-conditioned proactive assistance (Structure-Optional Learning for
Proactive Assistance), extending ProAct-Helper with history (H), task-graph
(G), and completion-conditioned QA (QA) supervision.

Due to the double-blind review period, trained checkpoints and datasets are
not included in this release; only training and evaluation code is provided.

## Repository layout

- `train/cot_memory_sft/train_proact.py` -- main two-stage LoRA training
  entry point (history + graph + QA joint training).
- `train/graph_add/` -- task-graph text/image rendering, completion-QA
  sampling (`qa_sampler.py`, `qa_dataset.py`), and QA pool generators
  (`gen_dlv3qa_yaml_medium.py`, `gen_apa_aligned_qa_medium.py`).
- `train/cot_sft_v2/`, `train/cot_sft_recurrent_v1/`, `train/l2/` -- shared
  two-stage training/metric utilities imported by `train_proact.py`.
- `test/onestep_planning_oracle/` -- one-step decision evaluation
  (step-success / parallel-action-rate / edit-distance scoring).

## Setup

```bash
pip install -r requirements.txt
```

## Training

The paper recipe is history K=12, graph dropout 0.7, remaining-step targets,
text-only Y2 QA, and bf16. Those are not the argparse defaults, so set them
explicitly:

```bash
export PYTHONPATH=.
export GRAPH_QA_SUBGRAPH_MODE=text
export QA_TYPES=y2
python -m accelerate.commands.launch --num_processes 8 \
  train/cot_memory_sft/train_proact.py \
  --model_name <path-to-Qwen2.5-VL-checkpoint> \
  --train_json <path-to-train.jsonl> --val_json <path-to-val.jsonl> \
  --frame_root <path-to-frames> --annotation <path-to-annotation.json> \
  --freeze_backbone --lora_stage w2 --no_proact_memory \
  --subgraph_mode text --subgraph_text_format compact_yaml \
  --subgraph_dropout 0.7 \
  --history_memory_mode recent_only --history_recent_k 12 \
  --future_steps_target remaining \
  --qa_train_json <path-to-qa-train.json> --qa_test_json <path-to-qa-test.json> \
  --qa_ratio 0.06 --qa_loss_weight 0.10 --qa_sampling_mode dlv3_yaml \
  --bf16 \
  --lora_rank 32 --learning_rate 2e-5 --num_train_epochs 10
```

See the docstrings in `train/cot_memory_sft/train_proact.py` for the full
argument list.

## Evaluation

```bash
python test/onestep_planning_oracle/eval_onestep_end2end_guard.py --help
```

## License

Code released for double-blind peer review. License to be finalized upon
acceptance.
