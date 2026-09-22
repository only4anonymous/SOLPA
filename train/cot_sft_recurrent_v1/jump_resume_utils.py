from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


def round_epoch(value: object) -> Optional[int]:
    try:
        epoch = float(value)
    except Exception:
        return None
    if epoch <= 0:
        return None
    return int(round(epoch))


def checkpoint_epoch(checkpoint_path: str | Path) -> Optional[int]:
    path = Path(checkpoint_path)
    if path.name.startswith("epoch_"):
        match = re.search(r"epoch_(\d+)", path.name)
        if match:
            return int(match.group(1))

    trainer_state = path / "trainer_state.json"
    if trainer_state.exists():
        try:
            payload = json.loads(trainer_state.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        epoch = round_epoch(payload.get("epoch"))
        if epoch is not None:
            return epoch
    return None


def collect_checkpoint_map(run_dir: str | Path) -> Dict[int, str]:
    base = Path(run_dir)
    out: Dict[int, str] = {}
    if not base.exists():
        return out
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        epoch = checkpoint_epoch(child)
        if epoch is None:
            continue
        out[epoch] = str(child)
    return dict(sorted(out.items()))


def find_latest_checkpoint(run_dir: str | Path) -> Optional[str]:
    checkpoint_map = collect_checkpoint_map(run_dir)
    if not checkpoint_map:
        return None
    latest_epoch = max(checkpoint_map)
    return checkpoint_map[latest_epoch]


def resolve_resume_checkpoint(target: str | Path) -> Optional[str]:
    path = Path(target)
    if not path.exists():
        return None
    if path.is_dir() and (path / "trainer_state.json").exists():
        return str(path)
    if path.is_dir():
        return find_latest_checkpoint(path)
    return str(path)


def _metrics_has_epoch(metrics_path: Path, epoch: int) -> bool:
    if not metrics_path.exists():
        return False
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    if int(str(row.get("epoch", "")).strip()) == int(epoch):
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False


def metrics_epoch_present(
    run_dir: str | Path,
    epoch: int,
    *,
    expected_world: int,
) -> bool:
    base = Path(run_dir)
    if not _metrics_has_epoch(base / "metrics.csv", epoch):
        return False

    eval_pred = base / "eval_pred"
    if expected_world <= 0:
        return True
    for rank in range(int(expected_world)):
        shard = eval_pred / f"epoch_{epoch}.rank{rank}.jsonl"
        if not shard.exists():
            return False
    return True


def discover_epochs_needing_metrics(
    *,
    run_dir: str | Path,
    checkpoint_map: Dict[int, str],
    resume_epoch: int,
    expected_world: int,
) -> list[int]:
    epochs = []
    for epoch in sorted(checkpoint_map):
        if int(epoch) < int(resume_epoch):
            continue
        if not metrics_epoch_present(run_dir, epoch, expected_world=expected_world):
            epochs.append(int(epoch))
    return epochs


def iter_checkpoint_epochs(checkpoint_map: Dict[int, str], start_epoch: int) -> Iterable[int]:
    for epoch in sorted(checkpoint_map):
        if int(epoch) >= int(start_epoch):
            yield int(epoch)
