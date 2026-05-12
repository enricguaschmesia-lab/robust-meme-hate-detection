"""Helpers to load deterministic train/train_held_out partitions.

The split is materialised once by ``scripts/make_train_split.py`` and stored
as a JSON file under ``data/processed/splits/``. Every training run reads the
same partition so that all seeds and stages compare like-for-like.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainSplit:
    train_ids: list[str]
    held_out_ids: list[str]
    seed: int
    source_jsonl: str
    held_out_size: int

    @property
    def train_id_set(self) -> set[str]:
        return set(self.train_ids)

    @property
    def held_out_id_set(self) -> set[str]:
        return set(self.held_out_ids)


def load_train_split(path: str | Path) -> TrainSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainSplit(
        train_ids=[str(x) for x in payload["train_ids"]],
        held_out_ids=[str(x) for x in payload["held_out_ids"]],
        seed=int(payload.get("seed", 0)),
        source_jsonl=str(payload.get("source_jsonl", "")),
        held_out_size=int(payload.get("held_out_size", len(payload["held_out_ids"]))),
    )


def filter_records_by_ids(records, ids):  # generic; works with any iterable of records that have .id
    keep = set(str(i) for i in ids)
    return [r for r in records if str(r.id) in keep]
