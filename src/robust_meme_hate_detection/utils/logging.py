"""Lightweight JSON-based metrics logging.

We deliberately do not add wandb here. Metrics are written as JSON to the run
directory; the project-side workflow can later sync to wandb offline if
desired (see project_planning/model_architecture.md §10).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


class JsonRunLogger:
    """Append-only JSONL logger plus a single-shot ``metrics.json`` snapshot.

    Usage:

        log = JsonRunLogger("/scratch/.../experiments/stage1-seed0")
        log.log_step({"epoch": 0, "step": 100, "train_loss": 0.62})
        log.write_metrics({"val_auroc": 0.74, "val_macro_f1": 0.69})
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps_file = self.out_dir / "steps.jsonl"
        self.metrics_file = self.out_dir / "metrics.json"
        self._start_ts = time.time()

    def log_step(self, payload: Mapping[str, Any]) -> None:
        record = {"ts": time.time() - self._start_ts, **payload}
        with self.steps_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_default) + "\n")

    def write_metrics(self, payload: Mapping[str, Any]) -> None:
        # Merge with any existing snapshot so summarize-results sees the latest.
        existing: dict[str, Any] = {}
        if self.metrics_file.exists():
            existing = json.loads(self.metrics_file.read_text(encoding="utf-8"))
        existing.update(payload)
        self.metrics_file.write_text(
            json.dumps(existing, indent=2, default=_default) + "\n", encoding="utf-8"
        )


def _default(o: Any) -> Any:
    try:
        # numpy scalar
        return o.item()
    except AttributeError:
        return str(o)
