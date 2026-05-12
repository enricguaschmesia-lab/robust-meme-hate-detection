"""Majority-class baseline for Hateful Memes.

No training. Determines the majority class from the train split (after the
deterministic train/held-out partition) and predicts that class for every dev
example. Writes a ``metrics.json`` in the same schema as ``train.stage1`` so
that ``cluster.sh summarize-results`` and the Phase-2 aggregation work
uniformly across all four baselines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from robust_meme_hate_detection.data.hateful_memes import load_hateful_memes_records
from robust_meme_hate_detection.data.splits import load_train_split
from robust_meme_hate_detection.eval.metrics import classification_metrics
from robust_meme_hate_detection.utils.logging import JsonRunLogger
from robust_meme_hate_detection.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(args.seed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    dataset_root = Path(data_cfg["dataset_root"])
    val_split = data_cfg.get("val_split", "dev")

    t0 = time.time()
    train_records = load_hateful_memes_records(dataset_root, "train")
    if "train_split_file" in data_cfg and data_cfg["train_split_file"]:
        split = load_train_split(data_cfg["train_split_file"])
        keep = split.train_id_set
        train_records = [r for r in train_records if str(r.id) in keep]
    val_records = load_hateful_memes_records(dataset_root, val_split)

    pos = sum(1 for r in train_records if r.label == 1)
    neg = sum(1 for r in train_records if r.label == 0)
    majority_label = 1 if pos > neg else 0
    train_pos_rate = pos / max(1, pos + neg)

    val_labels = [int(r.label) for r in val_records if r.label is not None]
    if len(val_labels) < len(val_records):
        raise RuntimeError(
            f"Majority baseline needs labelled dev: {len(val_records) - len(val_labels)} unlabelled records found."
        )

    # Probabilities are degenerate: assign train positive-rate to every example
    # so AUROC is defined (it will be 0.5 because all probs are equal, but the
    # function returns NaN if there's only one prediction class — we want a
    # clean number for the report).
    probs = [float(train_pos_rate)] * len(val_labels)
    metrics = classification_metrics(probs, val_labels, threshold=0.5)
    metrics_dict = metrics.to_dict()

    elapsed = time.time() - t0
    logger = JsonRunLogger(out_dir)
    logger.write_metrics({
        "config_path": str(args.config),
        "seed": int(args.seed),
        "modality": "majority",
        "device": "cpu",
        "n_train": len(train_records),
        "n_val": len(val_labels),
        "trainable_params": 0,
        "total_params": 0,
        "majority_label": majority_label,
        "train_pos_rate": train_pos_rate,
        **{f"val_{k}": v for k, v in metrics_dict.items()},
        "best_epoch": -1,
        "epochs_run": 0,
        "elapsed_seconds": elapsed,
    })

    print(
        f"majority_label={majority_label}  train_pos_rate={train_pos_rate:.4f}  "
        f"n_val={len(val_labels)}  val_accuracy={metrics_dict['accuracy']:.4f}  "
        f"val_macro_f1={metrics_dict['macro_f1']:.4f}  val_auroc={metrics_dict['auroc']:.4f}",
        flush=True,
    )

    summary_path = out_dir / "baseline_majority.json"
    summary_path.write_text(json.dumps({
        "majority_label": majority_label,
        "train_pos_rate": train_pos_rate,
        "n_train": len(train_records),
        "n_val": len(val_labels),
        "metrics": metrics_dict,
    }, indent=2))


if __name__ == "__main__":
    main()
