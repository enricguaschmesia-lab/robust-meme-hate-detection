#!/usr/bin/env python3
"""Materialize the deterministic train / train_held_out partition.

Reads ``train.jsonl`` from the staged dataset, performs a stratified split
(default 8000 train / 500 held-out), and writes the ID partition to a JSON
file. The output is committed to git so every run uses the same partition.

Usage:

    python scripts/make_train_split.py \\
        --train-jsonl /scratch/datasets/hate_meta/train.jsonl \\
        --out data/processed/splits/train_split.json \\
        --held-out 500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, help="Path to train.jsonl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--held-out", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with open(args.train_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit(f"No rows found in {args.train_jsonl}")

    # Stratify by label. IDs are zero-padded to 5 chars to match the
    # normalisation done by HatefulMemesRecord (see hateful_memes.py).
    by_label: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(str(r["id"]).zfill(5))

    rng = random.Random(args.seed)
    held_out: list[str] = []
    train_ids: list[str] = []
    total = sum(len(ids) for ids in by_label.values())
    for label, ids in by_label.items():
        rng.shuffle(ids)
        # Proportional held-out size, rounded.
        n_held = round(args.held_out * len(ids) / total)
        held_out.extend(ids[:n_held])
        train_ids.extend(ids[n_held:])

    # Trim/pad to exactly args.held_out if rounding drifted.
    if len(held_out) > args.held_out:
        # Move excess back to train (keeping order deterministic).
        excess = len(held_out) - args.held_out
        moved = held_out[-excess:]
        held_out = held_out[:-excess]
        train_ids.extend(moved)
    elif len(held_out) < args.held_out:
        # Pull from train (deterministically).
        deficit = args.held_out - len(held_out)
        rng.shuffle(train_ids)
        held_out.extend(train_ids[:deficit])
        train_ids = train_ids[deficit:]

    # Re-sort for deterministic on-disk order.
    train_ids = sorted(set(train_ids))
    held_out = sorted(set(held_out))
    overlap = set(train_ids) & set(held_out)
    if overlap:
        raise SystemExit(f"Internal error: {len(overlap)} ID overlap between train and held_out")

    payload = {
        "source_jsonl": args.train_jsonl,
        "seed": args.seed,
        "held_out_size": len(held_out),
        "train_size": len(train_ids),
        "train_ids": train_ids,
        "held_out_ids": held_out,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    pos_train = sum(1 for r in rows if str(r["id"]).zfill(5) in set(train_ids) and int(r["label"]) == 1)
    pos_held = sum(1 for r in rows if str(r["id"]).zfill(5) in set(held_out) and int(r["label"]) == 1)
    print(
        f"Wrote {out}: train={len(train_ids)} ({pos_train} pos, "
        f"{pos_train/len(train_ids):.3f}), held_out={len(held_out)} ({pos_held} pos, "
        f"{pos_held/len(held_out):.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
