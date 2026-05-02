"""Inspect a local Hateful Memes dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_meme_hate_detection.data.hateful_memes import HatefulMemesDataset, compute_split_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/data"))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--check-images", action="store_true", help="Open the first image from each available split.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = compute_split_stats(args.data_root, args.splits)
    print(json.dumps(stats, indent=2, sort_keys=True))

    if args.check_images:
        for split in args.splits:
            if split not in stats:
                continue
            dataset = HatefulMemesDataset(args.data_root, split=split, require_images=True)
            if len(dataset) == 0:
                continue
            sample = dataset[0]
            print(
                f"{split}: first sample id={sample['id']} label={sample['label']} "
                f"image_size={sample['image'].size} text_words={len(sample['text'].split())}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
