"""Build a Hateful Memes DataLoader and print one batch summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from robust_meme_hate_detection.data.hateful_memes import build_hateful_memes_dataloader
from robust_meme_hate_detection.data.transforms import Compose, NormalizeTensor, ResizeToTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/data"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transform = Compose([ResizeToTensor(224), NormalizeTensor()])
    loader = build_hateful_memes_dataloader(
        dataset_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_transform=transform,
    )
    batch = next(iter(loader))
    print(f"split={args.split}")
    print(f"batch ids={list(batch['id'])}")
    print(f"image tensor shape={tuple(batch['image'].shape)}")
    print(f"labels={batch['label'].tolist()}")
    print(f"first text={batch['text'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
