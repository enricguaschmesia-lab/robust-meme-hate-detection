"""Create a perturbed copy of the Hateful Memes dataset.

Output structure mirrors the original dataset:

    perturbed/
      hate_meta/
        img/
        train.jsonl
        dev.jsonl
        test.jsonl
        LICENSE.txt
        README.md

By default, this script perturbs BOTH text and image for every sample.
The goal is inspection/debugging: save a visibly perturbed dataset that you
can open and verify.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from PIL import Image

from robust_meme_hate_detection.data.hateful_memes import HatefulMemesDataset
from robust_meme_hate_detection.perturbations import TextPerturbation, ImagePerturbation


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


class RandomChoicePerturbation:
    """Apply exactly one randomly chosen perturbation from a weighted list."""

    def __init__(self, perturbations_with_weights: list[tuple[Any, float]]) -> None:
        self.perturbations = [p for p, _ in perturbations_with_weights]
        self.weights = [w for _, w in perturbations_with_weights]

    def __call__(self, x):
        perturbation = random.choices(self.perturbations, weights=self.weights, k=1)[0]
        return perturbation(x)


def build_text_perturbation() -> RandomChoicePerturbation:
    """Choose one mild text perturbation per sample."""
    return RandomChoicePerturbation(
        [
            (TextPerturbation(mode="leetspeak", probability=1.0, severity=0.35), 1.0),
            (TextPerturbation(mode="spacing", probability=1.0, severity=0.15), 1.0),
            (TextPerturbation(mode="char_swap", probability=1.0, severity=0.08), 0.8),
            (TextPerturbation(mode="punctuation", probability=1.0, severity=0.08), 0.7),
            (TextPerturbation(mode="case_noise", probability=1.0, severity=0.15), 0.6),
            (TextPerturbation(mode="censoring", probability=1.0, severity=0.12), 0.5),
            (TextPerturbation(mode="char_deletion", probability=1.0, severity=0.06), 0.4),
        ]
    )


def build_image_perturbation() -> RandomChoicePerturbation:
    """Choose one image perturbation per sample.

    Blur is deliberately less frequent so the saved dataset has better variety.
    """
    return RandomChoicePerturbation(
        [
            (ImagePerturbation(mode="gaussian_noise", probability=1.0, severity=0.04), 1.0),
            (ImagePerturbation(mode="compression", probability=1.0, severity=0.55), 1.0),
            (ImagePerturbation(mode="brightness", probability=1.0, severity=0.25), 1.0),
            (ImagePerturbation(mode="contrast", probability=1.0, severity=0.25), 1.0),
            (ImagePerturbation(mode="translation", probability=1.0, severity=0.06), 0.9),
            (ImagePerturbation(mode="crop", probability=1.0, severity=0.07), 0.9),
            (ImagePerturbation(mode="occlusion", probability=1.0, severity=0.08), 0.8),
            (ImagePerturbation(mode="blur", probability=1.0, severity=0.12), 0.35),
        ]
    )


def available_split_files(dataset_root: Path) -> list[Path]:
    """Return JSONL split files that exist in the dataset root."""
    candidates = [
        "train.jsonl",
        "dev.jsonl",
        "dev_seen.jsonl",
        "dev_unseen.jsonl",
        "test.jsonl",
        "test_seen.jsonl",
        "test_unseen.jsonl",
    ]
    return [dataset_root / name for name in candidates if (dataset_root / name).exists()]


def maybe_cast_id(record_id: str) -> int | str:
    return int(record_id) if record_id.isdigit() else record_id


def save_split(
    dataset_root: Path,
    output_root: Path,
    split_name: str,
    text_perturbation,
    image_perturbation,
) -> None:
    dataset = HatefulMemesDataset(
        dataset_root=dataset_root,
        split=split_name,
        image_transform=None,   # keep PIL images
        text_transform=None,
        require_images=True,
        apply_perturbations=True,
        image_perturbation=image_perturbation,
        text_perturbation=text_perturbation,
    )

    output_jsonl = output_root / f"{split_name}.jsonl"
    rows: list[dict[str, Any]] = []

    print(f"Processing split: {split_name} ({len(dataset)} samples)")

    for i in range(len(dataset)):
        sample = dataset[i]
        record = dataset.records[i]

        image = sample["image"]
        text = str(sample["text"])

        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL image, got {type(image)}. "
                "Do not pass image_transform when saving perturbed data."
            )

        # Preserve original relative image path, e.g. img/12345.png
        relative_img_path = Path(record.raw_image_ref)
        full_img_path = output_root / relative_img_path
        full_img_path.parent.mkdir(parents=True, exist_ok=True)

        image.save(full_img_path)

        item = {
            "id": maybe_cast_id(record.id),
            "img": record.raw_image_ref,
            "text": text,
        }

        if record.label is not None:
            item["label"] = int(record.label)

        rows.append(item)

        if (i + 1) % 1000 == 0:
            print(f"  saved {i + 1}/{len(dataset)}")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  wrote {output_jsonl}")


def copy_metadata_files(dataset_root: Path, output_root: Path) -> None:
    for filename in ["LICENSE.txt", "README.md"]:
        src = dataset_root / filename
        dst = output_root / filename
        if src.exists():
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data/raw/hate_meta")),
        help="Path to the original Hateful Memes dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to save the perturbed dataset. Default: <dataset_root.parent>/perturbed/<dataset_root.name>",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Which split to perturb: train, dev, test, dev_seen, dev_unseen, test_seen, test_unseen, or all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder first if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = args.dataset_root
    if args.output_root is None:
        output_root = dataset_root.parent / "perturbed" / dataset_root.name
    else:
        output_root = args.output_root

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")

    text_perturbation = build_text_perturbation()
    image_perturbation = build_image_perturbation()

    copy_metadata_files(dataset_root, output_root)

    split_files = available_split_files(dataset_root)
    split_names = [p.stem for p in split_files]

    if args.split != "all":
        if args.split not in split_names:
            raise ValueError(
                f"Split '{args.split}' not found under {dataset_root}. "
                f"Available splits: {split_names}"
            )
        split_names = [args.split]

    for split_name in split_names:
        save_split(
            dataset_root=dataset_root,
            output_root=output_root,
            split_name=split_name,
            text_perturbation=text_perturbation,
            image_perturbation=image_perturbation,
        )

    print("\nDone.")
    print(f"Perturbed dataset saved at: {output_root}")


if __name__ == "__main__":
    main()