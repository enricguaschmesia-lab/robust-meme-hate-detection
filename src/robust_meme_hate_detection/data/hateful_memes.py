"""PyTorch dataset utilities for the Hateful Memes dataset.

The original Hateful Memes release stores examples in JSONL files with fields
similar to:

    {"id": 42953, "img": "img/42953.png", "label": 0, "text": "..."}

Several public mirrors use slightly different split names. This module keeps the
loader permissive while preserving a stable output contract for the rest of the
project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from PIL import Image


SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train.jsonl",),
    "dev": ("dev.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl"),
    "dev_seen": ("dev_seen.jsonl",),
    "dev_unseen": ("dev_unseen.jsonl",),
    "test": ("test.jsonl", "test_seen.jsonl", "test_unseen.jsonl"),
    "test_seen": ("test_seen.jsonl",),
    "test_unseen": ("test_unseen.jsonl",),
}


@dataclass(frozen=True)
class HatefulMemesRecord:
    """One Hateful Memes example after path normalization."""

    id: str
    text: str
    label: int | None
    split: str
    image_path: Path
    raw_image_ref: str


def find_split_file(dataset_root: str | Path, split: str) -> Path:
    """Return the JSONL path for a split.

    If `split` maps to multiple files, this function returns the first existing
    file. Use `load_hateful_memes_records` with `split="dev"` or `split="test"`
    to concatenate all matching split files.
    """

    root = Path(dataset_root)
    candidates = SPLIT_ALIASES.get(split, (f"{split}.jsonl",))
    for filename in candidates:
        path = root / filename
        if path.exists():
            return path
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"No JSONL file for split '{split}' under {root}. Expected one of: {expected}")


def _existing_split_files(dataset_root: str | Path, split: str) -> list[Path]:
    root = Path(dataset_root)
    candidates = SPLIT_ALIASES.get(split, (f"{split}.jsonl",))
    return [root / filename for filename in candidates if (root / filename).exists()]


def _normalize_image_path(dataset_root: Path, item: Mapping[str, Any]) -> tuple[Path, str]:
    raw_ref = str(item.get("img") or item.get("image") or item.get("image_path") or "")
    if not raw_ref:
        example_id = str(item.get("id", "")).zfill(5)
        raw_ref = f"img/{example_id}.png"

    path = Path(raw_ref)
    if not path.is_absolute():
        path = dataset_root / path
    return path, raw_ref


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc


def load_hateful_memes_records(dataset_root: str | Path, split: str) -> list[HatefulMemesRecord]:
    """Load records from one logical split.

    `split="dev"` concatenates `dev_seen` and `dev_unseen` when both exist.
    `split="test"` concatenates `test_seen` and `test_unseen` when both exist.
    Missing labels are represented as `None` so unlabeled test files can still be
    loaded for inference.
    """

    root = Path(dataset_root)
    split_files = _existing_split_files(root, split)
    if not split_files:
        split_files = [find_split_file(root, split)]

    records: list[HatefulMemesRecord] = []
    for split_file in split_files:
        file_split = split_file.stem
        for item in _read_jsonl(split_file):
            image_path, raw_ref = _normalize_image_path(root, item)
            label = item.get("label")
            records.append(
                HatefulMemesRecord(
                    id=str(item.get("id", image_path.stem)).zfill(5),
                    text=str(item.get("text", "")),
                    label=None if label is None else int(label),
                    split=file_split,
                    image_path=image_path,
                    raw_image_ref=raw_ref,
                )
            )
    return records


class HatefulMemesDataset:
    """PyTorch-compatible dataset returning image, text, label, and metadata."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_transform: Callable[[Image.Image], Any] | None = None,
        text_transform: Callable[[str], Any] | None = None,
        require_images: bool = True,
        apply_perturbations: bool = False,
        image_perturbation: Callable[[Image.Image], Image.Image] | None = None,
        text_perturbation: Callable[[str], str] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.require_images = require_images
        self.apply_perturbations = apply_perturbations
        self.image_perturbation = image_perturbation
        self.text_perturbation = text_perturbation
        self.records = load_hateful_memes_records(self.dataset_root, split)

        if require_images:
            missing = [record.image_path for record in self.records if not record.image_path.exists()]
            if missing:
                sample = ", ".join(str(path) for path in missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} images referenced by split '{split}' are missing under {self.dataset_root}. "
                    f"First missing paths: {sample}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        
        # Apply perturbations first (before transforms)
        if self.apply_perturbations and self.image_perturbation is not None:
            image = self.image_perturbation(image)
        
        # Apply standard image transform
        if self.image_transform is not None:
            image = self.image_transform(image)

        text: Any = record.text
        
        # Apply perturbations first (before transforms)
        if self.apply_perturbations and self.text_perturbation is not None:
            text = self.text_perturbation(record.text)
        
        # Apply standard text transform
        if self.text_transform is not None:
            text = self.text_transform(text)

        label = -1 if record.label is None else record.label
        return {
            "id": record.id,
            "image": image,
            "text": text,
            "label": label,
            "split": record.split,
            "image_path": str(record.image_path),
        }


def build_hateful_memes_dataloader(
    dataset_root: str | Path,
    split: str,
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    image_transform: Callable[[Image.Image], Any] | None = None,
    text_transform: Callable[[str], Any] | None = None,
    require_images: bool = True,
    apply_perturbations: bool = False,
    image_perturbation: Callable[[Image.Image], Image.Image] | None = None,
    text_perturbation: Callable[[str], str] | None = None,
) -> Any:
    """Build a PyTorch DataLoader for Hateful Memes.

    Torch is imported lazily so metadata inspection can run on machines where
    PyTorch is not installed yet.
    
    Args:
        apply_perturbations: If True, apply perturbations to samples.
        image_perturbation: Callable to perturb images (e.g., ImagePerturbation or ComposePerturbation).
        text_perturbation: Callable to perturb text (e.g., TextPerturbation or ComposePerturbation).
    """

    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required to build dataloaders. Install project requirements first.") from exc

    dataset = HatefulMemesDataset(
        dataset_root=dataset_root,
        split=split,
        image_transform=image_transform,
        text_transform=text_transform,
        require_images=require_images,
        apply_perturbations=apply_perturbations,
        image_perturbation=image_perturbation,
        text_perturbation=text_perturbation,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def compute_split_stats(dataset_root: str | Path, splits: Iterable[str] = ("train", "dev", "test")) -> dict[str, Any]:
    """Compute lightweight split statistics without loading image pixels."""

    stats: dict[str, Any] = {}
    for split in splits:
        try:
            records = load_hateful_memes_records(dataset_root, split)
        except FileNotFoundError:
            continue
        labels = [record.label for record in records if record.label is not None]
        missing_images = sum(1 for record in records if not record.image_path.exists())
        label_counts = {str(label): labels.count(label) for label in sorted(set(labels))}
        text_lengths = [len(record.text.split()) for record in records]
        stats[split] = {
            "num_examples": len(records),
            "num_labeled": len(labels),
            "label_counts": label_counts,
            "missing_images": missing_images,
            "avg_text_words": round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0.0,
            "max_text_words": max(text_lengths) if text_lengths else 0,
        }
    return stats
