"""Hateful Memes data: record loader, train/held-out split, and CLIP transforms.

A *record* is one example ``{id, text, label, image_path}`` parsed from the
dataset's JSONL files (the loader is permissive about the public-mirror split
names). The image transform deliberately produces a raw ``[0, 1]`` tensor with
**no normalisation** -- normalisation happens inside the model so attacks can
reach raw pixels (see :mod:`robust_meme_hate_detection.model`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

# Public mirrors name the splits slightly differently; accept the common aliases.
SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train.jsonl",),
    "dev": ("dev.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl"),
    "test_seen": ("test_seen.jsonl",),
    "test_unseen": ("test_unseen.jsonl",),
}


@dataclass(frozen=True)
class MemeRecord:
    id: str
    text: str
    label: int | None  # None for unlabelled examples
    image_path: Path


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{n}") from exc


def load_records(dataset_root: str | Path, split: str) -> list[MemeRecord]:
    """Load every record for a logical split (concatenating aliased files)."""
    root = Path(dataset_root)
    files = [root / f for f in SPLIT_ALIASES.get(split, (f"{split}.jsonl",)) if (root / f).exists()]
    if not files:
        raise FileNotFoundError(f"No JSONL for split {split!r} under {root}")

    records: list[MemeRecord] = []
    for f in files:
        for item in _read_jsonl(f):
            ref = str(item.get("img") or item.get("image") or f"img/{str(item.get('id', '')).zfill(5)}.png")
            path = Path(ref)
            label = item.get("label")
            records.append(
                MemeRecord(
                    id=str(item.get("id", path.stem)).zfill(5),
                    text=str(item.get("text", "")),
                    label=None if label is None else int(label),
                    image_path=path if path.is_absolute() else root / path,
                )
            )
    return records


def load_train_ids(split_file: str | Path) -> set[str]:
    """Return the committed deterministic train-id set (held-out ids excluded)."""
    payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    return {str(x) for x in payload["train_ids"]}


# --------------------------------------------------------------------- transforms

CLIP_SIZE = 224


class ClipImageTransform:
    """PIL RGB image -> bicubic resize + centre-crop 224 -> ``[0, 1]`` CHW tensor.

    Not normalised: :class:`Normalize` lives inside the model.
    """

    def __init__(self, size: int = CLIP_SIZE) -> None:
        self.size = size

    def __call__(self, image: Image.Image):
        from torchvision import transforms as T

        ops = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(self.size),
            T.ToTensor(),  # -> [0, 1] CHW
        ])
        return ops(image)


class ClipTokenizer:
    """Wrap the OpenCLIP tokenizer so it maps a caption string to a ``(77,)`` tensor."""

    def __init__(self, arch: str = "ViT-B-32") -> None:
        import open_clip

        self._tok = open_clip.get_tokenizer(arch)

    def __call__(self, text: str):
        return self._tok([text])[0]


# ------------------------------------------------------------------------ dataset


class MemeDataset:
    """Yields the processed ``(image_tensor, token_ids, label)`` plus the raw PIL
    image and caption string.

    The raw views let the robust trainer build an on-the-fly perturbed view
    (see :class:`~robust_meme_hate_detection.augment.RobustAugmenter`); set
    ``keep_raw=False`` for plain evaluation to save memory.
    """

    def __init__(
        self,
        records: list[MemeRecord],
        image_tfm: Callable,
        text_tfm: Callable,
        *,
        keep_raw: bool = True,
    ) -> None:
        self.records = list(records)
        self.image_tfm = image_tfm
        self.text_tfm = text_tfm
        self.keep_raw = keep_raw
        missing = [r.image_path for r in self.records if not r.image_path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing (e.g. {missing[0]})")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        img = Image.open(rec.image_path).convert("RGB")
        item = {
            "id": rec.id,
            "image": self.image_tfm(img),
            "text": self.text_tfm(rec.text),
            "label": -1 if rec.label is None else int(rec.label),
        }
        if self.keep_raw:
            item["pil_image"] = img
            item["raw_text"] = rec.text
        return item


def raw_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that stacks tensors but keeps PIL images / strings as plain lists."""
    import torch

    out = {
        "id": [b["id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "text": torch.stack([b["text"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
    if "pil_image" in batch[0]:
        out["pil_image"] = [b["pil_image"] for b in batch]
        out["raw_text"] = [b["raw_text"] for b in batch]
    return out
