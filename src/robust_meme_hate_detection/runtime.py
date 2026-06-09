"""Shared runtime helpers used by both the train and evaluate entry points.

Keeps config loading, dataloader construction, the device/AMP boilerplate and the
clean inference loop in one place so the two CLIs stay short.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch.utils.data import DataLoader

from robust_meme_hate_detection.data import (
    ClipImageTransform,
    ClipTokenizer,
    MemeDataset,
    load_records,
    load_train_ids,
    raw_collate,
)

# How each modality turns a model into a single logit (baselines vs. fusion).
FORWARD_FNS: dict[str, Callable] = {
    "multimodal": lambda m, img, tok: m(img, tok),
    "image": lambda m, img, tok: m.forward_image_only(img),
    "text": lambda m, img, tok: m.forward_text_only(tok),
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_transforms(arch: str) -> tuple[ClipImageTransform, ClipTokenizer]:
    return ClipImageTransform(size=224), ClipTokenizer(arch=arch)


def build_records(data_cfg: dict[str, Any], split: str, *, restrict_train_split: bool = False):
    """Load records for ``split``; optionally keep only the committed train ids."""
    records = load_records(data_cfg["dataset_root"], split)
    if restrict_train_split and data_cfg.get("train_split_file"):
        keep = load_train_ids(data_cfg["train_split_file"])
        records = [r for r in records if r.id in keep]
    return records


def build_loader(
    records,
    image_tfm,
    text_tfm,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    keep_raw: bool,
) -> DataLoader:
    ds = MemeDataset(records, image_tfm, text_tfm, keep_raw=keep_raw)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        collate_fn=raw_collate if keep_raw else None,
    )


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def fuse_with_dropout(model, images, tokens, drop_text=None, drop_image=None):
    """``model(images, tokens)`` but optionally zero a branch's embedding per example.

    Zeroing at the embedding level mirrors ``forward_{image,text}_only`` and is how
    per-example modality dropout forces a real single-branch prediction.
    """
    v = model.encode_image(images).float()
    t = model.encode_text(tokens).float()
    if drop_text is not None:
        t = torch.where(drop_text, torch.zeros_like(t), t)
    if drop_image is not None:
        v = torch.where(drop_image, torch.zeros_like(v), v)
    return model.head(t, v)


@torch.no_grad()
def predict_probs(model, loader, device, *, modality: str = "multimodal") -> tuple[list[float], list[int], list[str]]:
    """Clean inference over a loader; returns ``(probs, labels, ids)``."""
    forward = FORWARD_FNS[modality]
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    ids: list[str] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["text"].to(device, non_blocking=True)
        logit = forward(model, images, tokens)
        probs.extend(torch.sigmoid(logit.float()).cpu().tolist())
        labels.extend(int(x) for x in batch["label"].tolist())
        ids.extend(batch["id"])
    return probs, labels, ids
