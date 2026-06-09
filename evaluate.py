"""Evaluate a trained checkpoint -- one entry point with four suites.

    python evaluate.py --ckpt runs/nat/best.pt --suite clean    --split test_unseen
    python evaluate.py --ckpt runs/nat/best.pt --suite modality  --split test_unseen
    python evaluate.py --ckpt runs/nat/best.pt --suite perturbed --split test_unseen
    python evaluate.py --ckpt runs/adv/best.pt --suite whitebox  --split test_unseen

Suites (each maps to a result in the report):

* ``clean``     -- AUROC / macro-F1 / accuracy / FPR / FNR on a split (Tables 1-3).
* ``modality``  -- full vs. image-only vs. text-only forward (the audit, Table 1).
* ``perturbed`` -- the naturalistic grid (text x image x severity) + composites;
                   per-cell AUROC, robustness gap dAUROC and ASR, plus the count
                   of examples that survive *every* cell (Tables 2-3).
* ``whitebox``  -- FGSM / PGD L-inf sweep over eps in {1,2,4,8}/255 (Table 3).

Use ``--max-samples`` to subsample for a quick CPU check; full splits are meant
to run on a GPU. Results are printed and written to ``--out`` (JSON) if given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from robust_meme_hate_detection.attacks import fgsm_image, pgd_image
from robust_meme_hate_detection.data import MemeRecord, load_records
from robust_meme_hate_detection.metrics import attack_success_rate, classification_metrics
from robust_meme_hate_detection.model import load_classifier
from robust_meme_hate_detection.perturbations import (
    IMAGE_MODES,
    TEXT_MODES,
    ImagePerturbation,
    TextPerturbation,
)
from robust_meme_hate_detection.runtime import build_loader, get_device, make_transforms, predict_probs
from robust_meme_hate_detection.seeding import seed_everything

LEVELS = ("low", "medium", "high")


def _seed_for(example_id: str, cell: str) -> int:
    """Deterministic SHA-256-keyed per-(example, cell) seed."""
    return int(hashlib.sha256(f"{example_id}:{cell}".encode()).hexdigest(), 16) % (2**31)


class _PerturbedDataset(Dataset):
    """Records with an optional per-example text and/or image perturbation applied."""

    def __init__(
        self,
        records: list[MemeRecord],
        image_tfm: Callable,
        text_tfm: Callable,
        *,
        cell: str,
        text_pert: Optional[Callable[[str, int], str]] = None,
        image_pert: Optional[Callable[[Image.Image, int], Image.Image]] = None,
    ) -> None:
        self.records = records
        self.image_tfm = image_tfm
        self.text_tfm = text_tfm
        self.cell = cell
        self.text_pert = text_pert
        self.image_pert = image_pert

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        seed = _seed_for(rec.id, self.cell)
        img = Image.open(rec.image_path).convert("RGB")
        text = rec.text
        if self.image_pert is not None:
            img = self.image_pert(img, seed)
        if self.text_pert is not None:
            text = self.text_pert(text, seed)
        return {
            "id": rec.id,
            "image": self.image_tfm(img),
            "text": self.text_tfm(text),
            "label": -1 if rec.label is None else int(rec.label),
        }


def _loader(ds: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ----------------------------------------------------------------------- suites


def suite_clean(model, records, image_tfm, text_tfm, device, args) -> dict[str, Any]:
    loader = build_loader(records, image_tfm, text_tfm, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, keep_raw=False)
    probs, labels, _ = predict_probs(model, loader, device)
    return classification_metrics(probs, labels).to_dict()


def suite_modality(model, records, image_tfm, text_tfm, device, args) -> dict[str, Any]:
    loader = build_loader(records, image_tfm, text_tfm, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, keep_raw=False)
    out = {}
    for modality in ("multimodal", "image", "text"):
        probs, labels, _ = predict_probs(model, loader, device, modality=modality)
        out[modality] = classification_metrics(probs, labels).to_dict()
    return out


def suite_perturbed(model, records, image_tfm, text_tfm, device, args) -> dict[str, Any]:
    # Clean reference.
    clean_ds = _PerturbedDataset(records, image_tfm, text_tfm, cell="clean")
    clean_probs, labels, ids = predict_probs(model, _loader(clean_ds, args.batch_size, args.num_workers), device)
    clean = classification_metrics(clean_probs, labels)
    clean_by_id = dict(zip(ids, clean_probs))

    cells: dict[str, Any] = {}
    survive = {i: True for i in ids}  # survives = correct at tau=0.5 in every cell

    def run_cell(name: str, *, text_pert=None, image_pert=None) -> None:
        ds = _PerturbedDataset(records, image_tfm, text_tfm, cell=name, text_pert=text_pert, image_pert=image_pert)
        probs, ys, cids = predict_probs(model, _loader(ds, args.batch_size, args.num_workers), device)
        m = classification_metrics(probs, ys)
        ref = [clean_by_id[i] for i in cids]
        cells[name] = {
            "auroc": m.auroc,
            "delta_auroc": clean.auroc - m.auroc,
            "asr": attack_success_rate(ref, probs, ys),
        }
        for i, p, y in zip(cids, probs, ys):
            if (1 if p >= 0.5 else 0) != y:
                survive[i] = False

    level = args.severity
    for mode in TEXT_MODES:
        run_cell(f"text/{mode}/{level}", text_pert=_text_factory(mode, level))
    for mode in IMAGE_MODES:
        run_cell(f"image/{mode}/{level}", image_pert=_image_factory(mode, level))
    # Composite cells (the realistic multi-edit user).
    run_cell("composite/2text/" + level, text_pert=_compose_text(("leetspeak", "char_deletion"), level))
    run_cell("composite/text_image/" + level,
             text_pert=_text_factory("leetspeak", level), image_pert=_image_factory("compression", level))

    worst_text = max((c["delta_auroc"] for n, c in cells.items() if n.startswith("text/")), default=float("nan"))
    return {
        "clean": clean.to_dict(),
        "severity": level,
        "cells": cells,
        "worst_text_delta_auroc": worst_text,
        "robust_count": sum(survive.values()),
        "n": len(ids),
    }


def suite_whitebox(model, records, image_tfm, text_tfm, device, args) -> dict[str, Any]:
    loader = build_loader(records, image_tfm, text_tfm, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, keep_raw=False)
    clean_probs, labels, _ = predict_probs(model, loader, device)
    clean = classification_metrics(clean_probs, labels)

    out: dict[str, Any] = {"clean": clean.to_dict(), "attacks": {}}
    for eps_num in args.eps:
        eps = eps_num / 255.0
        for name, gen in (
            (f"fgsm_eps{eps_num}", lambda im, tk, y: fgsm_image(model, im, tk, y, eps)),
            (f"pgd_eps{eps_num}", lambda im, tk, y: pgd_image(model, im, tk, y, eps, 0.25 * eps, args.pgd_steps)),
        ):
            probs, ys = _attacked_probs(model, loader, device, gen)
            m = classification_metrics(probs, ys)
            out["attacks"][name] = {
                "auroc": m.auroc,
                "delta_auroc": clean.auroc - m.auroc,
                "asr": attack_success_rate(clean_probs, probs, ys),
            }
    return out


def _attacked_probs(model, loader, device, gen) -> tuple[list[float], list[int]]:
    probs: list[float] = []
    labels: list[int] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["text"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).float()
        with torch.enable_grad():
            adv = gen(images, tokens, y)
        with torch.no_grad():
            probs.extend(torch.sigmoid(model(adv, tokens).float()).cpu().tolist())
        labels.extend(int(x) for x in batch["label"].tolist())
    return probs, labels


# ----------------------------------------------------- perturbation factories


def _text_factory(mode: str, level: str) -> Callable[[str, int], str]:
    return lambda text, seed: TextPerturbation.from_preset(mode, level, seed=seed)(text)


def _image_factory(mode: str, level: str) -> Callable[[Image.Image, int], Image.Image]:
    return lambda img, seed: ImagePerturbation.from_preset(mode, level, seed=seed)(img)


def _compose_text(modes: tuple[str, ...], level: str) -> Callable[[str, int], str]:
    def apply(text: str, seed: int) -> str:
        for k, mode in enumerate(modes):
            text = TextPerturbation.from_preset(mode, level, seed=seed + k)(text)
        return text

    return apply


SUITES = {
    "clean": suite_clean,
    "modality": suite_modality,
    "perturbed": suite_perturbed,
    "whitebox": suite_whitebox,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--suite", required=True, choices=list(SUITES))
    parser.add_argument("--dataset-root", required=True, help="Directory holding the split JSONL + images.")
    parser.add_argument("--split", default="test_unseen")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="Subsample for a quick check (0 = all).")
    parser.add_argument("--severity", default="medium", choices=list(LEVELS), help="perturbed suite severity.")
    parser.add_argument("--eps", type=int, nargs="+", default=[1, 2, 4, 8], help="whitebox eps numerators (/255).")
    parser.add_argument("--pgd-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    model = load_classifier(args.ckpt, device)
    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("config", {})
    image_tfm, text_tfm = make_transforms(cfg.get("arch", "ViT-B-32"))

    records = load_records(args.dataset_root, args.split)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"suite={args.suite}  split={args.split}  n={len(records)}  device={device}", flush=True)

    result = SUITES[args.suite](model, records, image_tfm, text_tfm, device, args)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
