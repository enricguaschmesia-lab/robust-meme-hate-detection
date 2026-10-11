"""Stage-1 training: frozen CLIP encoders, train only the fusion head.

Reads a YAML config (``configs/stage1.yaml`` by default), trains the head, runs
early stopping on dev macro F1, persists the best checkpoint, and writes a
``metrics.json`` summary suitable for ``cluster.sh summarize-results``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from robust_meme_hate_detection.utils.seeding import seed_everything


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve_pos_weight(records, *, auto: bool) -> float:
    if not auto:
        return 1.0
    pos = sum(1 for r in records if r.label == 1)
    neg = sum(1 for r in records if r.label == 0)
    return float(neg / max(pos, 1))


def _cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    """Local cosine-with-warmup scheduler (no transformers dependency)."""
    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:  # noqa: C901 — single entry point, kept linear for readability
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output directory (created if missing)")
    parser.add_argument(
        "--modality",
        choices=("multimodal", "text", "image"),
        default=None,
        help="Override model.modality from the config.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    seed_everything(seed)

    modality = args.modality or str(cfg.get("model", {}).get("modality", "multimodal"))
    if modality not in ("multimodal", "text", "image"):
        raise ValueError(f"Unknown modality: {modality}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    # Imports deferred so synthetic-tensor smoke can exercise other modules
    # without paying the torch import cost up front in unrelated tools.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from robust_meme_hate_detection.data.hateful_memes import (
        HatefulMemesDataset,
        load_hateful_memes_records,
    )
    from robust_meme_hate_detection.data.splits import load_train_split
    from robust_meme_hate_detection.data.transforms import ClipImage01Transform, ClipTokenize
    from robust_meme_hate_detection.eval.metrics import classification_metrics
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.utils.logging import JsonRunLogger

    # ------------------------------------------------------------------- data
    data_cfg = cfg["data"]
    dataset_root = Path(data_cfg["dataset_root"])
    image_tfm = ClipImage01Transform(size=224)
    text_tfm = ClipTokenize(arch=cfg["model"]["arch"])

    train_records = load_hateful_memes_records(dataset_root, "train")
    if "train_split_file" in data_cfg and data_cfg["train_split_file"]:
        split = load_train_split(data_cfg["train_split_file"])
        keep = split.train_id_set
        train_records = [r for r in train_records if str(r.id) in keep]
    val_records = load_hateful_memes_records(dataset_root, data_cfg.get("val_split", "dev"))

    train_ds = _RecordsDataset(train_records, dataset_root, image_tfm, text_tfm)
    val_ds = _RecordsDataset(val_records, dataset_root, image_tfm, text_tfm)

    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------ model
    model_cfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPHateMemeClassifier(
        arch=model_cfg["arch"],
        pretrained=model_cfg.get("pretrained", "laion2b_s34b_b79k"),
        freeze_encoders=bool(model_cfg.get("freeze_encoders", True)),
        head_hidden=int(model_cfg.get("head_hidden", 512)),
        head_dropout=float(model_cfg.get("head_dropout", 0.2)),
    ).to(device)

    pos_weight_val = _resolve_pos_weight(train_records, auto=bool(train_cfg.get("pos_weight_auto", True)))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["lr_head"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    epochs = int(train_cfg["epochs"])
    total_steps = max(1, epochs * max(1, len(train_loader)))
    warmup_steps = int(float(train_cfg.get("warmup_ratio", 0.1)) * total_steps)
    scheduler = _cosine_with_warmup(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    amp_dtype_name = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype_name, torch.bfloat16)
    use_amp = device.type == "cuda"

    logger = JsonRunLogger(out_dir)
    logger.write_metrics({
        "config_path": str(args.config),
        "seed": seed,
        "modality": modality,
        "device": str(device),
        "pos_weight": pos_weight_val,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "trainable_params": model.trainable_parameter_count(),
        "total_params": model.total_parameter_count(),
        "amp_dtype": amp_dtype_name,
    })

    # ------------------------------------------------------------------ loop
    best_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    patience = int(train_cfg.get("early_stop_patience", 3))
    epochs_since_improve = 0
    global_step = 0
    epoch = -1
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logit = _forward(model, modality, images, tokens)
                    loss = loss_fn(logit, labels)
            else:
                logit = _forward(model, modality, images, tokens)
                loss = loss_fn(logit, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.detach())
            n_batches += 1
            global_step += 1
            if global_step % 50 == 0:
                logger.log_step({"epoch": epoch, "step": global_step, "train_loss": float(loss.detach())})

        train_loss_epoch = epoch_loss / max(1, n_batches)
        val = _evaluate(model, val_loader, device, use_amp, amp_dtype, modality=modality)
        logger.log_step({"epoch": epoch, "train_loss_epoch": train_loss_epoch, **{f"val_{k}": v for k, v in val.items()}})
        print(
            f"epoch={epoch}  train_loss={train_loss_epoch:.4f}  "
            f"val_macro_f1={val['macro_f1']:.4f}  val_auroc={val['auroc']:.4f}",
            flush=True,
        )

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1 = val["macro_f1"]
            best_metrics = {f"val_{k}": v for k, v in val.items()}
            best_metrics["best_epoch"] = epoch
            best_metrics["train_loss_at_best"] = train_loss_epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "seed": seed,
                "epoch": epoch,
                "val_metrics": val,
            }, ckpt_dir / "best.pt")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch={epoch} (no improvement for {patience} epochs)", flush=True)
                break

    elapsed = time.time() - t0
    logger.write_metrics({**best_metrics, "epochs_run": epoch + 1, "elapsed_seconds": elapsed})
    print(f"Best dev macro_f1={best_f1:.4f}, AUROC={best_metrics.get('val_auroc', float('nan')):.4f}, elapsed {elapsed:.1f}s", flush=True)


def _forward(model, modality: str, images, tokens):
    if modality == "multimodal":
        return model(images, tokens)
    if modality == "text":
        return model.forward_text_only(tokens)
    if modality == "image":
        return model.forward_image_only(images)
    raise ValueError(f"Unknown modality: {modality}")


def _evaluate(model, loader, device, use_amp, amp_dtype, *, modality: str = "multimodal") -> dict[str, float]:
    import torch

    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            labels = batch["label"]
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logit = _forward(model, modality, images, tokens)
            else:
                logit = _forward(model, modality, images, tokens)
            probs = torch.sigmoid(logit.float()).detach().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(int(x) for x in labels.tolist())
    from robust_meme_hate_detection.eval.metrics import classification_metrics
    metrics = classification_metrics(all_probs, all_labels, threshold=0.5)
    return metrics.to_dict()


class _RecordsDataset:
    """Tiny adapter wrapping HatefulMemesDataset to filter by record list.

    We do not modify the existing loader; instead we instantiate one over the
    full split and intersect by ID at __getitem__ time. This keeps Phase 3
    surgical.
    """

    def __init__(self, records, dataset_root, image_transform, text_transform) -> None:
        from robust_meme_hate_detection.data.hateful_memes import HatefulMemesDataset
        # Build a backing dataset that does NOT filter by ID; we'll iterate by index.
        self._records = list(records)
        self._dataset_root = dataset_root
        self._image_transform = image_transform
        self._text_transform = text_transform
        # Sanity: ensure all referenced images exist (mirrors the loader's check).
        missing = [r.image_path for r in self._records if not r.image_path.exists()]
        if missing:
            sample = ", ".join(str(p) for p in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} images missing for {len(self._records)} records (sample: {sample})"
            )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        rec = self._records[index]
        img = Image.open(rec.image_path).convert("RGB")
        image = self._image_transform(img)
        tokens = self._text_transform(rec.text)
        label = -1 if rec.label is None else int(rec.label)
        return {"id": rec.id, "image": image, "text": tokens, "label": label}


if __name__ == "__main__":
    main()
