"""Stage-1 robust training: clean + perturbed views with optional KL terms.

Sibling of ``train/stage1.py``. Differs in three places:

1. Dataset returns the *raw* PIL image and caption string alongside the
   pre-processed ``(image_tensor, token_ids)`` so the augmenter can run
   on the raw side and re-apply ``image_tfm`` / ``text_tfm`` for the
   perturbed view.
2. Per-batch step runs two forwards (clean + perturbed) plus two
   image-only forwards (clean + perturbed) and combines them via
   ``train._robust_loss.robust_loss``.
3. Per-step log records each loss term individually so we can sanity-check
   that the KL terms are non-zero and shrinking through training.

Validation uses the clean dev forward only (unchanged from stage1) so the
"best macro-F1" metric remains comparable across the clean and robust
runs.
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
    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    seed_everything(seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    from robust_meme_hate_detection.data.hateful_memes import (
        load_hateful_memes_records,
    )
    from robust_meme_hate_detection.data.splits import load_train_split
    from robust_meme_hate_detection.data.transforms import (
        ClipImage01Transform,
        ClipTokenize,
    )
    from robust_meme_hate_detection.eval.metrics import classification_metrics
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.train._robust_augmenter import RobustAugmenter
    from robust_meme_hate_detection.train._robust_loss import robust_loss
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

    train_ds = _RawAndProcessedDataset(train_records, image_tfm, text_tfm)
    val_ds = _ProcessedOnlyDataset(val_records, image_tfm, text_tfm)

    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=_raw_collate,
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

    # ----------------------------------------------------------- robust block
    robust_cfg = cfg.get("robust", {})
    alpha = float(robust_cfg.get("alpha", 1.0))
    beta = float(robust_cfg.get("beta", 0.0))
    gamma = float(robust_cfg.get("gamma", 0.0))
    # Phase 5b: per-example modality dropout to force the head to use each branch
    # in isolation some fraction of the time. Defaults to 0 so existing configs
    # are unchanged.
    modality_dropout_text = float(robust_cfg.get("modality_dropout_text", 0.0))
    modality_dropout_image = float(robust_cfg.get("modality_dropout_image", 0.0))
    if not 0.0 <= modality_dropout_text <= 1.0:
        raise ValueError(f"modality_dropout_text must be in [0,1], got {modality_dropout_text}")
    if not 0.0 <= modality_dropout_image <= 1.0:
        raise ValueError(f"modality_dropout_image must be in [0,1], got {modality_dropout_image}")
    augmenter = RobustAugmenter(
        seed=int(robust_cfg.get("augmenter_seed", seed)),
        text_fraction=float(robust_cfg.get("text_fraction", 0.5)),
        image_fraction=float(robust_cfg.get("image_fraction", 0.3)),
        both_fraction=float(robust_cfg.get("both_fraction", 0.2)),
        text_attacks=robust_cfg.get(
            "text_attacks",
            ("leetspeak", "char_deletion", "char_swap", "censoring", "keyboard_typo"),
        ),
        image_attacks=robust_cfg.get(
            "image_attacks",
            ("gaussian_noise", "blur", "compression", "brightness_down", "occlusion", "typographic"),
        ),
        severity_weights_text=robust_cfg.get("severity_weights_text", (1.0, 2.0, 2.0)),
        severity_weights_image=robust_cfg.get("severity_weights_image", (1.0, 1.0, 1.0)),
    )

    logger = JsonRunLogger(out_dir)
    logger.write_metrics({
        "config_path": str(args.config),
        "seed": seed,
        "modality": "multimodal",
        "device": str(device),
        "pos_weight": pos_weight_val,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "trainable_params": model.trainable_parameter_count(),
        "total_params": model.total_parameter_count(),
        "amp_dtype": amp_dtype_name,
        "robust": {
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "modality_dropout_text": modality_dropout_text,
            "modality_dropout_image": modality_dropout_image,
        },
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
        sum_total = 0.0
        sum_ce_clean = 0.0
        sum_ce_pert = 0.0
        sum_kl_full = 0.0
        sum_kl_image = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()
            pil_images = batch["pil_image"]   # list[PIL.Image] (length = batch_size)
            raw_texts = batch["raw_text"]     # list[str]

            pert_img_t, pert_tok_t = augmenter(
                pil_images, raw_texts, image_tfm=image_tfm, text_tfm=text_tfm,
            )
            pert_img_t = pert_img_t.to(device, non_blocking=True)
            pert_tok_t = pert_tok_t.to(device, non_blocking=True)

            # Per-example modality dropout masks (same for clean + perturbed views
            # so KL terms compare matched effective inputs).
            if modality_dropout_text > 0.0 or modality_dropout_image > 0.0:
                drop_text = (torch.rand(images.shape[0], device=device)
                             < modality_dropout_text).view(-1, 1)
                drop_image = (torch.rand(images.shape[0], device=device)
                              < modality_dropout_image).view(-1, 1)
            else:
                drop_text = None
                drop_image = None

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logit_clean = _fuse_with_dropout(model, images, tokens, drop_text, drop_image)
                    logit_pert = _fuse_with_dropout(model, pert_img_t, pert_tok_t, drop_text, drop_image)
                    if gamma > 0.0:
                        image_logit_clean = model.forward_image_only(images)
                        image_logit_pert = model.forward_image_only(pert_img_t)
                    else:
                        image_logit_clean = None
                        image_logit_pert = None
                    parts = robust_loss(
                        logit_clean=logit_clean, logit_pert=logit_pert, labels=labels,
                        alpha=alpha, beta=beta, gamma=gamma,
                        image_logit_clean=image_logit_clean,
                        image_logit_pert=image_logit_pert,
                        pos_weight=pos_weight,
                    )
            else:
                logit_clean = _fuse_with_dropout(model, images, tokens, drop_text, drop_image)
                logit_pert = _fuse_with_dropout(model, pert_img_t, pert_tok_t, drop_text, drop_image)
                image_logit_clean = model.forward_image_only(images) if gamma > 0 else None
                image_logit_pert = model.forward_image_only(pert_img_t) if gamma > 0 else None
                parts = robust_loss(
                    logit_clean=logit_clean, logit_pert=logit_pert, labels=labels,
                    alpha=alpha, beta=beta, gamma=gamma,
                    image_logit_clean=image_logit_clean,
                    image_logit_pert=image_logit_pert,
                    pos_weight=pos_weight,
                )
            parts["total"].backward()
            optimizer.step()
            scheduler.step()

            sum_total += float(parts["total"].detach())
            sum_ce_clean += float(parts["ce_clean"].detach())
            sum_ce_pert += float(parts["ce_pert"].detach())
            sum_kl_full += float(parts["kl_full"].detach())
            sum_kl_image += float(parts["kl_image"].detach())
            n_batches += 1
            global_step += 1
            if global_step % 50 == 0:
                logger.log_step({
                    "epoch": epoch, "step": global_step,
                    "train_loss": float(parts["total"].detach()),
                    "ce_clean": float(parts["ce_clean"].detach()),
                    "ce_pert": float(parts["ce_pert"].detach()),
                    "kl_full": float(parts["kl_full"].detach()),
                    "kl_image": float(parts["kl_image"].detach()),
                })

        train_loss_epoch = sum_total / max(1, n_batches)
        val = _evaluate(model, val_loader, device, use_amp, amp_dtype)
        logger.log_step({
            "epoch": epoch,
            "train_loss_epoch": train_loss_epoch,
            "ce_clean_epoch": sum_ce_clean / max(1, n_batches),
            "ce_pert_epoch": sum_ce_pert / max(1, n_batches),
            "kl_full_epoch": sum_kl_full / max(1, n_batches),
            "kl_image_epoch": sum_kl_image / max(1, n_batches),
            **{f"val_{k}": v for k, v in val.items()},
        })
        print(
            f"epoch={epoch}  train_loss={train_loss_epoch:.4f}  "
            f"ce_c={sum_ce_clean/max(1,n_batches):.4f}  "
            f"ce_p={sum_ce_pert/max(1,n_batches):.4f}  "
            f"kl_f={sum_kl_full/max(1,n_batches):.4f}  "
            f"kl_i={sum_kl_image/max(1,n_batches):.4f}  "
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
    print(
        f"Best dev macro_f1={best_f1:.4f}, "
        f"AUROC={best_metrics.get('val_auroc', float('nan')):.4f}, "
        f"elapsed {elapsed:.1f}s",
        flush=True,
    )


def _fuse_with_dropout(model, images, tokens, drop_text, drop_image):
    """Forward through encoders, optionally zero each branch's embedding per-example, then fuse.

    Equivalent to ``model(images, tokens)`` when both masks are ``None``.
    When ``drop_text[b]`` is True, the *text embedding* for example b is zeroed
    before the fusion head — this mirrors ``forward_text_only`` /
    ``forward_image_only`` (which also operate at the embedding level) and
    avoids feeding pad-only tokens through the text encoder.
    """
    import torch

    v = model.encode_image(images).float()
    t = model.encode_text(tokens).float()
    if drop_text is not None:
        t = torch.where(drop_text, torch.zeros_like(t), t)
    if drop_image is not None:
        v = torch.where(drop_image, torch.zeros_like(v), v)
    return model.head(t, v)


def _evaluate(model, loader, device, use_amp, amp_dtype) -> dict[str, float]:
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
                    logit = model(images, tokens)
            else:
                logit = model(images, tokens)
            probs = torch.sigmoid(logit.float()).detach().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(int(x) for x in labels.tolist())
    from robust_meme_hate_detection.eval.metrics import classification_metrics
    metrics = classification_metrics(all_probs, all_labels, threshold=0.5)
    return metrics.to_dict()


class _RawAndProcessedDataset:
    """Dataset that yields both raw (PIL, str) and pre-processed (tensor, ids)."""

    def __init__(self, records, image_transform, text_transform) -> None:
        self._records = list(records)
        self._image_transform = image_transform
        self._text_transform = text_transform
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
        image_tensor = self._image_transform(img)
        tokens = self._text_transform(rec.text)
        label = -1 if rec.label is None else int(rec.label)
        return {
            "id": rec.id,
            "image": image_tensor,
            "text": tokens,
            "label": label,
            "pil_image": img,
            "raw_text": rec.text,
        }


class _ProcessedOnlyDataset:
    """Same as ``train.stage1._RecordsDataset`` — used for the val loader."""

    def __init__(self, records, image_transform, text_transform) -> None:
        self._records = list(records)
        self._image_transform = image_transform
        self._text_transform = text_transform
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


def _raw_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that stacks tensors but leaves PIL images / strings as lists."""
    import torch

    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = torch.stack([b["text"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    ids = [b["id"] for b in batch]
    pil_images = [b["pil_image"] for b in batch]
    raw_texts = [b["raw_text"] for b in batch]
    return {
        "id": ids, "image": images, "text": texts, "label": labels,
        "pil_image": pil_images, "raw_text": raw_texts,
    }


if __name__ == "__main__":
    main()
