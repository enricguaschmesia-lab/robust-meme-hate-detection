"""Train the CLIP-fusion classifier -- one config-driven entry point for every
recipe in the report.

The ``robust:`` block of the YAML config selects the regime (no code change):

* **clean baseline** / **unimodal baselines** -- no ``robust`` block; the
  ``model.modality`` field (``multimodal`` | ``image`` | ``text``) picks the
  forward. Trains BCE only.
* **naturalistic** (``robust.naturalistic.enabled``) -- clean + an on-the-fly
  perturbed view with the consistency loss and optional text-modality dropout
  (the report's KLDrop recipe).
* **adversarial** (``robust.adversarial.enabled``) -- clean + a PGD view
  (Madry-style; the config also unfreezes the vision encoder).
* **combined** -- both blocks enabled.

Usage::

    python train.py --config configs/naturalistic.yaml --seed 0 --out runs/nat-seed0

Runs on GPU if available, otherwise CPU. Writes ``<out>/best.pt`` (full
checkpoint) and ``<out>/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import torch

from robust_meme_hate_detection.attacks import pgd_image
from robust_meme_hate_detection.losses import robust_loss
from robust_meme_hate_detection.metrics import classification_metrics
from robust_meme_hate_detection.model import CLIPHateMemeClassifier, ModelConfig
from robust_meme_hate_detection.augment import RobustAugmenter
from robust_meme_hate_detection.runtime import (
    FORWARD_FNS,
    build_loader,
    build_records,
    cosine_with_warmup,
    fuse_with_dropout,
    get_device,
    load_yaml,
    make_transforms,
    predict_probs,
)
from robust_meme_hate_detection.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Recipe YAML (see configs/).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output directory for checkpoint + metrics.")
    parser.add_argument("--init-ckpt", default=None, help="Optional checkpoint to warm-start from.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    modality = model_cfg.get("modality", "multimodal")

    robust = cfg.get("robust", {})
    nat = robust.get("naturalistic", {})
    adv = robust.get("adversarial", {})
    nat_on = bool(nat.get("enabled", False))
    adv_on = bool(adv.get("enabled", False))
    if (nat_on or adv_on) and modality != "multimodal":
        raise ValueError("Robust recipes require model.modality == 'multimodal'.")

    # -------------------------------------------------------------------- data
    image_tfm, text_tfm = make_transforms(model_cfg["arch"])
    train_records = build_records(data_cfg, "train", restrict_train_split=True)
    val_records = build_records(data_cfg, data_cfg.get("val_split", "dev"))
    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(train_cfg["batch_size"])

    # The raw PIL image / caption are only needed to build the naturalistic view.
    train_loader = build_loader(
        train_records, image_tfm, text_tfm,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, keep_raw=nat_on,
    )
    val_loader = build_loader(
        val_records, image_tfm, text_tfm,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, keep_raw=False,
    )

    # ------------------------------------------------------------------- model
    config = ModelConfig(
        arch=model_cfg["arch"],
        pretrained=model_cfg.get("pretrained", "laion2b_s34b_b79k"),
        freeze_encoders=bool(model_cfg.get("freeze_encoders", True)),
        freeze_image_encoder=model_cfg.get("freeze_image_encoder"),
        freeze_text_encoder=model_cfg.get("freeze_text_encoder"),
        head_hidden=int(model_cfg.get("head_hidden", 512)),
        head_dropout=float(model_cfg.get("head_dropout", 0.2)),
    )
    model = CLIPHateMemeClassifier(config).to(device)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"], strict=False)
        print(f"Warm-started from {args.init_ckpt}", flush=True)

    pos = sum(r.label == 1 for r in train_records)
    neg = sum(r.label == 0 for r in train_records)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device) if train_cfg.get("pos_weight_auto", True) else None

    # Two LR groups when an encoder is unfrozen (adversarial): head + encoder.
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.") and p.requires_grad]
    enc_params = [p for n, p in model.named_parameters() if not n.startswith("head.") and p.requires_grad]
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": float(train_cfg["lr_head"])}]
    if enc_params:
        groups.append({"params": enc_params, "lr": float(train_cfg.get("lr_vision", 5e-5))})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(train_cfg.get("weight_decay", 1e-2)))

    epochs = int(train_cfg["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    scheduler = cosine_with_warmup(optimizer, total_steps, int(float(train_cfg.get("warmup_ratio", 0.1)) * total_steps))

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    # ------------------------------------------------------------ robust setup
    clean_weight = float(robust.get("clean_weight", 1.0))
    alpha, beta, gamma = (float(nat.get(k, d)) for k, d in (("alpha", 1.0), ("beta", 0.0), ("gamma", 0.0)))
    p_drop_text = float(nat.get("modality_dropout_text", 0.0))
    augmenter = _build_augmenter(nat, args.seed) if nat_on else None

    adv_eps = int(adv.get("adv_eps_num", 8)) / 255.0
    adv_steps = int(adv.get("pgd_steps", 10))
    adv_alpha = float(adv.get("pgd_alpha_frac", 0.25)) * adv_eps
    adv_bce = float(adv.get("weight_bce", 1.0))
    adv_kl = float(adv.get("weight_kl", 0.0))
    select_metric = str(train_cfg.get("select_metric", "clean_macro_f1"))

    print(
        f"device={device}  modality={modality}  naturalistic={nat_on}  adversarial={adv_on}  "
        f"trainable={model.trainable_parameter_count():,}",
        flush=True,
    )

    # -------------------------------------------------------------- train loop
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    patience = int(train_cfg.get("early_stop_patience", 3))
    stale = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            drop_text = None
            if p_drop_text > 0.0:
                drop_text = (torch.rand(images.shape[0], device=device) < p_drop_text).view(-1, 1)

            pert_img = pert_tok = None
            if nat_on:
                pert_img, pert_tok = augmenter(batch["pil_image"], batch["raw_text"], image_tfm=image_tfm, text_tfm=text_tfm)
                pert_img, pert_tok = pert_img.to(device), pert_tok.to(device)

            adv_img = None
            if adv_on:
                adv_img = pgd_image(model, images, tokens, labels, adv_eps, adv_alpha, adv_steps)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else _nullcontext():
                parts = _loss_for_batch(
                    model, modality, images, tokens, labels, drop_text,
                    pert_img, pert_tok, adv_img,
                    alpha=alpha, beta=beta, gamma=gamma,
                    adv_bce=adv_bce, adv_kl=adv_kl, clean_weight=clean_weight, pos_weight=pos_weight,
                )
            parts["total"].backward()
            optimizer.step()
            scheduler.step()
            running += float(parts["total"].detach())

        clean_val = _evaluate(model, val_loader, device, modality)
        adv_val = _evaluate_adv(model, val_loader, device, adv_eps, adv_alpha, adv_steps) if adv_on else None
        score = _select_score(select_metric, clean_val, adv_val)
        print(
            f"epoch={epoch}  loss={running / steps_per_epoch:.4f}  "
            f"val_f1={clean_val['macro_f1']:.4f}  val_auroc={clean_val['auroc']:.4f}  "
            + (f"adv_f1={adv_val['macro_f1']:.4f}  " if adv_val else "")
            + f"select={score:.4f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_metrics = {"val": clean_val, "adv_val": adv_val, "best_epoch": epoch, "select_score": score}
            torch.save(
                {"model_state": model.state_dict(), "config": _config_dict(config),
                 "modality": modality, "seed": args.seed, "epoch": epoch, "val_metrics": clean_val},
                out_dir / "best.pt",
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    best_metrics["elapsed_seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(f"Best select={best_score:.4f}; checkpoint at {out_dir / 'best.pt'}", flush=True)


# --------------------------------------------------------------------- helpers


def _build_augmenter(nat: dict[str, Any], seed: int) -> RobustAugmenter:
    return RobustAugmenter(
        seed=int(nat.get("augmenter_seed", seed)),
        text_fraction=float(nat.get("text_fraction", 0.5)),
        image_fraction=float(nat.get("image_fraction", 0.3)),
        both_fraction=float(nat.get("both_fraction", 0.2)),
        text_attacks=nat.get("text_attacks", ("leetspeak", "char_deletion", "char_swap", "censoring", "keyboard_typo")),
        image_attacks=nat.get("image_attacks", ("gaussian_noise", "blur", "compression", "brightness_down", "occlusion", "typographic")),
        severity_weights_text=nat.get("severity_weights_text", (1.0, 2.0, 2.0)),
        severity_weights_image=nat.get("severity_weights_image", (1.0, 1.0, 1.0)),
    )


def _loss_for_batch(
    model, modality, images, tokens, labels, drop_text,
    pert_img, pert_tok, adv_img, *, alpha, beta, gamma, adv_bce, adv_kl, clean_weight, pos_weight,
) -> dict[str, torch.Tensor]:
    """Assemble the per-batch robust loss (terms absent for baselines collapse to 0)."""
    if modality == "multimodal":
        logit_clean = fuse_with_dropout(model, images, tokens, drop_text)
    else:
        logit_clean = FORWARD_FNS[modality](model, images, tokens)

    logit_pert = image_clean = image_pert = logit_adv = None
    if pert_img is not None:
        logit_pert = fuse_with_dropout(model, pert_img, pert_tok, drop_text)
        if gamma > 0.0:
            image_clean = model.forward_image_only(images)
            image_pert = model.forward_image_only(pert_img)
    if adv_img is not None:
        logit_adv = fuse_with_dropout(model, adv_img, tokens, drop_text)

    return robust_loss(
        logit_clean=logit_clean, labels=labels,
        logit_pert=logit_pert, alpha=alpha, beta=beta, gamma=gamma,
        image_logit_clean=image_clean, image_logit_pert=image_pert,
        logit_adv=logit_adv, delta=adv_bce, epsilon_kl=adv_kl,
        clean_weight=clean_weight, pos_weight=pos_weight,
    )


def _evaluate(model, loader, device, modality) -> dict[str, float]:
    probs, labels, _ = predict_probs(model, loader, device, modality=modality)
    return classification_metrics(probs, labels).to_dict()


def _evaluate_adv(model, loader, device, eps, alpha, steps, max_batches: int = 8) -> dict[str, float]:
    """PGD dev evaluation, capped at ``max_batches`` to bound cost during training."""
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["text"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).float()
        with torch.enable_grad():
            adv = pgd_image(model, images, tokens, y, eps, alpha, steps)
        with torch.no_grad():
            probs.extend(torch.sigmoid(model(adv, tokens).float()).cpu().tolist())
        labels.extend(int(x) for x in batch["label"].tolist())
    return classification_metrics(probs, labels).to_dict()


def _select_score(metric: str, clean_val: dict[str, float], adv_val: Optional[dict[str, float]]) -> float:
    if adv_val is None:
        return clean_val["macro_f1"]
    if metric == "adv_macro_f1":
        return adv_val["macro_f1"]
    if metric == "mean_macro_f1":
        return 0.5 * (clean_val["macro_f1"] + adv_val["macro_f1"])
    return clean_val["macro_f1"]


def _config_dict(config: ModelConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(config)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
