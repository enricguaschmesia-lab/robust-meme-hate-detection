"""Unified Stage-1 robust trainer: naturalistic and/or adversarial views.

A single trainer that subsumes three regimes, selected purely by the
``robust:`` block of the config (no code change to switch modes):

* **naturalistic** — ``robust.naturalistic.enabled: true`` only. Clean + an
  on-the-fly symbolic-perturbation view (text/image/both via ``RobustAugmenter``)
  with the Phase-5 loss ``BCE_clean + α·BCE_pert + β·KL_full + γ·KL_image`` and
  optional per-example modality dropout. Reproduces the ``kldrop`` recipe.
* **adversarial** — ``robust.adversarial.enabled: true`` only. Clean + a PGD
  view (Madry-style ``δ·BCE_adv`` plus optional TRADES ``ε·KL(clean‖adv)``).
* **both** — both sub-blocks enabled. Clean + naturalistic + adversarial views
  combined in one composite loss.

All three share the same dataset, model, optimizer, schedule, logging and
validation, so this is the canonical robust-training entrypoint. The legacy
``train/stage1_robust.py`` (naturalistic-only) and ``train/stage1.py`` are kept
untouched to reproduce previously reported numbers.

Heavy lifting is reused: the dataset/collate/fusion helpers come from
``train.stage1_robust``, the loss from ``train._robust_loss.robust_loss``, the
augmenter from ``train._robust_augmenter.RobustAugmenter``, and PGD from
``attacks.pgd.pgd_image``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

from robust_meme_hate_detection.utils.seeding import seed_everything


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--ckpt", default=None,
        help="Optional checkpoint to initialise model weights from (e.g. a clean stage-1 run).",
    )
    args = parser.parse_args()

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
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.train._robust_augmenter import RobustAugmenter
    from robust_meme_hate_detection.train._robust_loss import robust_loss
    from robust_meme_hate_detection.train.stage1_robust import (
        _ProcessedOnlyDataset,
        _RawAndProcessedDataset,
        _cosine_with_warmup,
        _evaluate,
        _fuse_with_dropout,
        _load_yaml,
        _raw_collate,
        _resolve_pos_weight,
    )
    from robust_meme_hate_detection.utils.logging import JsonRunLogger

    cfg = _load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    seed_everything(seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

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
        freeze_image_encoder=model_cfg.get("freeze_image_encoder"),
        freeze_text_encoder=model_cfg.get("freeze_text_encoder"),
        head_hidden=int(model_cfg.get("head_hidden", 512)),
        head_dropout=float(model_cfg.get("head_dropout", 0.2)),
    ).to(device)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"Loaded initial weights from {args.ckpt}", flush=True)

    pos_weight_val = _resolve_pos_weight(train_records, auto=bool(train_cfg.get("pos_weight_auto", True)))
    pos_weight = torch.tensor([pos_weight_val], device=device)

    # Two param groups when any encoder weight is trainable (e.g. unfrozen vision
    # for adversarial training): head at lr_head, encoder at lr_vision. Falls back
    # to a single group when everything but the head is frozen.
    lr_head = float(train_cfg["lr_head"])
    lr_vision = float(train_cfg.get("lr_vision", 5.0e-5))
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    enc_params = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]
    param_groups: list[dict[str, Any]] = [{"params": head_params, "lr": lr_head}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": lr_vision})
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
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
    clean_weight = float(robust_cfg.get("clean_weight", 1.0))

    nat_cfg = robust_cfg.get("naturalistic", {})
    nat_enabled = bool(nat_cfg.get("enabled", False))
    alpha = float(nat_cfg.get("alpha", 1.0))
    beta = float(nat_cfg.get("beta", 0.0))
    gamma = float(nat_cfg.get("gamma", 0.0))
    modality_dropout_text = float(nat_cfg.get("modality_dropout_text", 0.0))
    modality_dropout_image = float(nat_cfg.get("modality_dropout_image", 0.0))
    if not 0.0 <= modality_dropout_text <= 1.0:
        raise ValueError(f"modality_dropout_text must be in [0,1], got {modality_dropout_text}")
    if not 0.0 <= modality_dropout_image <= 1.0:
        raise ValueError(f"modality_dropout_image must be in [0,1], got {modality_dropout_image}")

    augmenter = None
    if nat_enabled:
        augmenter = RobustAugmenter(
            seed=int(nat_cfg.get("augmenter_seed", seed)),
            text_fraction=float(nat_cfg.get("text_fraction", 0.5)),
            image_fraction=float(nat_cfg.get("image_fraction", 0.3)),
            both_fraction=float(nat_cfg.get("both_fraction", 0.2)),
            text_attacks=nat_cfg.get(
                "text_attacks",
                ("leetspeak", "char_deletion", "char_swap", "censoring", "keyboard_typo"),
            ),
            image_attacks=nat_cfg.get(
                "image_attacks",
                ("gaussian_noise", "blur", "compression", "brightness_down", "occlusion", "typographic"),
            ),
            severity_weights_text=nat_cfg.get("severity_weights_text", (1.0, 2.0, 2.0)),
            severity_weights_image=nat_cfg.get("severity_weights_image", (1.0, 1.0, 1.0)),
        )

    adv_cfg = robust_cfg.get("adversarial", {})
    adv_enabled = bool(adv_cfg.get("enabled", False))
    adv_weight_bce = float(adv_cfg.get("weight_bce", 1.0))
    adv_weight_kl = float(adv_cfg.get("weight_kl", 0.0))
    adv_eps = float(int(adv_cfg.get("adv_eps_num", 8))) / 255.0
    adv_pgd_steps = int(adv_cfg.get("pgd_steps", 10))
    adv_pgd_alpha = float(adv_cfg.get("pgd_alpha_frac", 0.25)) * adv_eps
    adv_norm = str(adv_cfg.get("norm", "linf"))
    adv_random_start = bool(adv_cfg.get("random_start", True))

    if not nat_enabled and not adv_enabled:
        raise ValueError(
            "At least one of robust.naturalistic.enabled / robust.adversarial.enabled must be true"
        )

    select_metric = str(train_cfg.get("select_metric", "clean_macro_f1"))
    eval_adv_max_batches = int(train_cfg.get("eval_adv_max_batches", 8))
    # 0 = full epoch; >0 caps train batches per epoch (used by the smoke config).
    max_train_batches = int(train_cfg.get("max_train_batches", 0))

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
        "select_metric": select_metric,
        "robust": {
            "clean_weight": clean_weight,
            "naturalistic": {
                "enabled": nat_enabled, "alpha": alpha, "beta": beta, "gamma": gamma,
                "modality_dropout_text": modality_dropout_text,
                "modality_dropout_image": modality_dropout_image,
            },
            "adversarial": {
                "enabled": adv_enabled, "weight_bce": adv_weight_bce, "weight_kl": adv_weight_kl,
                "epsilon_over_255": int(adv_cfg.get("adv_eps_num", 8)),
                "pgd_steps": adv_pgd_steps, "pgd_alpha_frac": float(adv_cfg.get("pgd_alpha_frac", 0.25)),
                "norm": adv_norm,
            },
        },
    })

    from robust_meme_hate_detection.attacks.pgd import pgd_image

    # ------------------------------------------------------------------ loop
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    patience = int(train_cfg.get("early_stop_patience", 3))
    epochs_since_improve = 0
    global_step = 0
    epoch = -1
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        sums = {k: 0.0 for k in ("total", "ce_clean", "ce_pert", "kl_full", "kl_image", "ce_adv", "kl_adv")}
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            # Shared per-example modality-dropout masks (clean / pert / adv use
            # the same masks so the KL terms compare matched effective inputs).
            if modality_dropout_text > 0.0 or modality_dropout_image > 0.0:
                drop_text = (torch.rand(images.shape[0], device=device)
                             < modality_dropout_text).view(-1, 1)
                drop_image = (torch.rand(images.shape[0], device=device)
                              < modality_dropout_image).view(-1, 1)
            else:
                drop_text = None
                drop_image = None

            # Naturalistic perturbed view (raw-side, re-tfm'd to clean shapes).
            if nat_enabled:
                pert_img_t, pert_tok_t = augmenter(
                    batch["pil_image"], batch["raw_text"], image_tfm=image_tfm, text_tfm=text_tfm,
                )
                pert_img_t = pert_img_t.to(device, non_blocking=True)
                pert_tok_t = pert_tok_t.to(device, non_blocking=True)

            # Adversarial view (PGD on the clean image tensor) — generated outside
            # autocast, mirroring the Madry adversarial-training recipe.
            if adv_enabled:
                adv_images = pgd_image(
                    model, images, tokens, labels,
                    epsilon=adv_eps, alpha=adv_pgd_alpha, steps=adv_pgd_steps,
                    random_start=adv_random_start, norm=adv_norm,
                )
                model.train()

            optimizer.zero_grad(set_to_none=True)

            def _forward_all():
                logit_clean = _fuse_with_dropout(model, images, tokens, drop_text, drop_image)
                logit_pert = image_logit_clean = image_logit_pert = logit_adv = None
                if nat_enabled:
                    logit_pert = _fuse_with_dropout(model, pert_img_t, pert_tok_t, drop_text, drop_image)
                    if gamma > 0.0:
                        image_logit_clean = model.forward_image_only(images)
                        image_logit_pert = model.forward_image_only(pert_img_t)
                if adv_enabled:
                    logit_adv = _fuse_with_dropout(model, adv_images, tokens, drop_text, drop_image)
                return robust_loss(
                    logit_clean=logit_clean, labels=labels,
                    logit_pert=logit_pert, alpha=alpha, beta=beta, gamma=gamma,
                    image_logit_clean=image_logit_clean, image_logit_pert=image_logit_pert,
                    logit_adv=logit_adv, delta=adv_weight_bce, epsilon_kl=adv_weight_kl,
                    clean_weight=clean_weight, pos_weight=pos_weight,
                )

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    parts = _forward_all()
            else:
                parts = _forward_all()

            parts["total"].backward()
            optimizer.step()
            scheduler.step()

            for k in sums:
                sums[k] += float(parts[k].detach())
            n_batches += 1
            global_step += 1
            if max_train_batches and n_batches >= max_train_batches:
                break
            if global_step % 50 == 0:
                logger.log_step({
                    "epoch": epoch, "step": global_step,
                    "train_loss": float(parts["total"].detach()),
                    **{k: float(parts[k].detach()) for k in
                       ("ce_clean", "ce_pert", "kl_full", "kl_image", "ce_adv", "kl_adv")},
                })

        train_loss_epoch = sums["total"] / max(1, n_batches)
        clean_val = _evaluate(model, val_loader, device, use_amp, amp_dtype)
        adv_val: Optional[dict[str, float]] = None
        if adv_enabled:
            adv_val = _evaluate_adv(
                model, val_loader, device, use_amp, amp_dtype, pgd_image,
                epsilon=adv_eps, alpha=adv_pgd_alpha, steps=adv_pgd_steps,
                norm=adv_norm, max_batches=eval_adv_max_batches,
            )

        clean_f1 = clean_val["macro_f1"]
        adv_f1 = adv_val["macro_f1"] if adv_val is not None else float("nan")
        score = _select_value(select_metric, clean_f1, adv_f1)

        log_row = {
            "epoch": epoch,
            "train_loss_epoch": train_loss_epoch,
            **{f"{k}_epoch": sums[k] / max(1, n_batches) for k in
               ("ce_clean", "ce_pert", "kl_full", "kl_image", "ce_adv", "kl_adv")},
            **{f"val_{k}": v for k, v in clean_val.items()},
        }
        if adv_val is not None:
            log_row.update({f"adv_val_{k}": v for k, v in adv_val.items()})
        logger.log_step(log_row)
        print(
            f"epoch={epoch}  train_loss={train_loss_epoch:.4f}  "
            f"val_macro_f1={clean_f1:.4f}  adv_val_macro_f1={adv_f1:.4f}  "
            f"select({select_metric})={score:.4f}",
            flush=True,
        )

        improved = score > best_score
        if improved:
            best_score = score
            best_metrics = {f"val_{k}": v for k, v in clean_val.items()}
            if adv_val is not None:
                best_metrics.update({f"adv_val_{k}": v for k, v in adv_val.items()})
            best_metrics["best_epoch"] = epoch
            best_metrics["select_metric"] = select_metric
            best_metrics["select_score"] = score
            best_metrics["train_loss_at_best"] = train_loss_epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "seed": seed,
                "epoch": epoch,
                "val_metrics": clean_val,
                "adv_val_metrics": adv_val,
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
        f"Best select({select_metric})={best_score:.4f}, "
        f"clean dev macro_f1={best_metrics.get('val_macro_f1', float('nan')):.4f}, "
        f"elapsed {elapsed:.1f}s",
        flush=True,
    )


def _select_value(metric_name: str, clean_f1: float, adv_f1: float) -> float:
    """Map the configured selection metric to a scalar for best-checkpoint choice."""
    if metric_name == "adv_macro_f1":
        return adv_f1 if adv_f1 == adv_f1 else clean_f1  # NaN-safe fallback
    if metric_name == "mean_macro_f1":
        return 0.5 * (clean_f1 + adv_f1) if adv_f1 == adv_f1 else clean_f1
    return clean_f1


def _evaluate_adv(
    model, loader, device, use_amp, amp_dtype, pgd_image,
    *, epsilon: float, alpha: float, steps: int, norm: str, max_batches: int,
) -> dict[str, float]:
    """PGD dev evaluation, capped at ``max_batches`` batches to bound cost."""
    import torch

    from robust_meme_hate_detection.eval.metrics import classification_metrics

    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["text"].to(device, non_blocking=True)
        labels = batch["label"]
        labels_f = labels.to(device, non_blocking=True).float()
        with torch.enable_grad():
            adv = pgd_image(
                model, images, tokens, labels_f,
                epsilon=epsilon, alpha=alpha, steps=steps,
                random_start=True, norm=norm,
            )
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logit = model(adv, tokens)
            else:
                logit = model(adv, tokens)
        all_probs.extend(torch.sigmoid(logit.float()).detach().cpu().tolist())
        all_labels.extend(int(x) for x in labels.tolist())
        n += 1
        if max_batches and n >= max_batches:
            break
    return classification_metrics(all_probs, all_labels, threshold=0.5).to_dict()


if __name__ == "__main__":
    main()
