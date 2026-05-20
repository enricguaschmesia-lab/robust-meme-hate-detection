"""Stage-1 adversarial training using PGD on the image tensor.

This script follows the project's training conventions and performs per-batch
PGD on the image tensor to produce adversarial examples, then uses those
examples for the training step (Madry-style). It's intentionally minimal —
it reuses the existing dataset, model, logging and evaluation helpers.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import yaml

from robust_meme_hate_detection.utils.seeding import seed_everything


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _load_attack_settings(robust_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    settings_cfg = robust_cfg.get("attack_settings")
    if settings_cfg:
        settings: list[dict[str, Any]] = []
        for entry in settings_cfg:
            if not isinstance(entry, dict):
                raise ValueError("Each robust.attack_settings entry must be a mapping")
            eps_num = entry.get("epsilon_over_255", entry.get("adv_eps_num"))
            if eps_num is None:
                raise ValueError("Each attack setting needs epsilon_over_255 or adv_eps_num")
            settings.append({
                "epsilon_over_255": int(eps_num),
                "epsilon": float(int(eps_num)) / 255.0,
                "pgd_steps": int(entry.get("pgd_steps", robust_cfg.get("pgd_steps", 10))),
                "pgd_alpha_frac": float(entry.get("pgd_alpha_frac", robust_cfg.get("pgd_alpha_frac", 0.25))),
            })
        return settings

    eps_num = int(robust_cfg.get("adv_eps_num", 8))
    return [{
        "epsilon_over_255": eps_num,
        "epsilon": float(eps_num) / 255.0,
        "pgd_steps": int(robust_cfg.get("pgd_steps", 10)),
        "pgd_alpha_frac": float(robust_cfg.get("pgd_alpha_frac", 0.25)),
    }]


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint to initialize model weights from.")
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
    
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.utils.logging import JsonRunLogger
    from robust_meme_hate_detection.attacks.pgd import pgd_image

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

    class _DS:
        def __init__(self, recs):
            self.records = list(recs)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            from PIL import Image

            r = self.records[i]
            img = Image.open(r.image_path).convert("RGB")
            return {
                "id": r.id,
                "image": image_tfm(img),
                "text": text_tfm(r.text),
                "label": int(r.label),
            }

    train_ds = _DS(train_records)
    val_ds = _DS(val_records)

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
        freeze_image_encoder=model_cfg.get("freeze_image_encoder"),
        freeze_text_encoder=model_cfg.get("freeze_text_encoder"),
        head_hidden=int(model_cfg.get("head_hidden", 512)),
        head_dropout=float(model_cfg.get("head_dropout", 0.2)),
    ).to(device)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"Loaded initial weights from {args.ckpt}", flush=True)

    pos_weight_val = 1.0
    import torch as _torch
    pos_weight = _torch.tensor([pos_weight_val], device=device)

    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    vision_params = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": float(train_cfg["lr_head"])},
        {"params": vision_params, "lr": 5.0e-5}],
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    epochs = int(train_cfg["epochs"])
    total_steps = max(1, epochs * max(1, len(train_loader)))
    warmup_steps = int(float(train_cfg.get("warmup_ratio", 0.1)) * total_steps)
    scheduler = _cosine_with_warmup(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    amp_dtype_name = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = {"bfloat16": _torch.bfloat16, "float16": _torch.float16}.get(amp_dtype_name, _torch.bfloat16)
    use_amp = device.type == "cuda"

    # ----------------------------------------------------------------- adversarial params
    robust_cfg = cfg.get("robust", {})
    adv_settings = _load_attack_settings(robust_cfg)

    # Initial evaluation before training starts
    initial_clean_val = _evaluate(model, val_loader, device, use_amp, amp_dtype)
    initial_perturbed_val = _evaluate_robust(model, val_loader, device, use_amp, amp_dtype, adv_settings)

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
        "adversarial_settings": adv_settings,
        "initial_clean_dev": initial_clean_val["macro_f1"],
        "initial_perturbed_dev": initial_perturbed_val["macro_f1"],
    })

    clean_f1 = initial_clean_val["macro_f1"]
    perturbed_f1 = initial_perturbed_val["macro_f1"]
    print(f"Initial performance: Clean F1: {clean_f1:.4f}, Perturbed F1: {perturbed_f1:.4f}", flush=True)

    # Save a few perturbed example images per attack setting for quick inspection
    try:
        from PIL import Image as _Image

        samples_dir = out_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        saved_sample_paths: list[str] = []
        num_samples_per_setting = int(cfg.get("debug", {}).get("samples_per_setting", 3))
        
        # Grab just the first batch from the val_loader
        first_batch = next(iter(val_loader))
        b_img = first_batch["image"].to(device, non_blocking=True)
        b_tok = first_batch["text"].to(device, non_blocking=True)
        b_lbl = first_batch["label"].to(device, non_blocking=True).float()

        for setting in adv_settings:
            adv_eps = float(setting["epsilon"])
            adv_alpha = float(setting["pgd_alpha_frac"]) * adv_eps
            
            adv_images = pgd_image(
                model, b_img, b_tok, b_lbl, 
                epsilon=adv_eps, alpha=adv_alpha, 
                steps=int(setting["pgd_steps"]), random_start=True
            )
            
            eps_val = setting.get("epsilon_over_255", 0)
            n = min(num_samples_per_setting, int(adv_images.shape[0]))
            for i in range(n):
                img = adv_images[i].detach().cpu().clamp(0.0, 1.0)
                arr = (img * 255.0).to(_torch.uint8).permute(1, 2, 0).numpy()
                im = _Image.fromarray(arr)
                fname = samples_dir / f"pert_eps{eps_val}_idx{i}.png"
                im.save(fname)
                saved_sample_paths.append(str(fname))

        if saved_sample_paths:
            logger.write_metrics({"perturbed_sample_paths": saved_sample_paths})
            print(f"Saved {len(saved_sample_paths)} perturbed sample images to {samples_dir}", flush=True)
    except Exception as e:
        print(f"Skipping sample saving: {e}")

    # ------------------------------------------------------------------ loop
    best_perturbed_f1 = -1.0
    best_clean_f1 = -1.0
    patience = int(train_cfg.get("early_stop_patience", 3))
    epochs_since_improve = 0
    global_step = 0
    t0 = time.time()

    import torch.nn.functional as F

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        n_batches = 0
        epoch_train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        for batch in epoch_train_loader:
            adv_setting = adv_settings[global_step % len(adv_settings)]
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            # generate adversarial images (detach returned tensor)
            adv_epsilon = adv_setting["epsilon"]
            adv_alpha = adv_setting["pgd_alpha_frac"] * adv_epsilon
            adv_images = pgd_image(
                model,
                images,
                tokens,
                labels,
                epsilon=adv_epsilon,
                alpha=adv_alpha,
                steps=adv_setting["pgd_steps"],
                random_start=True,
            )
            model.train()

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logit_clean = model(images, tokens)
                    loss_clean = F.binary_cross_entropy_with_logits(logit_clean, labels, pos_weight=pos_weight)
                    
                    logit_adv = model(adv_images.to(device), tokens)
                    loss_adv = F.binary_cross_entropy_with_logits(logit_adv, labels, pos_weight=pos_weight)

                    loss = 0.5 * loss_clean + 0.5 * loss_adv
            else:
                logit_clean = model(images, tokens)
                loss_clean = F.binary_cross_entropy_with_logits(logit_clean, labels, pos_weight=pos_weight)
                
                logit_adv = model(adv_images.to(device), tokens)
                loss_adv = F.binary_cross_entropy_with_logits(logit_adv, labels, pos_weight=pos_weight)

                loss = 0.5 * loss_clean + 0.5 * loss_adv

            loss.backward()
            optimizer.step()
            scheduler.step()

            sum_loss += float(loss.detach())
            n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                logger.log_step({
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(loss.detach()),
                    "adv_epsilon_over_255": adv_setting["epsilon_over_255"],
                    "adv_pgd_steps": adv_setting["pgd_steps"],
                    "adv_pgd_alpha_frac": adv_setting["pgd_alpha_frac"],
                })

        train_loss_epoch = sum_loss / max(1, n_batches)
        
        # Validation
        val = _evaluate(model, val_loader, device, use_amp, amp_dtype)
        # Dynamic robust evaluation against the new model weights
        perturbed_val = _evaluate_robust(model, val_loader, device, use_amp, amp_dtype, adv_settings)

        logger.log_step({
            "epoch": epoch,
            "train_loss_epoch": train_loss_epoch,
            **{f"val_{k}": v for k, v in val.items()},
            **{f"perturbed_{k}": v for k, v in perturbed_val.items()},
        })
        print(f"epoch={epoch}  train_loss={train_loss_epoch:.4f}  val_f1={val['macro_f1']:.4f} val_perturbed_f1={perturbed_val['macro_f1']:.4f}", flush=True)

        improved = perturbed_val['macro_f1'] > best_perturbed_f1
        if improved:
            best_perturbed_f1 = perturbed_val['macro_f1']
            best_clean_f1 = val["macro_f1"]
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "seed": seed,
                "epoch": epoch,
                "val_metrics": val,
                "perturbed_val_mean": perturbed_val['macro_f1'],
            }, ckpt_dir / "best.pt")
            print(f"Saved best checkpoint to {ckpt_dir / 'best.pt'} (perturbed_macro_f1={best_perturbed_f1:.4f})", flush=True)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch={epoch} (no improvement)", flush=True)
                break

    elapsed = time.time() - t0
    # Write final metrics: clean best and perturbed mean
    logger.write_metrics({
        "best_dev_macro_f1": best_clean_f1,
        "elapsed_seconds": elapsed,
        "dev_perturbed_mean_macro_f1": best_perturbed_f1,
    })
    print(f"Best perturbed dev macro_f1={best_perturbed_f1:.4f}, elapsed {elapsed:.1f}s", flush=True)


def _evaluate(model, loader, device, use_amp, amp_dtype) -> dict[str, float]:
    import torch

    # Adapt loader-based evaluation to the shared batch evaluator.
    def _iter_loader_batches():
        with torch.no_grad():
            for batch in loader:
                yield {
                    "images": batch["image"].to(device, non_blocking=True),
                    "tokens": batch["text"].to(device, non_blocking=True),
                    "labels": batch["label"],
                }

    return _evaluate_batches(model, _iter_loader_batches(), device, use_amp, amp_dtype)


def _evaluate_robust(model, loader, device, use_amp, amp_dtype, adv_settings) -> dict[str, float]:
    from robust_meme_hate_detection.attacks.pgd import pgd_image
    import torch

    def _iter_adv_batches():
        for setting in adv_settings:
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                tokens = batch["text"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True).float()

                # Ensure eval mode so PGD calculates gradients correctly against the current validation state
                model.eval() 
                
                adv_epsilon = float(setting["epsilon"])
                adv_alpha = float(setting["pgd_alpha_frac"]) * adv_epsilon
                
                # Dynamically generate the attack for the current epoch's weights
                with torch.enable_grad():
                    adv_images = pgd_image(
                        model,
                        images,
                        tokens,
                        labels,
                        epsilon=adv_epsilon,
                        alpha=adv_alpha,
                        steps=int(setting["pgd_steps"]),
                        random_start=True,
                    )

                yield {
                    "images": adv_images.detach(),
                    "tokens": tokens,
                    "labels": labels,
                }

    return _evaluate_batches(model, _iter_adv_batches(), device, use_amp, amp_dtype)


def _evaluate_batches(
    model,
    batches_iter,
    device,
    use_amp,
    amp_dtype,
) -> dict[str, float]:
    """Evaluate an iterator of batches in the form {'images','tokens','labels'}.

    Returns the same dict as `classification_metrics(...).to_dict()`.
    """
    import torch
    from robust_meme_hate_detection.eval.metrics import classification_metrics

    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in batches_iter:
            images = batch["images"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logit = model(images, tokens)
            else:
                logit = model(images, tokens)
            probs = torch.sigmoid(logit.float()).detach().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(int(x) for x in labels.tolist())

    metrics = classification_metrics(all_probs, all_labels, threshold=0.5)
    return metrics.to_dict()


if __name__ == '__main__':
    raise SystemExit(main())