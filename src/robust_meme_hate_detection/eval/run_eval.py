"""Evaluate a trained checkpoint: clean metrics, threshold sweep, optional PGD.

Loads ``best.pt`` written by the Stage-1 training script, builds the model on
GPU, runs a forward pass over a labelled split, computes classification
metrics, sweeps the decision threshold on macro-F1, and (optionally) runs a
small PGD attack-readiness check that compares clean vs attacked AUROC and
attack success rate.

Outputs a single JSON file at ``<out>/eval.json``. Designed to be submitted
as a short Run:AI job via ``cluster.sh submit-cmd``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best.pt")
    parser.add_argument("--config", required=True, help="Training config (yaml) for model + data settings")
    parser.add_argument("--split", default="dev", help="Logical split name to evaluate on")
    parser.add_argument("--out", required=True, help="Output dir for eval.json")
    parser.add_argument("--pgd-epsilon", type=float, default=4 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=1 / 255)
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--pgd-max-batches", type=int, default=8, help="0 to skip PGD")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    from robust_meme_hate_detection.attacks.pgd import pgd_image
    from robust_meme_hate_detection.data.hateful_memes import load_hateful_memes_records
    from robust_meme_hate_detection.data.transforms import ClipImage01Transform, ClipTokenize
    from robust_meme_hate_detection.eval.metrics import (
        attack_success_rate,
        classification_metrics,
        robustness_gap,
    )
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.utils.threshold import sweep_threshold

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    dataset_root = Path(data_cfg["dataset_root"])

    image_tfm = ClipImage01Transform(size=224)
    text_tfm = ClipTokenize(arch=model_cfg["arch"])

    records = load_hateful_memes_records(dataset_root, args.split)
    records = [r for r in records if r.label is not None]

    class _DS:
        def __init__(self, records):
            self.records = list(records)
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            from PIL import Image
            r = self.records[i]
            img = Image.open(r.image_path).convert("RGB")
            return {"id": r.id, "image": image_tfm(img), "text": text_tfm(r.text), "label": int(r.label)}

    loader = DataLoader(_DS(records), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPHateMemeClassifier(
        arch=model_cfg["arch"],
        pretrained=model_cfg.get("pretrained", "laion2b_s34b_b79k"),
        freeze_encoders=True,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # --------- clean pass ---------
    clean_probs: list[float] = []
    labels: list[int] = []
    ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = model(images, tokens)
            clean_probs.extend(torch.sigmoid(logit.float()).detach().cpu().tolist())
            labels.extend(int(x) for x in batch["label"].tolist())
            ids.extend(batch["id"])

    clean_metrics = classification_metrics(clean_probs, labels, threshold=0.5)
    best_threshold, threshold_grid = sweep_threshold(clean_probs, labels)
    clean_at_best = classification_metrics(clean_probs, labels, threshold=best_threshold.threshold)

    payload = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "n": len(labels),
        "clean": {
            "at_0.5": clean_metrics.to_dict(),
            "best_threshold": best_threshold.threshold,
            "at_best_threshold": clean_at_best.to_dict(),
        },
        "threshold_grid": [r.to_dict() for r in threshold_grid],
    }

    # --------- optional PGD ---------
    if args.pgd_max_batches > 0:
        attacked_probs: list[float] = []
        attacked_labels: list[int] = []
        seen_batches = 0
        for batch in loader:
            if seen_batches >= args.pgd_max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text"].to(device, non_blocking=True)
            label_t = batch["label"].to(device).float()

            adv = pgd_image(
                model, images, tokens, label_t,
                epsilon=args.pgd_epsilon, alpha=args.pgd_alpha, steps=args.pgd_steps,
            )
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logit_adv = model(adv, tokens)
                attacked_probs.extend(torch.sigmoid(logit_adv.float()).detach().cpu().tolist())
                attacked_labels.extend(int(x) for x in batch["label"].tolist())
            seen_batches += 1

        # Use the same prefix of clean predictions to compute deltas correctly.
        clean_subset = clean_probs[: len(attacked_probs)]
        attacked = classification_metrics(attacked_probs, attacked_labels, threshold=best_threshold.threshold)
        clean_subset_metrics = classification_metrics(clean_subset, attacked_labels, threshold=best_threshold.threshold)

        payload["pgd"] = {
            "epsilon": args.pgd_epsilon,
            "alpha": args.pgd_alpha,
            "steps": args.pgd_steps,
            "n_attacked": len(attacked_probs),
            "clean_subset_at_best_threshold": clean_subset_metrics.to_dict(),
            "attacked_at_best_threshold": attacked.to_dict(),
            "robustness_gap": robustness_gap(clean_subset_metrics, attacked),
            "attack_success_rate": attack_success_rate(
                clean_subset, attacked_probs, attacked_labels, threshold=best_threshold.threshold
            ),
        }

    out_path = out_dir / "eval.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("clean.at_best_threshold:", payload["clean"]["at_best_threshold"])
    if "pgd" in payload:
        print("pgd.attacked_at_best_threshold:", payload["pgd"]["attacked_at_best_threshold"])
        print("pgd.attack_success_rate:", payload["pgd"]["attack_success_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
