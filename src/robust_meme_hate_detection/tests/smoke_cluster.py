"""Cluster-side smoke job: real model + real dataset + one forward+backward.

Exercised in Phase 7 of the implementation plan. Loads the configured Hateful
Memes dataset, instantiates the model, trains for a few mini-batches at
random head, reports random-AUROC on a small dev subset, and writes a tiny
``metrics.json`` so ``summarize-results`` has something to read.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from robust_meme_hate_detection.utils.seeding import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-batches", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(cfg.get("seed", 0)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    from robust_meme_hate_detection.data.hateful_memes import load_hateful_memes_records
    from robust_meme_hate_detection.data.transforms import ClipImage01Transform, ClipTokenize
    from robust_meme_hate_detection.eval.metrics import classification_metrics
    from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier
    from robust_meme_hate_detection.utils.logging import JsonRunLogger

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})

    dataset_root = Path(data_cfg["dataset_root"])
    image_tfm = ClipImage01Transform(size=224)
    text_tfm = ClipTokenize(arch=model_cfg["arch"])

    val_records = load_hateful_memes_records(dataset_root, data_cfg.get("val_split", "dev"))

    # Adapter (same shape used by stage1 — duplicated here to avoid import cycles).
    class _RecordsDataset:
        def __init__(self, records):
            from PIL import Image  # noqa: F401
            self.records = list(records)

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
                "label": -1 if r.label is None else int(r.label),
            }

    val_ds = _RecordsDataset(val_records)
    # Smoke uses num_workers=0 to avoid Run:AI default /dev/shm limits.
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg.get("batch_size", 32)), shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPHateMemeClassifier(
        arch=model_cfg["arch"],
        pretrained=model_cfg.get("pretrained", "laion2b_s34b_b79k"),
        freeze_encoders=True,
    ).to(device)

    print(
        f"device={device}",
        f"trainable_params={model.trainable_parameter_count()}",
        f"total_params={model.total_parameter_count()}",
        f"n_val={len(val_records)}",
        flush=True,
    )

    # Forward + backward on a few mini-batches.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    t0 = time.time()
    all_probs: list[float] = []
    all_labels: list[int] = []
    for i, batch in enumerate(val_loader):
        if i >= args.max_batches:
            break
        images = batch["image"].to(device)
        tokens = batch["text"].to(device)
        labels = batch["label"].to(device).float()

        with torch.autocast(device_type=device.type if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(images, tokens)
            loss = loss_fn(logit, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logit.float()).detach().cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(int(x) for x in labels.cpu().tolist())
        print(f"batch={i} loss={float(loss):.4f}", flush=True)

    elapsed = time.time() - t0
    metrics = classification_metrics(all_probs, all_labels, threshold=0.5) if all_labels else None
    payload = {
        "smoke": True,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "trainable_params": model.trainable_parameter_count(),
        "total_params": model.total_parameter_count(),
        "n_val_seen": len(all_labels),
    }
    if metrics is not None:
        for k, v in metrics.to_dict().items():
            payload[f"val_{k}"] = v
    JsonRunLogger(out_dir).write_metrics(payload)
    print("smoke_cluster OK", payload, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
