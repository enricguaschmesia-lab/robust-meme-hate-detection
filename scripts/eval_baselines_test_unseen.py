"""Offline test_unseen clean-eval of the Table 1 rows at **dev-selected thresholds**.

Table 1 reports clean results on the held-out test_unseen split. Per the report's
protocol ("thresholds fixed on clean dev and reused"), macro-F1 / accuracy are
evaluated at each model's dev-optimal threshold (AUROC is threshold-free, so it is
unchanged). The dev-optimal thresholds were read from the dev evals:

    multimodal full        0.30   (modality-ablation-stage1-seed0, dev)
    text-only forward      0.35   (   "   )
    image-only forward     0.38   (   "   )
    image-only (dedicated) 0.46   (perturbed-baseline-image-seed0, dev)
    text-only (dedicated)  0.48   (perturbed-baseline-text-seed0, dev)

Runs fully offline on CPU. The text-forward rows need no images (``forward_text_only``)
and the checkpoints are self-contained (frozen CLIP weights included), so no
download is needed. The two image-dependent rows (multimodal full, image-only
forward) cannot be recomputed locally -- no test_unseen images in this checkout --
and are filled from a cluster eval (see ``--image-rows-json``).
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import torch

from robust_meme_hate_detection.data.transforms import ClipTokenize
from robust_meme_hate_detection.eval.metrics import classification_metrics
from robust_meme_hate_detection.models.clip_fusion import CLIPHateMemeClassifier

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "data/processed/splits/test_unseen_labels.jsonl"

DEV_THRESHOLDS = {
    "multimodal": 0.30,
    "text_only_fwd": 0.35,
    "image_only_fwd": 0.38,
    "image_dedicated": 0.46,
    "text_dedicated": 0.48,
}


def load_records() -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        texts.append(str(item.get("text", "")))
        labels.append(int(item["label"]))
    return texts, labels


def build_model(ckpt_path: Path) -> CLIPHateMemeClassifier:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]["model"]
    model = CLIPHateMemeClassifier(
        arch=cfg["arch"], pretrained=None, freeze_encoders=True,
        head_hidden=cfg["head_hidden"], head_dropout=cfg["head_dropout"],
    )
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def text_only_probs(model: CLIPHateMemeClassifier, texts: list[str], bs: int = 64) -> list[float]:
    tok = ClipTokenize(arch="ViT-B-32")
    probs: list[float] = []
    for i in range(0, len(texts), bs):
        ids = torch.stack([tok(t) for t in texts[i : i + bs]])
        probs.extend(torch.sigmoid(model.forward_text_only(ids).squeeze(-1)).tolist())
    return probs


def main() -> None:
    texts, labels = load_records()
    n, pos = len(labels), sum(labels)
    print(f"test_unseen: n={n}, positives={pos} ({pos / n:.3f})")
    rows: dict[str, dict] = {}

    # --- Majority (all-0): threshold-independent ---
    m = classification_metrics([0.0] * n, labels, threshold=0.5)
    rows["majority"] = {"auroc": m.auroc, "macro_f1": m.macro_f1, "accuracy": m.accuracy, "thr": None}

    # --- Image-only dedicated @0.46: read from the existing test_unseen grid ---
    img = json.loads(
        (REPO / "cluster-results/perturbed-baseline-image-test-unseen-seed0/perturbed_eval.json").read_text()
    )
    grid = {round(r["threshold"], 2): r for r in img["threshold_grid"]}
    g = grid[DEV_THRESHOLDS["image_dedicated"]]
    rows["image_dedicated"] = {
        "auroc": img["clean"]["at_0.5"]["auroc"], "macro_f1": g["macro_f1"],
        "accuracy": g["accuracy"], "thr": DEV_THRESHOLDS["image_dedicated"],
    }

    # --- Text-only dedicated @0.48 ---
    t_probs = text_only_probs(build_model(REPO / "cluster-results/baseline-text-seed0/ckpt/best.pt"), texts)
    mt = classification_metrics(t_probs, labels, threshold=DEV_THRESHOLDS["text_dedicated"])
    rows["text_dedicated"] = {"auroc": mt.auroc, "macro_f1": mt.macro_f1, "accuracy": mt.accuracy,
                              "thr": DEV_THRESHOLDS["text_dedicated"]}

    # --- Multimodal text-only forward @0.35, 3-seed mean ---
    f1s, accs, aurocs = [], [], []
    for ck in sorted(glob.glob(str(REPO / "cluster-results/stage1-seed*/ckpt/best.pt"))):
        p = text_only_probs(build_model(Path(ck)), texts)
        r = classification_metrics(p, labels, threshold=DEV_THRESHOLDS["text_only_fwd"])
        f1s.append(r.macro_f1); accs.append(r.accuracy); aurocs.append(r.auroc)
    rows["text_only_fwd"] = {"auroc": st.mean(aurocs), "macro_f1": st.mean(f1s),
                             "accuracy": st.mean(accs), "thr": DEV_THRESHOLDS["text_only_fwd"],
                             "n_seeds": len(f1s)}

    print("\n=== test_unseen, macro-F1/Acc at dev-selected thresholds ===")
    for name, r in rows.items():
        thr = "n/a" if r["thr"] is None else f"{r['thr']:.2f}"
        print(f"{name:18s} thr={thr}  AUROC={r['auroc']:.4f}  F1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}")
    print("\nNOT computable locally (need test_unseen images): multimodal full @0.30, image-only fwd @0.38")

    (REPO / "cluster-results/baselines-test-unseen-devthr.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
