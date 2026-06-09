"""Interactive inference demo for the four headline models in the report.

For each model the demo shows the 10 bundled memes in three versions -- the
**clean** meme, a **naturalistic** label-preserving edit, and a **white-box PGD**
adversarial example crafted against that model -- then, on a key press, reveals
the model's prediction on each and whether it is correct.

    python demo/demo.py              # interactive matplotlib window (default)
    python demo/demo.py --live       # recompute predictions with the model (slower)
    python demo/demo.py --save       # write a result board PNG per model (headless)
    python demo/demo.py --prepare    # rebuild the cached perturbed views + predictions

Controls:  SPACE = reveal predictions / next model   .  arrow keys = prev/next   .  q = quit

The cached perturbed images and predictions ship with the repo, so the default
view needs no GPU and no model download. ``--prepare`` regenerates them and is
the only step that loads the checkpoints (and downloads the OpenCLIP base
weights the first time).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

# The four models cycled through, in report order, with the cross-robustness
# headline (Table: cross-robustness on test_unseen) for the on-screen caption.
MODELS = [
    ("clean", "Clean baseline", "broken by both threats"),
    ("naturalistic", "Naturalistic (KLDrop)", "robust to edits, broken by PGD"),
    ("adversarial", "Adversarial (Madry)", "robust to PGD, costs clean AUROC"),
    ("combined", "Combined", "robust to both threats"),
]
VIEWS = [("clean", "Clean"), ("nat", "Naturalistic edit"), ("adv", "White-box PGD")]
THRESHOLD = 0.5

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
CACHE = HERE / "cache"
CHECKPOINTS = HERE / "checkpoints"

# The clean / naturalistic checkpoints are tiny (frozen encoders) and ship in the
# repo. The adversarial / combined checkpoints carry a trained vision tower
# (~339 MB), exceed GitHub's 100 MB file limit, and are hosted as Release assets;
# they are fetched on demand (only --live / --prepare need any checkpoint at all).
_RELEASE = "https://github.com/enricguaschmesia-lab/robust-meme-hate-detection/releases/download/demo-checkpoints"
CHECKPOINT_URLS = {
    "adversarial": f"{_RELEASE}/adversarial.pt",
    "combined": f"{_RELEASE}/combined.pt",
}


def _load_samples() -> list[dict]:
    return json.loads((SAMPLES / "memes.json").read_text())


def ensure_checkpoint(name: str) -> Path:
    """Return the checkpoint path, downloading it from the Release if missing."""
    path = CHECKPOINTS / f"{name}.pt"
    if path.exists():
        return path
    if name not in CHECKPOINT_URLS:
        raise FileNotFoundError(f"Missing checkpoint {path} and no download URL for {name!r}.")
    import shutil
    import urllib.error
    import urllib.request

    url = CHECKPOINT_URLS[name]
    print(f"[download] {name}.pt (~339 MB) from {url} ...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.part")
    try:
        headers = {}
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response, tmp.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(response, out)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"\nCould not download {name}.pt automatically ({exc}).\n"
            f"This happens if the GitHub repository is private or you are offline.\n"
            f"Download it manually from the 'demo-checkpoints' release and place it at:\n"
            f"    {path}\n"
            f"URL: {url}\n"
            f"(The default demo -- `python demo/demo.py` -- needs no checkpoints; only "
            f"--live / --prepare do.)"
        ) from exc
    tmp.rename(path)
    print(f"[download] saved {path}", flush=True)
    return path


def label_str(label: int) -> str:
    return "HATEFUL" if label == 1 else "benign"


# ============================================================================
# Prepare: build the cached naturalistic + adversarial views and predictions.
# ============================================================================


def prepare(pgd_steps: int = 25) -> None:
    import torch
    from PIL import Image

    sys.path.insert(0, str(HERE.parent / "src"))
    from robust_meme_hate_detection.attacks import pgd_image
    from robust_meme_hate_detection.data import ClipImageTransform, ClipTokenizer
    from robust_meme_hate_detection.model import load_classifier
    from robust_meme_hate_detection.perturbations import Compose, ImagePerturbation, TextPerturbation
    from robust_meme_hate_detection.seeding import seed_everything

    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tfm, text_tfm = ClipImageTransform(), ClipTokenizer()
    memes = _load_samples()
    (CACHE / "nat").mkdir(parents=True, exist_ok=True)

    # --- Naturalistic view (model-independent). ---
    nat_text: dict[str, str] = {}
    for m in memes:
        img = Image.open(SAMPLES / m["image"]).convert("RGB")
        seed = int(m["id"])
        image_edit = Compose([
            ImagePerturbation.from_preset("gaussian_noise", "high", seed=seed),
            ImagePerturbation.from_preset("contrast_up", "high", seed=seed),
        ])
        text_edit = Compose([
            TextPerturbation.from_preset("leetspeak", "medium", seed=seed),
            TextPerturbation.from_preset("char_deletion", "medium", seed=seed),
        ])
        edited = image_edit(img)
        edited.save(CACHE / "nat" / f"{m['id']}.png")
        nat_text[m["id"]] = text_edit(m["text"])
    (CACHE / "nat_text.json").write_text(json.dumps(nat_text, indent=2))

    # --- Per-model adversarial views + predictions on all three views. ---
    predictions: dict[str, dict] = {}
    for name, title, _ in MODELS:
        print(f"[prepare] {title}: loading checkpoint + crafting PGD views ...", flush=True)
        model = load_classifier(ensure_checkpoint(name), device)
        (CACHE / name).mkdir(parents=True, exist_ok=True)
        predictions[name] = {}
        for m in memes:
            clean_img = Image.open(SAMPLES / m["image"]).convert("RGB")
            nat_img = Image.open(CACHE / "nat" / f"{m['id']}.png").convert("RGB")
            clean_t = image_tfm(clean_img).unsqueeze(0).to(device)
            nat_t = image_tfm(nat_img).unsqueeze(0).to(device)
            tok = text_tfm(m["text"]).unsqueeze(0).to(device)
            nat_tok = text_tfm(nat_text[m["id"]]).unsqueeze(0).to(device)
            y = torch.tensor([float(m["label"])], device=device)
            eps = 8 / 255.0
            adv_t = pgd_image(model, clean_t, tok, y, eps, 0.25 * eps, pgd_steps)
            _save_tensor_image(adv_t[0], CACHE / name / f"{m['id']}.png")
            with torch.no_grad():
                predictions[name][m["id"]] = {
                    "clean": float(torch.sigmoid(model(clean_t, tok)).item()),
                    "nat": float(torch.sigmoid(model(nat_t, nat_tok)).item()),
                    "adv": float(torch.sigmoid(model(adv_t, tok)).item()),
                }
    (CACHE / "predictions.json").write_text(json.dumps(predictions, indent=2))
    print(f"[prepare] done -> {CACHE}", flush=True)


def _save_tensor_image(t, path: Path) -> None:
    from PIL import Image
    import numpy as np

    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ============================================================================
# Display
# ============================================================================


def _view_image_path(model_name: str, view: str, meme_id: str) -> Path:
    if view == "clean":
        return SAMPLES / "images" / f"{meme_id}.png"
    if view == "nat":
        return CACHE / "nat" / f"{meme_id}.png"
    return CACHE / model_name / f"{meme_id}.png"


def _predict_live(model_name: str, memes: list[dict]) -> dict[str, dict]:
    """Recompute predictions for one model with the actual network (``--live``)."""
    import torch
    from PIL import Image

    sys.path.insert(0, str(HERE.parent / "src"))
    from robust_meme_hate_detection.data import ClipImageTransform, ClipTokenizer
    from robust_meme_hate_detection.model import load_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tfm, text_tfm = ClipImageTransform(), ClipTokenizer()
    nat_text = json.loads((CACHE / "nat_text.json").read_text())
    model = load_classifier(ensure_checkpoint(model_name), device)
    preds: dict[str, dict] = {}
    for m in memes:
        out = {}
        for view, _ in VIEWS:
            img = Image.open(_view_image_path(model_name, view, m["id"])).convert("RGB")
            text = nat_text[m["id"]] if view == "nat" else m["text"]
            it = image_tfm(img).unsqueeze(0).to(device)
            tk = text_tfm(text).unsqueeze(0).to(device)
            with torch.no_grad():
                out[view] = float(torch.sigmoid(model(it, tk)).item())
        preds[m["id"]] = out
    return preds


class DemoBoard:
    """A matplotlib board: rows = clean / nat / adv, columns = 10 memes."""

    def __init__(self, memes, all_preds, *, live: bool):
        import matplotlib.pyplot as plt
        from PIL import Image

        self.plt = plt
        self.Image = Image
        self.memes = memes
        self.all_preds = all_preds
        self.nat_text = json.loads((CACHE / "nat_text.json").read_text())
        self.live = live
        self.model_idx = 0
        self.revealed = False

        n = len(memes)
        self.fig, self.axes = plt.subplots(3, n, figsize=(2.7 * n, 8.0))
        self.fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.11, hspace=0.35, wspace=0.08)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    # ------------------------------------------------------------------ draw
    def _preds_for_current(self):
        name = MODELS[self.model_idx][0]
        if self.live and name not in self._live_cache:
            self._live_cache[name] = _predict_live(name, self.memes)
        return self._live_cache[name] if self.live else self.all_preds[name]

    _live_cache: dict = {}

    def _draw(self):
        name, title, blurb = MODELS[self.model_idx]
        preds = self._preds_for_current() if self.revealed else None
        for c, m in enumerate(self.memes):
            for r, (view, view_label) in enumerate(VIEWS):
                ax = self.axes[r][c]
                ax.clear()
                ax.imshow(self.Image.open(_view_image_path(name, view, m["id"])).convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"#{m['id']} [{label_str(m['label'])}]", fontsize=9, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(view_label, fontsize=10, fontweight="bold", rotation=0,
                                  ha="right", va="center", labelpad=42)
                caption = self.nat_text[m["id"]] if view == "nat" else m["text"]
                ax.text(0.5, -0.08, textwrap.fill(caption, width=28),
                        transform=ax.transAxes, ha="center", va="top", fontsize=6.7)
                if preds is not None:
                    p = preds[m["id"]][view]
                    pred = 1 if p >= THRESHOLD else 0
                    correct = pred == m["label"]
                    ax.text(0.5, 0.04, f"{label_str(pred)} p={p:.2f} {'OK' if correct else 'X'}",
                            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
                            color="white", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc=("#2e7d32" if correct else "#c62828"), ec="none"))
        acc = self._accuracy(preds) if preds is not None else None
        hint = "SPACE: next model" if self.revealed else "SPACE: run inference"
        head = f"Model {self.model_idx + 1}/4:  {title}  ({blurb})"
        if acc is not None:
            head += f"\naccuracy  clean {acc['clean']:.0%} | nat {acc['nat']:.0%} | PGD {acc['adv']:.0%}"
        head += f"      [{hint} | <- -> switch | q quit]"
        self.fig.suptitle(head, fontsize=11)
        self.fig.canvas.draw_idle()

    def _accuracy(self, preds):
        out = {}
        for view, _ in VIEWS:
            hits = sum((1 if preds[m["id"]][view] >= THRESHOLD else 0) == m["label"] for m in self.memes)
            out[view] = hits / len(self.memes)
        return out

    # ------------------------------------------------------------------ keys
    def _on_key(self, event):
        if event.key in ("q", "escape"):
            self.plt.close(self.fig)
        elif event.key == " ":
            if not self.revealed:
                self.revealed = True
            else:
                self.model_idx = (self.model_idx + 1) % len(MODELS)
                self.revealed = False
            self._draw()
        elif event.key in ("right", "left"):
            self.model_idx = (self.model_idx + (1 if event.key == "right" else -1)) % len(MODELS)
            self.revealed = False
            self._draw()

    def show(self):
        self.plt.show()

    def save_boards(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.revealed = True
        for i in range(len(MODELS)):
            self.model_idx = i
            self._draw()
            path = out_dir / f"{i + 1}_{MODELS[i][0]}.png"
            self.fig.savefig(path, dpi=130)
            print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Rebuild cached views + predictions.")
    parser.add_argument("--pgd-steps", type=int, default=25, help="PGD steps used when building adversarial views.")
    parser.add_argument("--live", action="store_true", help="Recompute predictions with the model.")
    parser.add_argument("--save", action="store_true", help="Write result board PNGs instead of opening a window.")
    args = parser.parse_args()

    if args.prepare:
        prepare(pgd_steps=args.pgd_steps)
        return

    if not (CACHE / "predictions.json").exists():
        sys.exit("No cached predictions. Run:  python demo/demo.py --prepare")

    memes = _load_samples()
    all_preds = json.loads((CACHE / "predictions.json").read_text())

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    board = DemoBoard(memes, all_preds, live=args.live)
    if args.save:
        board.save_boards(HERE / "out")
    else:
        board.show()


if __name__ == "__main__":
    main()
