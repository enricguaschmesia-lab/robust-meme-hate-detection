# Robust Meme Hate Detection

**Robustness of CLIP-fusion classifiers under naturalistic and adversarial perturbations**
EE-559 mini-project, Group 49 — Enric Guasch Mesià, Henrik Paul Gruber, Jane Klavir (EPFL).

This is the clean delivery of the code behind the report. A CLIP-fusion hateful-meme
classifier is audited under two threat models — **naturalistic** label-preserving
edits (text + image perturbations) and **white-box** L∞ adversarial attacks
(FGSM / PGD) — and improved with two training-time defenses:

* **Naturalistic (KLDrop)** — clean↔perturbed KL-consistency plus per-example
  text-modality dropout. Revives the under-used image branch and lifts
  naturalistic robustness at ~zero clean-accuracy cost.
* **Adversarial (Madry)** — PGD adversarial training. Closes the white-box gap at
  a ~0.12 clean-AUROC cost.
* **Combined** — one model robust to both threats.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate #(Unix shell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1 #(Windows Powershell)

pip install -e .          # installs the package + dependencies (torch, open_clip, ...)
pytest                    # unit tests (CPU, no dataset, no network)
```

### 1. Interactive inference demo (no dataset, no GPU needed)

The fastest way to see the result. For each of the four headline models it shows
the 10 bundled memes in three versions — **clean**, a **naturalistic edit**, and
a **white-box PGD** example crafted against that model — and reveals the model's
prediction (and whether it is correct) on a key press.

```bash
python demo/demo.py                # interactive matplotlib window
python demo/demo.py --save         # headless: writes a result board PNG per model to demo/out/
python demo/demo.py --live         # recompute predictions with the model instead of the cache
```

Controls: **SPACE** = run inference / advance to next model · **← →** = switch model · **q** = quit.

The cached perturbed images and predictions ship with the repo, so **the default
view needs neither a GPU, a model download, nor the dataset**. The small
clean / naturalistic checkpoints live in `demo/checkpoints/`; the larger
adversarial / combined checkpoints (a trained vision tower, ~339 MB each) exceed
GitHub's file limit and are hosted as assets of the **`demo-checkpoints`**
Release. Only `--live` / `--prepare` need them: they download automatically on
first use (as does the OpenCLIP base weight). If automatic download fails (e.g.
the repository is private), download `adversarial.pt` and `combined.pt` from the
release page and drop them into `demo/checkpoints/`.

What to look for: the **clean** and **naturalistic** models are flipped to the
wrong label by PGD, while **adversarial** and **combined** resist it — and the
naturalistic / combined models survive the naturalistic edit that fools the clean
baseline. (Demo PGD uses 10 steps for speed; the report uses 25.)

### 2. Getting the data
Since the datasets are too huge to be submitted or hosted on GitHub, we did not include them. This section will outline how to proceed. For the Meta Hate Dataset (raw from https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset), no preprocessing is needed, as it is already in the required shape. <br>
For MAMI (raw download from https://drive.google.com/file/d/169qe9n4EbNlVbzFWNMjVX3N74Hh5Jcqr/view), please run 

```bash
python -m robust_meme_hate_detection.preprocessing.prepare_mami \
    --raw-dir "data/raw/MAMI DATASET" \
    --out-dir data/mami \
    --copy \
    --missing-ok
```
For the MultiOFF dataset (raw download from https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6), please run 

```bash
python -m robust_meme_hate_detection.preprocessing.prepare_multioff \
    --raw-dir data/raw/MultiOFF_Dataset \
    --out-dir data/multioff \
    --copy \
    --missing-ok
```
If you want to train/evaluate on them, please adapt the corresponding paths in the config files.

### 3. Train a model

One config-driven entry point covers every recipe (the `robust:` block selects
the regime). Set `data.dataset_root` in the config to your Hateful Memes
directory first (see [`data/README.md`](data/README.md)).

```bash
python train.py --config configs/clean.yaml        --seed 0 --out runs/clean
python train.py --config configs/naturalistic.yaml --seed 0 --out runs/naturalistic
python train.py --config configs/adversarial.yaml  --seed 0 --out runs/adversarial
python train.py --config configs/combined.yaml     --seed 0 --out runs/combined
# audit baselines:
python train.py --config configs/image_only.yaml   --seed 0 --out runs/image_only
python train.py --config configs/text_only.yaml    --seed 0 --out runs/text_only
```

Runs on GPU if available, else CPU. Writes `<out>/best.pt` and `<out>/metrics.json`.

### 4. Evaluate a checkpoint

One entry point with four suites, each mapping to a result in the report:

```bash
python evaluate.py --ckpt runs/naturalistic/best.pt --dataset-root <root> --suite clean
python evaluate.py --ckpt runs/naturalistic/best.pt --dataset-root <root> --suite modality   # the audit
python evaluate.py --ckpt runs/naturalistic/best.pt --dataset-root <root> --suite perturbed  # naturalistic grid
python evaluate.py --ckpt runs/adversarial/best.pt  --dataset-root <root> --suite whitebox    # FGSM/PGD sweep
```

Use `--split test_unseen` (default), `--max-samples N` for a quick check, and
`--out result.json` to save.

## Repository layout

```text
src/robust_meme_hate_detection/
  model.py          CLIP dual-encoder + fusion head (+ slim checkpoint save/load)
  data.py           Hateful Memes records, splits, CLIP transforms, dataset
  perturbations.py  naturalistic suite: 8 text x 11 image attacks x 3 severities + Compose
  attacks.py        FGSM / PGD on raw [0,1] pixels
  losses.py         the unified robust loss (clean / naturalistic / adversarial terms)
  augment.py        RobustAugmenter — on-the-fly perturbed view for training
  metrics.py        AUROC / macro-F1 / FPR / FNR / attack-success-rate
  runtime.py        shared device / transforms / dataloader / inference helpers
  seeding.py        reproducibility
train.py            train any recipe (config-driven)
evaluate.py         evaluate any checkpoint (clean | modality | perturbed | whitebox)
demo/               interactive inference demo + 10 sample memes + 4 slim checkpoints
configs/            one YAML per recipe
data/               where to place the dataset + the committed train split
tests/              CPU unit tests
```

## Model

OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) image + text encoders map a meme to
`v, t ∈ R^512`. A fusion head consumes `z = [t; v; |t − v|; t ⊙ v] ∈ R^2048`
through `LayerNorm → Linear(2048,512) → GELU → Linear(512,1)`. Pixel
normalization is a module **inside** the model, so `model(images01, tokens)` is
differentiable w.r.t. raw `[0,1]` pixels — that is what lets white-box attacks
and consistency / adversarial training share one forward graph. Encoders are
frozen by default (only the ~1 M-parameter head trains); adversarial training
unfreezes the vision tower.

## Reproducibility & compute

All training uses `seed_everything`; the train/held-out partition is the
committed `data/processed/splits/train_split.json`. Frozen-encoder recipes
(clean / naturalistic / baselines) train in minutes on one GPU; adversarial /
combined recipes are heavier because each step runs a PGD inner loop and the
vision encoder is unfrozen. There is **no cluster automation** here — `train.py`
and `evaluate.py` are plain scripts you run directly (locally or on a GPU node).
