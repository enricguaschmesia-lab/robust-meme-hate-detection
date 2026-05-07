# Robust Meme Hate Detection

Course project (EE-559, EPFL): **adversarial robustness of multimodal hateful
meme detectors**. We train a CLIP-based dual-encoder classifier on the Hateful
Memes dataset, attack it with FGSM/PGD and naturalistic perturbations, and
study a robust training variant that combines perturbation augmentation with a
KL consistency loss.

## Status

Phase 1-2 (baseline + attack-readiness) is **complete**. See
[`project_planning/Phase1-2_Completion_Report.md`](project_planning/Phase1-2_Completion_Report.md).

| Metric (dev, 500 examples, 3 seeds) | Value |
|---|---:|
| Stage-1 AUROC (mean ± σ)            | 0.734 ± 0.007 |
| Best-τ macro F1 (seed 0)            | 0.672 |
| PGD ε=4/255, 5 steps — attacked AUROC | 0.0004 |
| Attack success rate                 | 98.8% |
| Wall clock per seed (A100 80 GB)    | ~140 s |

Next: Phase 3 (perturbation suite) and Phase 5 (Stage-3 robust training).

## Architecture (one-paragraph)

OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) image and text encoders, frozen by
default. A small fusion head consumes
`f = [t, v, |t − v|, t ⊙ v] ∈ ℝ^{2048}` through `LayerNorm → Linear(2048→512)
→ GELU → Linear(512→1)`. Image normalization is a `nn.Module` *inside* the
model, so `model(images01, tokens)` is fully differentiable w.r.t. raw `[0,1]`
pixels — that is what makes white-box image attacks and consistency-loss
training share the same forward graph. Full spec:
[`project_planning/model_architecture.md`](project_planning/model_architecture.md).

## Repository layout

```text
configs/                       Stage and data YAML configs
  stage1.yaml                  Frozen-encoder training
  smoke.yaml                   Short cluster smoke
  data/hateful_memes.yaml      Dataset paths
src/robust_meme_hate_detection/
  models/clip_fusion.py        CLIPHateMemeClassifier + Normalize + FusionHead
  data/                        Hateful Memes loader, splits, transforms
  attacks/pgd.py               FGSM / PGD on [0,1] images
  perturb/                     Naturalistic perturbations (Phase 3 — stub)
  train/stage1.py              Frozen-encoder trainer
  eval/run_eval.py             Clean + threshold sweep + PGD eval
  eval/metrics.py              Accuracy, F1, AUROC, ASR, robustness gap
  utils/                       Seeding, JSON logging, threshold sweep
  tests/smoke_synthetic.py     CPU-only synthetic forward+backward+PGD test
  tests/smoke_cluster.py       Cluster-side real-GPU smoke
scripts/
  make_train_split.py          One-shot 8000/500 stratified ID partition
  inspect_hateful_memes.py     EDA helper
  cluster_entrypoint.sh        PYTHONPATH wrapper for Run:AI jobs
cluster/
  cluster.sh                   Project-side cluster automation (build, sync, submit, pull)
  config.env                   Project, paths, image repo
  state.local.env              Active image tag (gitignored)
  sync-code.exclude            rsync exclude list
tests/                         Pytest unit tests
data/processed/splits/
  train_split.json             Deterministic 8000/500 ID partition (committed)
cluster-results/, cluster-checkpoints/   Pulled job artifacts (gitignored)
project_planning/              Architecture spec, worklog, completion report
docs/                          Cluster-data and split-policy notes
Dockerfile                     Custom image (open_clip + pre-baked CLIP weights)
requirements.txt               Pip dependencies
```

## Dataset

**Already staged on the EPFL group scratch** at `/scratch/datasets/hate_meta`
(visible from inside Run:AI jobs). 8 500 labelled train, 500 labelled dev,
1 000 unlabelled test, 10 000 PNGs. No download or registration is required
for cluster runs. See [`docs/CLUSTER_DATA.md`](docs/CLUSTER_DATA.md) and
[`model_architecture.md` §4](project_planning/model_architecture.md).

The 8 000 / 500 train / held-out partition is precomputed (seed 0, stratified)
and committed at `data/processed/splits/train_split.json`. Reproduce with:

```bash
python scripts/make_train_split.py \
    --train-jsonl /scratch/datasets/hate_meta/train.jsonl \
    --out data/processed/splits/train_split.json \
    --held-out 500 --seed 0
```

## Running the code

There are two tracks. Track A is local and proves the code is correct. Track B
runs the actual experiments on the cluster.

### Track A — local (no GPU, no dataset)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . -r requirements.txt

pytest tests/                                                 # 7 passed, 4 skipped
python -m robust_meme_hate_detection.tests.smoke_synthetic    # forward + backward + PGD
```

The 4 skipped tests need the dataset on disk; they run on the cluster.

### Track B — EPFL Run:AI cluster

Preflight:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/epfl-ssh-key
./cluster/cluster.sh doctor          # ssh, rsync, docker, runai, scratch, dataset
./cluster/cluster.sh remote-check    # jumphost-side state
```

Custom image (only when `Dockerfile` or `requirements.txt` change):

```bash
./cluster/cluster.sh build-image v0.2
./cluster/cluster.sh push-image  v0.2
```

Sync code to jumphost:

```bash
./cluster/cluster.sh sync-code
```

Real-GPU smoke (4 batches, ~3 s on A100):

```bash
./cluster/cluster.sh submit-smoke smoke-stage1
./cluster/cluster.sh wait-job     smoke-stage1
./cluster/cluster.sh pull-artifacts smoke-stage1
```

Stage-1 training (submit → wait → pull → summarize, ~140 s per seed):

```bash
./cluster/cluster.sh run-train stage1-seed0
./cluster/cluster.sh run-train stage1-seed1
./cluster/cluster.sh run-train stage1-seed2
```

Eval — clean metrics + threshold sweep + PGD smoke:

```bash
./cluster/cluster.sh submit-cmd eval-seed0 -- \
    python -m robust_meme_hate_detection.eval.run_eval \
        --ckpt /scratch/robust-meme-hate-detection/experiments/stage1-seed0-*/ckpt/best.pt \
        --config configs/stage1.yaml \
        --out   /scratch/robust-meme-hate-detection/experiments/eval-seed0
./cluster/cluster.sh wait-job        eval-seed0
./cluster/cluster.sh pull-artifacts  eval-seed0
```

Artifacts land in `cluster-results/<job>/` (metrics, eval JSON, step logs) and
`cluster-checkpoints/<job>/ckpt/best.pt`.

## Reproducibility

- All training uses `utils/seeding.seed_everything` (Python, NumPy, PyTorch).
- The train / held-out partition is a committed JSON ID list — never resampled
  per run.
- Each cluster job writes a `metrics.json` with hyperparameters, env, and
  per-epoch metrics; `eval/run_eval.py` writes a single `eval.json` with the
  threshold grid and per-attack numbers.
- Image: `registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2`
  with CLIP weights pre-baked at build time.

## Key design choices (and why)

- **Frozen encoders, small fusion head** — 1.05 M trainable params on top of
  ~152 M frozen. Stage 1 trains in seconds and clears the AUROC bar; keeps
  Stage-3 attribution clean (changes are due to the defense, not the
  architecture).
- **`Normalize` inside the model** — input to `forward` is a `[0,1]` tensor
  with `requires_grad=True`. PGD just calls `torch.autograd.grad` against it.
- **No Hugging Face `transformers.CLIPModel`** — OpenCLIP exposes the same
  weights with simpler module names, which makes the optional last-block
  unfreeze (Stage 2) trivial.

## Documentation

- [`project_planning/model_architecture.md`](project_planning/model_architecture.md) — full spec
- [`project_planning/Phase1-2_Completion_Report.md`](project_planning/Phase1-2_Completion_Report.md) — results
- [`project_planning/Implementation_Worklog.md`](project_planning/Implementation_Worklog.md) — chronology
- [`docs/CLUSTER_DATA.md`](docs/CLUSTER_DATA.md) — dataset placement on cluster
- [`docs/DATA.md`](docs/DATA.md) — split policy
