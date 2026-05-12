# Robust Meme Hate Detection

Course project (EE-559, EPFL): **adversarial robustness of multimodal hateful
meme detectors**. We train a CLIP-based dual-encoder classifier on the Hateful
Memes dataset, attack it with FGSM/PGD and a naturalistic perturbation suite
(8 text × 10 image attacks × 3 severities, deterministically seeded), and
study a robust training variant that combines perturbation augmentation with a
KL consistency loss.

## Status

Phase 1-2 (baseline + attack-readiness) is **complete**, including all four
baselines required by the roadmap. See
[`project_planning/Phase1-2_Completion_Report.md`](project_planning/Phase1-2_Completion_Report.md).

**Clean dev results (500 examples, τ=0.5)** — multimodal beats best unimodal by
+0.11 AUROC, confirming the fusion classifier is not a text shortcut:

| Baseline | AUROC | macro F1 | accuracy | trainable params |
|---|---:|---:|---:|---:|
| Majority class (predict 0)              | 0.5000 | 0.3333 | 0.500 | 0 |
| Image-only CLIP + head (seed 0)         | 0.6232 | 0.5738 | 0.574 | 1.05 M |
| Text-only CLIP + head (seed 0)          | 0.6321 | 0.5966 | 0.598 | 1.05 M |
| **Multimodal CLIP fusion (seed 0)**     | **0.7432** | **0.6373** | **0.648** | **1.05 M** |
| Multimodal seed 1 / seed 2              | 0.7254 / 0.7329 | 0.6509 / 0.6388 | 0.652 / 0.648 | 1.05 M |
| Multimodal **best-τ** (τ=0.41, seed 0)  | 0.7432 | **0.6725** | 0.676 | 1.05 M |
| Multimodal mean ± σ across 3 seeds      | **0.7338 ± 0.0073** | 0.6423 ± 0.0061 | 0.649 ± 0.002 | — |

**Attack-readiness smoke (PGD ε=4/255, 5 steps, seed 0)**:

| Metric (at τ=0.41) | Clean | After PGD | ASR |
|---|---:|---:|---:|
| AUROC    | 0.7432 | 0.0004 | — |
| accuracy | 0.6760 | 0.0080 | **98.8%** |

Wall-clock per Stage-1 seed: ~140 s on A100 80 GB. Total Phase 1-2 cluster
budget: < 1 A100-hour.

**Phase 3 (perturbation benchmark) — code merged, label-preservation review in
progress.** Eight text attacks (`leetspeak`, `char_deletion`, `char_swap`,
`spacing`, `punctuation`, `case_noise`, `censoring`, `keyboard_typo`) and ten
image attacks (`gaussian_noise`, `blur`, `compression`, `brightness_up/down`,
`contrast_up/down`, `translation`, `crop`, `occlusion`) at three severity
levels each, with deterministic SHA-256-keyed per-cell seeds. Driven by
`eval/run_perturbed.py` (benchmark) and `eval/inspect_perturbations.py`
(inspection set for the manual 50–100 label-preservation check). Next: Phase 4
robustness eval against `stage1-seed{0,1,2}`, then Phase 5 robust training.

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
  stage1.yaml                  Frozen-encoder multimodal training
  stage1_text.yaml             Text-only baseline
  stage1_image.yaml            Image-only baseline
  smoke.yaml                   Short cluster smoke
  data/hateful_memes.yaml      Dataset paths
src/robust_meme_hate_detection/
  models/clip_fusion.py        CLIPHateMemeClassifier + Normalize + FusionHead
  data/                        Hateful Memes loader, splits, transforms
  attacks/pgd.py               FGSM / PGD on [0,1] images
  perturbations/               Naturalistic text + image perturbations
                               (TextPerturbation, ImagePerturbation, ComposePerturbation;
                                from_preset(level) + SHA-256-keyed seeds)
  train/stage1.py              Frozen-encoder trainer (--modality multimodal|text|image)
  train/baseline_majority.py   Majority-class predict-0 baseline (writes same metrics.json schema)
  eval/run_eval.py             Clean + threshold sweep + PGD eval
  eval/run_perturbed.py        Phase-3 perturbation benchmark (attack × severity grid → JSON)
  eval/inspect_perturbations.py Cluster entrypoint for the label-preservation inspection set
  eval/metrics.py              Accuracy, F1, AUROC, ASR, robustness gap
  utils/                       Seeding, JSON logging, threshold sweep
  tests/smoke_synthetic.py     CPU-only synthetic forward+backward+PGD test
  tests/smoke_cluster.py       Cluster-side real-GPU smoke
scripts/
  make_train_split.py          One-shot 8000/500 stratified ID partition
  inspect_hateful_memes.py     EDA helper
  inspect_perturbations.py     Materialise a per-cell (attack/severity) inspection set
                               for the manual 50–100 label-preservation review
  example_perturbations.py     Legacy perturbation demo on a tiny dataset slice
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

### Track B — EPFL Run:AI cluster (manual workflow)

Use this path if you access the cluster manually (no laptop-side automation
set up). All steps run on the jumphost.

```bash
ssh <your-epfl-username>@<jumphost>
git clone <this-repo> ~/robust-meme-hate-detection      # one-off
cd ~/robust-meme-hate-detection
git pull                                                 # before each session

# Make sure runai points at YOUR project, not someone else's.
runai config project course-ee-559-<your-epfl-username>

# Edit `scripts/runai_submit_template.sh`:
#   - set JOB to a unique name
#   - set IMAGE (use a teammate's pushed image or your own)
#   - uncomment exactly one CMD preset (smoke / train / eval)
bash scripts/runai_submit_template.sh

# Watch and inspect the job.
runai logs <job-name> -f
runai describe job <job-name>

# Results land on shared scratch, readable by all teammates:
ls /scratch/robust-meme-hate-detection/experiments/<job-name>/
```

The template does not depend on `cluster.sh`. It is the recommended entry
point for collaborators. The smoke preset is the fastest way to confirm your
runai project, mounts, and image are wired up correctly before launching a
real training run.

### Track B — EPFL Run:AI cluster (automated workflow)

If you maintain a laptop-side setup with SSH agent + Docker login + your own
`cluster/config.env` (copy `cluster/config.env.example` and fill in your
values), you can use `cluster/cluster.sh` to drive everything from your
laptop.

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
    /home/guasch/robust-meme-hate-detection/scripts/cluster_entrypoint.sh \
    robust_meme_hate_detection.eval.run_eval \
        --ckpt /scratch/robust-meme-hate-detection/experiments/stage1-seed0-*/ckpt/best.pt \
        --config configs/stage1.yaml \
        --out   /scratch/robust-meme-hate-detection/experiments/eval-seed0
./cluster/cluster.sh wait-job        eval-seed0
./cluster/cluster.sh pull-artifacts  eval-seed0
```

Unimodal / majority baselines (single seed):

```bash
./cluster/cluster.sh submit-train baseline-text-seed0  --config configs/stage1_text.yaml  --seed 0
./cluster/cluster.sh submit-train baseline-image-seed0 --config configs/stage1_image.yaml --seed 0
./cluster/cluster.sh submit-cmd   baseline-majority-seed0 -- \
    /home/guasch/robust-meme-hate-detection/scripts/cluster_entrypoint.sh \
    robust_meme_hate_detection.train.baseline_majority \
        --config configs/stage1.yaml --seed 0 \
        --out /scratch/robust-meme-hate-detection/experiments/baseline-majority-seed0
```

Perturbation inspection set (Phase 3 label-preservation review; no GPU needed):

```bash
./cluster/cluster.sh submit-cmd inspect-perturb-dev -- \
    /home/guasch/robust-meme-hate-detection/scripts/cluster_entrypoint.sh \
    robust_meme_hate_detection.eval.inspect_perturbations \
        --dataset-root /scratch/datasets/hate_meta \
        --out /scratch/robust-meme-hate-detection/experiments/inspect-perturb-dev \
        --split dev --n-samples 1000 --seed 0
./cluster/cluster.sh wait-job       inspect-perturb-dev
./cluster/cluster.sh pull-artifacts inspect-perturb-dev
```

Perturbation benchmark (Phase 3 against a trained checkpoint):

```bash
./cluster/cluster.sh submit-cmd perturbed-seed0 -- \
    /home/guasch/robust-meme-hate-detection/scripts/cluster_entrypoint.sh \
    robust_meme_hate_detection.eval.run_perturbed \
        --ckpt /scratch/robust-meme-hate-detection/experiments/stage1-seed0-*/ckpt/best.pt \
        --config configs/stage1.yaml \
        --out   /scratch/robust-meme-hate-detection/experiments/perturbed-seed0
./cluster/cluster.sh wait-job       perturbed-seed0
./cluster/cluster.sh pull-artifacts perturbed-seed0
```

Artifacts land in `cluster-results/<job>/` (metrics, eval JSON, perturbation
manifest + samples, step logs) and `cluster-checkpoints/<job>/ckpt/best.pt`.

## Phase 3 label-preservation review (50–100 manual samples)

The roadmap requires confirming that low/medium-severity perturbations do not
change the human label. Once `pull-artifacts` lands the inspection job in
`cluster-results/<job>/`, the layout is:

```text
cluster-results/<job>/
  manifest.csv                              # one row per (attack, level, sample)
  summary.json
  text/<attack>/<level>/pairs.csv           # id, label, original, perturbed
  image/<attack>/<level>/<id>.png           # perturbed view
  image/<attack>/<level>/_orig/<id>.png     # original (same id)
```

A practical review protocol that satisfies the DoD with ~80 spot checks:

```bash
JOB=inspect-perturb-...                  # the timestamped job dir under cluster-results/

# 1. Text — read a random 5 rows per (attack × level) cell. 24 text cells × 5 = 120 rows.
for f in cluster-results/$JOB/text/*/*/pairs.csv; do
  echo "=== $f"; shuf -n 5 "$f"
done | less

# 2. Image — open the per-cell folders side-by-side with their _orig/ siblings.
xdg-open cluster-results/$JOB/image/blur/medium &
xdg-open cluster-results/$JOB/image/blur/medium/_orig &
# Repeat for the cells you want to sanity-check (compression / brightness_down
# / occlusion are the ones most likely to break readability).

# 3. Optional: random subset across all cells, weighted by modality.
shuf -n 80 cluster-results/$JOB/manifest.csv > /tmp/review_sample.csv
```

For each row, judge: *does the human-readable hate vs non-hate label still
apply after the perturbation?* Tally the disagreements per attack/level; report
any cell with > ~5 % label drift. High severity may drop meaning by design and
is reported separately, per the roadmap.

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
- [`project_planning/Phase1-2_Completion_Report.md`](project_planning/Phase1-2_Completion_Report.md) — Phase 1-2 results (including unimodal + majority baselines)
- [`project_planning/ProjectIdeaHenrik_Feasibility_Roadmap.md`](project_planning/ProjectIdeaHenrik_Feasibility_Roadmap.md) — full project roadmap
- [`project_planning/Implementation_Worklog.md`](project_planning/Implementation_Worklog.md) — chronology
- [`docs/CLUSTER_DATA.md`](docs/CLUSTER_DATA.md) — dataset placement on cluster
- [`docs/DATA.md`](docs/DATA.md) — split policy
- [`docs/PERTURBATIONS.md`](docs/PERTURBATIONS.md) — perturbation suite spec
