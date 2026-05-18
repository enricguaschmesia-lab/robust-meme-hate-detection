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

**Phase 3 (perturbation benchmark) — done.** Eight text attacks (`leetspeak`,
`char_deletion`, `char_swap`, `spacing`, `punctuation`, `case_noise`,
`censoring`, `keyboard_typo`) and ten image attacks (`gaussian_noise`, `blur`,
`compression`, `brightness_up/down`, `contrast_up/down`, `translation`, `crop`,
`occlusion`) at three severity levels each, with deterministic SHA-256-keyed
per-cell seeds. Severities recalibrated against the manual review notes in
[`project_planning/perturbations_calibration.md`](project_planning/perturbations_calibration.md).
Driven by `eval/run_perturbed.py` (benchmark) and
`eval/inspect_perturbations.py` (inspection set).

**Phase 4 (robustness eval) — done.** Naturalistic benchmark against
`stage1-seed{0,1,2}` + `baseline-text-seed0` + `baseline-image-seed0`, and an
L∞ ε ∈ {1,2,4,8}/255 × {FGSM, PGD-10} white-box image-attack sweep against
the multimodal seeds + image-only baseline (new
`eval/run_whitebox.py`). Headline: white-box PGD ≥ ε=2/255 fully breaks the
model (ASR ≥ 0.98); among naturalistic attacks the most damaging are
high-severity text edits (`censoring`, `char_deletion`, `leetspeak`)
dropping multimodal AUROC by ~0.11 — and the multimodal model is *more*
text-fragile than the text-only baseline, suggesting text dominates the
multimodal decision. Full tables, figures, and worst-case analysis:
[`project_planning/Phase3_4_Completion_Report.md`](project_planning/Phase3_4_Completion_Report.md)
and `project_planning/phase4/`. Aggregator: `scripts/aggregate_phase4.py`
(local, no GPU). Next: Phase 5 robust training (KL consistency loss; severity
mix weighted toward high-severity text per Phase-4 findings).

**Phase 5 (robust training) — done.** Three robust recipes: `augonly`
(`BCE_clean + BCE_pert`), `kl` (adds `β·KL_full + γ·KL_image_branch`),
and `kldrop` (`kl` + per-example text-modality dropout p=0.30, which
forces the fusion head to make a real image-only prediction 30 % of
the time). Per-batch on-the-fly text/image augmentation via
`train/_robust_augmenter.py`, sampling `{text, image, both}` with
`(0.5, 0.3, 0.2)` weights. Nine trainings + 27 evals.

Headline (3-seed mean), reported under the project's primary threat
model (naturalistic attacks; white-box PGD reported separately):

- **`kl` is the clean-accuracy-preserving champion**: 0.737 clean AUROC
  (vs 0.742 clean baseline), naturalistic fully-robust count
  **105 → 134 / 500** (+28 % relative, the headline robustness win).
- **`kldrop` revives the dead image branch**: image-only forward AUROC
  goes **0.593 → 0.636**, exceeding the dedicated image-only baseline
  of 0.628 (the only recipe to do so). Best per-cell text robustness
  (worst-cell text ΔAUROC 0.115 → 0.067, 42 % reduction). Cost:
  0.037 AUROC on clean inputs.
- **White-box PGD** remains a hard floor for all recipes at ε ≥ 2/255
  (expected non-target; would require adversarial training).

Full write-up:
[`project_planning/Phase5_Completion_Report.md`](project_planning/Phase5_Completion_Report.md);
robust-vs-clean comparison table in `project_planning/phase4/robust_vs_clean.md`.

**Phase 5c (generalisation analysis) — done.** Three orthogonal
extensions on Phase 5b: in-pool vs OOD attack reporting, composite
attacks (multiple perturbations per sample), and severity-restricted
training (`kl_lowmed`).

Findings (3-seed mean):

- **Augmentation gain transfers to held-out image attacks** — Δ(OOD−in)
  negative for every robust recipe; `kldrop` shows the strongest
  transfer (Δ −0.005 image OOD).
- **Composite attacks are the worst threat in the benchmark**:
  clean-ckpt `composite_2text_2image` at high severity hits ΔAUROC
  0.145, exceeding the single-cell worst of 0.115. `kldrop` is the
  strict Pareto winner on every composite cell — cuts the realistic
  medium-severity `composite_2text` gap by 43 % vs clean
  (0.090 → 0.051) and `composite_text_image` by 51 % (0.050 → 0.024).
- **`kl_lowmed` (severity-restricted) is Pareto-dominated by `kl`** —
  clean negative result: high-severity augmentation is necessary, not
  optional. Removing it loses 0.5 % AUROC at medium severity and
  4 dev examples on naturalistic survival.

For the project's primary threat model (internet user combining 1–2
perturbations at moderate severity), `kldrop` is the recommended ckpt.
Full write-up in `project_planning/Phase5_Completion_Report.md` § 13;
data in `project_planning/phase4/robust_vs_clean.md`.

**Phase 6 (failure analysis narrative) — done.** Extended
`scripts/failure_analysis.py` to multi-seed, multi-recipe,
composite-aware (new bucket `B5_composite_only_failure`). Strongest
finding: `kldrop` and `kl` trade off **class-asymmetrically**. The
34 examples `kldrop` saves vs `kl` are 33/34 label=0 (non-hate —
revived image branch prevents text-attack false positives); the 49
examples `kl` saves vs `kldrop` are 46/49 label=1 (hate —
preserved clean accuracy keeps borderline-hate predictions on the
right side of the threshold after text-attack margin erosion). The
strongest deployment-conditional finding of the project:
false-positive-sensitive contexts prefer `kldrop`; recall-sensitive
contexts prefer `kl`. Full write-up in
`project_planning/Phase6_Completion_Report.md`.

**Phase 7 (held-out test evaluation) — done.** All five recipes
re-evaluated on the labelled `test_seen` (n=1000, 49 % pos) and
`test_unseen` (n=2000, 37.5 % pos) splits. 90 cluster jobs
(5 recipes × 3 seeds × 2 splits × 3 eval kinds). Labelled jsonls
acquired from the `neuralcatcher/hateful_memes` HF mirror; the
2000 test_unseen images sourced from `limjiayi/hateful_memes_expanded`
and staged on cluster scratch. All numbers in
`project_planning/phase4/robust_vs_clean.test_seen.md` and
`robust_vs_clean.test_unseen.md`.

Headline reproductions:

- **`kldrop` is the worst-cell text robustness winner on every split**
  (0.067 dev → 0.074 test_seen → **0.055 test_unseen** — the
  project's lowest single-cell text gap is on the held-out
  naturalistic-prior split).
- **`kldrop` image-only AUROC exceeds the dedicated image-only baseline
  on every split** (0.636 dev / 0.641 test_seen / **0.655 test_unseen**
  vs baseline 0.628). Modality dropout's image-branch revival
  generalises to held-out data.
- **`kl` preserves clean AUROC on both test splits within seed noise**
  (0.745 vs clean 0.747 on test_seen; 0.738 = clean 0.738 on
  test_unseen).
- **`kldrop` is the medium-severity composite winner on 3/4 cells**
  on test_seen and test_unseen (matches dev).
- **`kllowmed` partial refutation**: unexpectedly has the highest
  naturalistic-robust count on test_unseen (387 / 2000) — beating
  `kl` (372.7). The Phase 5c-3 "strictly Pareto-dominated" claim
  doesn't hold on the naturalistic-prior test split; recipe is now
  reported as a borderline case rather than a clean negative
  result.

Updated Pareto recommendation: `kldrop` for the realistic composite
threat model (project's recommended ckpt); `kl` for clean-accuracy-
preserving; `augonly` promoted to "secondary alternative" on
test_seen; `kllowmed` is the borderline result. Full write-up in
`project_planning/Phase7_Completion_Report.md`.

**Phase 8 (test-side failure analysis) — done.** Ran
`scripts/failure_analysis.py` on test_seen and test_unseen
(after fixing a bug where `--split` only changed the caption file,
not the cluster-results lookup). The Phase 6 class-asymmetric
finding partially reproduces: kldrop-wins-toward-label=0 holds on
both test splits (66 % / 71 %, vs 97 % dev — smaller effect size)
but kl-wins-toward-label=1 breaks on test_unseen (43 %). The
defensible held-out claim: "kldrop systematically reduces
text-attack-induced false-positive flips on non-hate examples
(image-branch revival mechanism)." Full write-up in
`project_planning/Phase8_TestFailure_Report.md`.

**Phase 9 (mixed composites + dropout-rate sweep) — done.**

- **9a (mixed-severity composites)**: extends `_apply_composite` so
  each component samples severity ∈ {low, medium, high}
  independently. New `mixed` severity column in every composite
  table. Finding: mixed ≈ medium on every (recipe, composite,
  split) cell — realistic adversarial-user threat is medium-class,
  not high-class. Validates the project's "medium = realistic"
  framing.
- **9b (modality-dropout-rate sweep)**: trains and evaluates
  `kldrop-p015` (p=0.15) and `kldrop-p050` (p=0.50) alongside the
  existing `kldrop` (p=0.30). **The untuned p=0.30 was *not*
  Pareto-optimal.** `kldrop-p015` strictly dominates `kldrop` on
  every split: image-branch revival within seed noise (image-only
  AUROC 0.638 dev / 0.638 test_seen / 0.658 test_unseen) AND clean
  AUROC within seed noise of the `kl` recipe (0.735 / 0.748 /
  0.743). **New recommended default: `kldrop-p015`**, with
  `kldrop-p050` as the conservative alternative for composite-
  heaviest threat models. Original `kldrop` (p=0.30) is now
  Pareto-dominated. Reports: `project_planning/Phase9a_MixedComposites_Report.md`,
  `project_planning/Phase9b_DropoutSweep_Report.md`.

**Poster figures.** `scripts/make_poster_figures.py` regenerates
6 publication-styled figures (PNG + PDF + caption) into
`project_planning/poster_figures/`:
headline Pareto, image-branch revival, composite escalation,
class-asymmetric trade-off, severity curves, dropout sweep.
Reuses the aggregator + failure_analysis loaders so figures
stay in sync with the per-phase tables.

Reproducibility:
```bash
PYTHONPATH=src .venv/bin/python3 scripts/aggregate_phase4.py
PYTHONPATH=src .venv/bin/python3 scripts/make_poster_figures.py
```

**Important caveats** (also documented in the per-phase limitations sections):

- **PGD ε ≥ 2/255 is fully undefended** on every recipe across all three
  splits (AUROC ≈ 0). Naturalistic-only is the project's threat-model
  scope; adversarial training is owned separately.
- **Class-asymmetric finding refreshed for `kldrop-p015`.** Phase 6's
  original dev finding (kldrop saves label=0, kl saves label=1) was on
  the now-Pareto-dominated p=0.30 recipe with small disagreement
  samples. On the recommended `kldrop-p015`, the label=0 bias on its
  wins side is **stronger** on test_unseen (97 % at n=131, 95 % CI
  94–99 %) than the original kldrop-vs-kl finding (74 % at n=85).
  Phase 8 § 4 has the full reproduction; figure 4 in the poster uses
  the refreshed pair.
- **Dropout sweep tested at p ∈ {0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50}**;
  `kldrop-p015` wins the headline metrics. Out of that sweep range,
  global optimality is not claimed.
- **Augmentation has a ceiling around 70 % consensus failures** on
  test (66 % on dev, 72 % on test_seen, 73 % on test_unseen). Even
  the best recipe leaves the majority of examples vulnerable to
  *some* perturbation in the 73-cell grid; pushing further would
  likely require a stronger text encoder or representation-level
  robustness loss rather than more augmentation.

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
  stage1_robust_augonly.yaml   Robust trainer: BCE_clean + BCE_pert (no KL)
  stage1_robust_kl.yaml        Robust trainer: + β·KL_full + γ·KL_image_branch
  stage1_robust_kl_drop.yaml   Robust trainer: kl + per-example text-modality dropout (p=0.30)
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
  train/stage1_robust.py       Phase-5 robust trainer (on-the-fly aug + optional KL terms)
  train/_robust_augmenter.py   RobustAugmenter — per-example text/image perturbation sampler
  train/_robust_loss.py        binary_kl + robust_loss utilities
  train/baseline_majority.py   Majority-class predict-0 baseline (writes same metrics.json schema)
  eval/run_eval.py             Clean + threshold sweep + PGD eval
  eval/run_perturbed.py        Phase-3 perturbation benchmark (attack × severity grid → JSON)
  eval/run_whitebox.py         Phase-4 white-box FGSM/PGD ε-sweep (→ whitebox_eval.json)
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
  aggregate_phase4.py          Local Phase-4/5 aggregator: reads all perturbed_/whitebox_eval.json
                               + modality_ablation.json under cluster-results/ and writes
                               project_planning/phase4/ tables + figures (incl. robust_vs_clean.md)
  failure_analysis.py          Per-example failure bucketing + robust-status annotation
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
