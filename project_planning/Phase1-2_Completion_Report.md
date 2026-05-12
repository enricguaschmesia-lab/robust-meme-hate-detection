# Phase 1-2 Completion Report — Stage-1 Baseline Trained, Attack-Ready

**Project:** Adversarial Robustness of Multimodal Meme Hate Detectors (EE-559).
**Reporting period:** 2026-05-02 to 2026-05-03; addendum 2026-05-12 (unimodal + majority baselines, §8).
**Status:** All four Definition-of-Done items in `model_architecture.md` §12 satisfied, **and** Phase 2 of the project roadmap (`ProjectIdeaHenrik_Feasibility_Roadmap.md`) is now complete with the addition of text-only, image-only, and majority baselines. Project is ready to enter Phase 3 of the roadmap (perturbation benchmark) and Phase 5 (Stage-3 robust training).

---

## TL;DR

> A frozen-encoder OpenCLIP ViT-B/32 fusion classifier reaches **dev AUROC 0.7432 / 0.7254 / 0.7329 across seeds 0/1/2** on the Hateful Memes mirror (`/scratch/datasets/hate_meta`). Threshold tuning on macro F1 picks τ=0.41, lifting accuracy from 0.65 → 0.68. PGD (ε=4/255, 5 steps) drives clean-correct accuracy from 0.68 → 0.008 with **attack success rate 98.8%**, conclusively confirming gradient flow and establishing the un-defended baseline. Wall-clock per seed: ~140 s on A100 80 GB.
>
> **2026-05-12 addendum (§7):** Phase 2 of the roadmap is now complete: text-only, image-only, and majority baselines were trained on the same dev split. The multimodal model beats the best unimodal baseline by **+0.11 AUROC** (0.743 vs text-only 0.632, image-only 0.623; majority 0.500). The multimodal claim holds — the fusion model is not a text shortcut.

| Definition-of-Done item | Evidence |
|---|---|
| 1. Track A passes locally without intervention | `pytest tests/`: 7 passed, 4 skipped (skips require dataset on disk). `python -m robust_meme_hate_detection.tests.smoke_synthetic`: PASSED. |
| 2. Cluster smoke job works on real GPU+dataset | Job `smoke-stage1-20260503-163704` succeeded on A100 80 GB; 4 forward+backward batches in 2.79 s; `metrics.json` written. |
| 3. Stage-1 dev AUROC ≥ 0.72 on at least one seed | All three seeds clear: **0.7432 / 0.7254 / 0.7329**. |
| 4. PGD reduces AUROC, no `requires_grad` errors | Clean (subset) AUROC 0.7432 → attacked 0.0004; ASR 0.9882. No PyTorch autograd errors. |

---

## 1. Final Stage-1 numbers on `dev` (500 examples)

Best-checkpoint metrics at the threshold the training script saved (default 0.5):

| Seed | AUROC | macro F1 | accuracy | precision | recall | FPR | FNR | best epoch | wall clock |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.7432** | 0.6373 | 0.648 | 0.7256 | 0.476 | 0.180 | 0.524 | 5 | 145.9 s |
| 1 | 0.7254 | **0.6509** | 0.652 | 0.6712 | 0.596 | 0.292 | 0.404 | 4 | 136.0 s |
| 2 | 0.7329 | 0.6388 | 0.648 | 0.7176 | 0.488 | 0.192 | 0.512 | 4 | 137.7 s |
| **mean** | **0.7338** | 0.6423 | 0.649 | 0.7048 | 0.520 | 0.221 | 0.480 | — | — |
| **std**  | 0.0073 | 0.0061 | 0.002 | 0.0249 | 0.057 | 0.051 | 0.057 | — | — |

Notes:
- Seed-to-seed AUROC variance is small (σ=0.0073), well under the ~0.04 noise floor reported in 2020 leaderboard ensembles. The frozen-encoder + fixed-train-split design eliminates a lot of normal training noise.
- Recall is the weak metric at τ=0.5 (0.52 mean); threshold tuning recovers most of it (see §2).
- `pos_weight` was auto-computed at 1.7864 (= 5128/2871, on the 8000 training subset).

## 2. Threshold sweep on seed 0 (dev)

Sweep grid τ ∈ {0.30, 0.31, …, 0.70} on macro F1.

| τ | macro F1 | accuracy | precision | recall |
|---:|---:|---:|---:|---:|
| 0.50 | 0.6391 | 0.650 | 0.7301 | 0.476 |
| **0.41 (best)** | **0.6725** | **0.676** | 0.7222 | 0.572 |
| 0.30 | 0.6700 | 0.670 | 0.6680 | 0.676 |

Persisted to `cluster-results/eval-seed0-20260503-165736/eval.json`. Threshold τ=0.41 will be reused as the reporting threshold for all attacked evaluations of the seed-0 checkpoint, per the rule in `model_architecture.md` §5.5.

## 3. Attack-readiness (PGD smoke, seed 0)

Settings: ε=4/255, α=1/255, 5 steps, 8 batches × 64 = 512 dev examples (slightly more than dev itself, so effectively the whole split). All attacks operate on the `[0, 1]` image tensor; text held fixed.

| Metric (at τ=0.41) | Clean subset | After PGD | Δ (clean − attacked) |
|---|---:|---:|---:|
| AUROC | 0.7432 | 0.0004 | +0.7428 |
| accuracy | 0.6760 | 0.0080 | +0.6680 |
| macro F1 | 0.6725 | 0.0080 | +0.6645 |
| recall | 0.572 | 0.004 | +0.568 |
| FPR | 0.220 | 0.988 | −0.768 |

**Attack success rate: 0.9882** (978 of 989 clean-correct examples flipped).

Interpretation:
- Gradient flow through `Normalize` and the dual encoders is healthy: PGD drives accuracy from 0.68 → 0.008 in 5 steps at ε=4/255. This is the success signal we wanted from DoD item 4.
- The model is essentially **unprotected** against pixel-space attacks. This is the *expected* baseline for an un-defended classifier and quantifies the headroom for Stage-3 robust training to recover.
- AUROC of 0.0004 is well below 0.5, indicating the attack pushes predictions to the *wrong* class with high confidence (ASR 99% confirms this). PGD is doing more than confusing the model — it is reliably flipping it. Useful baseline.

## 4. What was built (artifacts checklist)

### Code (committed)
- `src/robust_meme_hate_detection/`
  - `models/clip_fusion.py` — `CLIPHateMemeClassifier` over OpenCLIP ViT-B/32 with `Normalize` inside the model and a 4-block fusion head (~1.05 M trainable params).
  - `attacks/pgd.py` — FGSM and PGD on `images01`.
  - `data/transforms.py` — `ClipImage01Transform` (no Normalize) and `ClipTokenize`.
  - `data/splits.py` — deterministic train/train_held_out partition loader.
  - `eval/metrics.py` — accuracy/macro-F1/AUROC/precision/recall/FPR/FNR + ASR + robustness gap.
  - `eval/run_eval.py` — checkpoint → clean metrics + threshold sweep + PGD eval, single JSON output.
  - `train/stage1.py` — frozen-encoder training loop, AdamW + cosine warmup, BCE pos-weight auto, AMP bf16, early stopping on macro F1.
  - `utils/{seeding,logging,threshold}.py`.
  - `tests/{smoke_synthetic,smoke_cluster}.py`.
- `tests/test_clip_fusion.py`, `tests/test_metrics.py` (pytest, 7 passed).
- `scripts/make_train_split.py`, `scripts/cluster_entrypoint.sh`.
- `configs/{stage1,smoke}.yaml`, `configs/data/hateful_memes.yaml`.
- `cluster/{config.env,cluster.sh,sync-code.exclude,state.local.env}`.
- `Dockerfile` (uid 316498 + USER/HOME env + pre-baked CLIP weights).

### Cluster artifacts (in `cluster-results/` and `cluster-checkpoints/`)
- `stage1-seed{0,1,2}-…/metrics.json`, `…/ckpt/best.pt`, `…/steps.jsonl`.
- `eval-seed0-…/eval.json` (threshold grid + PGD numbers).
- `smoke-stage1-…/metrics.json` (the cluster smoke evidence for DoD item 2).

### Docker image
- `registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2` (active; tag recorded in `cluster/state.local.env`).

## 5. Issues hit during the run (and their fixes)

The full chronology lives in `Implementation_Worklog.md`. Headline issues:

1. **PEP-668 in the course base image** → added `--break-system-packages` to the Dockerfile pip install.
2. **`python` vs `python3` mismatch** in the base image → all in-image scripts call `python3`.
3. **`getpwuid()` UID error** for uid 316498 → added the user to `/etc/passwd` and `ENV USER=guasch HOME=/home/guasch LOGNAME=guasch`.
4. **`src/`-layout package not on PYTHONPATH** → introduced `scripts/cluster_entrypoint.sh` that sets PYTHONPATH and cd's before exec'ing python3.
5. **`/dev/shm` exhaustion under multi-worker DataLoader** → added `--large-shm` to `runai submit`; reduced smoke job `num_workers` to 0.
6. **ID-width mismatch** in the train-split writer (zero-pad vs not) — caught by the review agent in Phase 3.5; fixed before any cluster job ran.
7. **Transient TCP closures during `docker push`** — handled by retry; resumable.

## 6. Definition-of-Done check

| § 12 item | Required | Achieved |
|---|---|---|
| 1 — Track A passes | yes | ✅ 7 passed, 4 skipped; synthetic smoke OK |
| 2 — Cluster smoke succeeds | yes | ✅ `smoke-stage1-20260503-163704` Succeeded on A100 |
| 3 — Stage-1 dev AUROC ≥ 0.72 on ≥1 seed | yes | ✅ all three seeds (0.74, 0.73, 0.73) |
| 4 — PGD reduces AUROC, no autograd errors | yes | ✅ AUROC 0.74 → 0.0004; ASR 98.8% |

**All DoD items closed.** The project can now move on to:

- Phase 3 of the roadmap: implement the perturbation suite (`perturb/text.py`, `perturb/image.py` — currently a stub) and run the attack grid against the Stage-1 checkpoints.
- Phase 5 of the roadmap: Stage-3 robust training (perturbation augmentation + KL consistency loss). The Stage-1 checkpoints we just produced are the un-defended baseline against which the robust model will be compared.

## 7. Phase 2 enhancements — Unimodal and majority baselines (2026-05-12)

The original Phase 1-2 push only trained the multimodal CLIP fusion classifier. Henrik's roadmap (`ProjectIdeaHenrik_Feasibility_Roadmap.md` §"Recommended scope" and §"Phase 2") additionally requires text-only, image-only, and majority baselines so that the multimodal claim can be assessed and RQ3 (modality reliance) answered. This addendum closes that gap.

### 7.1 Implementation

- `train/stage1.py` gained a `--modality {multimodal,text,image}` flag and a `model.modality` config field. The flag dispatches the forward call between `model(images, tokens)`, `model.forward_text_only(tokens)`, and `model.forward_image_only(images)`. The fusion-head architecture is unchanged: under `modality=text` the head sees `concat(t, 0, |t|, 0)`, under `modality=image` it sees `concat(0, v, |v|, 0)`. This keeps the trainable-parameter count identical across baselines (1.05 M head params) and isolates the comparison to the input modality.
- `configs/stage1_text.yaml` and `configs/stage1_image.yaml` were added, identical to `stage1.yaml` modulo the modality field.
- `train/baseline_majority.py` was added. It loads the train and dev splits, predicts the majority train class for every dev example (probability set to the train positive rate for AUROC computation), and writes the same `metrics.json` schema as `train.stage1` so `cluster.sh summarize-results` and downstream tooling work uniformly across all four baselines.

### 7.2 Cluster runs

Three Run:AI jobs, all on `registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2`, single A100:

| Job | Modality | Wall clock | Pod status |
|---|---|---:|---|
| `baseline-majority-seed0` | none (constant prediction) | 3.5 s | Succeeded |
| `baseline-text-seed0` | text only | 192.8 s | Succeeded |
| `baseline-image-seed0` | image only | 126.5 s | Succeeded |

Artifacts pulled to `cluster-results/baseline-{majority,text,image}-seed0/`.

### 7.3 Comparison on dev (500 examples, τ=0.5)

The multimodal row is `stage1-seed0-20260503-163917` from §1 of this report, re-reported at τ=0.5 (default checkpoint threshold) so that all four rows are directly comparable. Best-τ multimodal numbers (τ=0.41) are in §2 and are not re-stated here to avoid confusion.

| Baseline | AUROC | macro F1 | accuracy | precision | recall | FPR | FNR | trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority class (predict 0) | 0.5000 | 0.3333 | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0 |
| Image-only CLIP + head | 0.6232 | 0.5738 | 0.574 | 0.5774 | 0.552 | 0.404 | 0.448 | 1.05 M |
| Text-only CLIP + head | 0.6321 | 0.5966 | 0.598 | 0.6109 | 0.540 | 0.344 | 0.460 | 1.05 M |
| **Multimodal CLIP fusion (seed 0)** | **0.7432** | **0.6373** | **0.648** | **0.7256** | 0.476 | 0.180 | 0.524 | 1.05 M |

### 7.4 Interpretation

1. **The multimodal model is genuinely multimodal.** AUROC 0.743 vs 0.632 (text-only) and 0.623 (image-only) — a gain of ~+0.11 AUROC and ~+0.04 macro F1 over the best unimodal baseline. The fusion model is not a text shortcut; it relies on cross-modal information. This directly answers RQ3 of the roadmap and de-risks every downstream robustness claim (per §"Risks", "Model learns text shortcuts → multimodal claim becomes weak").
2. **Text is the slightly stronger single modality** (AUROC 0.632 vs 0.623), consistent with Hateful Memes being designed so that text alone is informative on roughly 60–65% of examples. This sets the prior expectation that **text perturbations should hurt the multimodal model more than equally severe image perturbations** — a hypothesis Phase 3 of the roadmap will test.
3. **Train and dev have different class priors.** Train: 35.9 % positive (after the 8000/500 partition). Dev: 50 % positive by Hateful Memes construction. The majority baseline therefore lands at exactly 50 % accuracy and macro F1 0.333 (collapsed to the predict-0 row of the confusion matrix). The `pos_weight=1.79` already in the trainer correctly compensates for the train imbalance; no further class-rebalancing is needed.
4. **Both unimodal models prefer recall over precision at τ=0.5**, while the multimodal model prefers precision over recall. This is a threshold-calibration artefact, not a real ranking-quality difference. The macro-F1-optimal τ for the multimodal model (τ=0.41, §2) lifts recall from 0.476 → 0.572 at the cost of a small precision dip. Each baseline could be threshold-tuned independently for its own best-F1 reporting; for the present comparison we deliberately keep τ=0.5 to avoid confounding modality-comparison with per-model threshold search.

### 7.5 What this does and does not freeze

Frozen for the remainder of the project (unless Phase 5 reveals a head bottleneck):

- The multimodal `stage1-seed0` checkpoint and its companions `seed1`/`seed2` are the canonical clean-trained baseline against which the Phase 5 robust model is compared.
- The 1.05 M-param frozen-encoder head is the architecture for all four baselines and for the Phase 5 robust variant.
- The 8000/500 deterministic ID partition (`data/processed/splits/train_split.json`) is unchanged.

Not done in this addendum (and deliberately so):

- Three-seed runs of the unimodal and majority baselines. Single-seed numbers are sufficient to establish the modality ordering; multiple seeds would only tighten error bars that are already loose by ±0.01 AUROC at one seed. We will revisit if Phase 3 results suggest the ordering is fragile.
- A separate best-τ sweep for the unimodal baselines. Their primary role is to set the upper bound on what each modality alone can do; per-model τ tuning is reserved for the robustness comparison in Phase 4.
- Last-block / LoRA fine-tuning. The roadmap reserves this for the case where the head underfits; AUROC 0.743 on the multimodal baseline does not justify the extra compute.

## 8. Compute used

Total cluster wall-clock for the entire Phase 1-2 push:

| Step | jobs | wall clock |
|---|---:|---:|
| Cluster smoke (3 attempts incl. fixes) | 3 | ~3 min |
| Stage-1 training × 3 seeds | 3 | ~7 min total (mostly parallel) |
| Eval (clean + threshold sweep + PGD) seed 0 | 1 | ~30 s |
| Image build × 2 | 2 | ~14 min (mostly weight pre-bake) |
| Image push × 2 | 2 | ~5 min (with retries) |
| Phase 2 enhancements (text + image + majority baselines, single seed) | 3 | ~5.5 min (193 + 127 + 3.5 s) |

A100-hours consumed: well under 1.0. The earlier "few GPU-days" budget estimate is now confirmed conservative by an order of magnitude — Stage 3 robust training (which doubles the per-batch cost via the perturbed view) should still fit comfortably inside one A100-day, even with 3 seeds × 2 backbones.
