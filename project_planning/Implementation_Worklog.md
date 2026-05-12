# Implementation Worklog — Phases 1–11

This log records every step taken from the green-light decision (2026-05-02) through to the Definition of Done in `model_architecture.md` §12. It is appended to as work progresses; existing entries are not edited except to append outcomes.

**Operational policy for this run** (per user instructions):

- `uv venv` for the local environment is approved.
- Docker is logged in to `registry.rcp.epfl.ch`.
- If Stage-1 dev AUROC ≥ 0.72, proceed without checking back; if below, stop and report.
- wandb stays offline.
- Spawn review agents (Agent tool, `general-purpose` subagent type — `/spawn-agent` was not configured) at code-checkpoint boundaries; apply fixes inline.

---

## Phase 1 — Stale config cleanup

**Goal:** point all dataset references at the actual cluster path `/scratch/datasets/hate_meta` and the 3-split + unlabeled-test layout.

**2026-05-02** — Updated three files:

- `configs/data/hateful_memes.yaml`: `cluster_root: /scratch/datasets/hate_meta`, batch 128, num_workers 8, added `train_split_file` reference. Removed `recommended_hf_repo` / `metadata_only_hf_repo` (not used now that the dataset is staged).
- `docs/CLUSTER_DATA.md`: replaced the registration / rsync / "download to laptop" workflow with the actual staged-path workflow, including the verified diagnostic job reference and the held-out split policy.
- `docs/DATA.md`: rewritten for the cluster-only layout and the 8000/500 stratified partition; explicit "don't tune on train_held_out" rule.

Phase 1 done.

## Phase 2 — Project-side cluster automation

**2026-05-02** — Created:

- `cluster/config.env`: SSH_HOST=epfl-jumphost, RUNAI_PROJECT=course-ee-559-guasch, REMOTE_PROJECT_DIR=/home/guasch/robust-meme-hate-detection, REMOTE_SCRATCH_PROJECT_DIR=…/scratch-g49/robust-meme-hate-detection, JOB_DATA_ROOT=/scratch/datasets/hate_meta, IMAGE_REPO=registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection, IMAGE_TAG=v0.1.
- `cluster/sync-code.exclude`: keeps `data/`, `experiments/`, `cluster-results/`, `wandb/`, virtual envs, etc. out of the rsync.
- `cluster/cluster.sh`: adapted from the tutorial. Drops `sync-data` (dataset is staged read-only); adds `submit-cmd <job> -- <cmd>` as an arbitrary-command escape hatch; replaces fruit-dataset / practice-3 specifics with `python -m robust_meme_hate_detection.train.stage1 --config <yaml>`-style invocations; `summarize-results` reads `metrics.json` instead of `metrics.csv`.

`./cluster/cluster.sh doctor` passes:

```
ssh ok    rsync ok    docker ok    git ok    codex ok
docker daemon ok
config ok    sync excludes ok    ssh batch mode ok
remote runai ok (course-ee-559-guasch)
remote scratch ok (/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch-g49)
staged dataset ok (.../datasets/hate_meta)
active image  registry.rcp.epfl.ch/ee559/environment-with-packages:latest
```

Phase 2 done.

## Phase 3 — Implement Python modules

**2026-05-02** — Wrote and committed (under `src/robust_meme_hate_detection/`):

- `models/clip_fusion.py`: `CLIPHateMemeClassifier`, `FusionHead`, `Normalize`. Uses OpenCLIP's high-level `encode_image` / `encode_text` so we are not coupled to internal attribute names. `Normalize` lives inside the model so PGD has gradient access to `[0,1]` pixels. Includes `unfreeze_last_blocks(n=2)` for Stage 2 and a `_MinimalClipStub` so unit tests do not need network access.
- `attacks/pgd.py`: `fgsm_image` and `pgd_image`. L∞ on `[0,1]`, BCE-with-logits loss by default.
- `data/splits.py`: `load_train_split` for the deterministic 8000/500 partition; `data/transforms.py` extended with `ClipImage01Transform` (resize → CenterCrop → ToTensor in [0,1], no Normalize) and `ClipTokenize` wrapping the OpenCLIP tokenizer.
- `eval/metrics.py`: `classification_metrics`, `attack_success_rate`, `robustness_gap`. Sklearn-backed.
- `utils/seeding.py`, `utils/logging.py` (`JsonRunLogger` writes `metrics.json` + `steps.jsonl`), `utils/threshold.py` (sweep on macro F1).
- `train/stage1.py`: full training loop with frozen encoders, AdamW + cosine warmup, BCE pos_weight auto-computed, AMP bf16, early stopping on dev macro-F1, best-checkpoint save, JSON metrics dump.
- `tests/smoke_synthetic.py`: CPU-only synthetic test of the full forward + backward + PGD loop with random weights via `_MinimalClipStub`.
- `tests/smoke_cluster.py`: cluster-side smoke job entry point that builds the dataset against `/scratch/datasets/hate_meta` and does a few forward+backward batches.
- `perturb/__init__.py` (stub for now; full text/image perturbation suites are Phase 3+ of the roadmap).
- `scripts/make_train_split.py`: stratified deterministic 8000/500 ID partition writer.
- `tests/test_clip_fusion.py`, `tests/test_metrics.py`: pytest-style unit tests with no GPU/network requirements.
- `requirements.txt`: added `open_clip_torch>=2.24`, `transformers>=4.41`, `matplotlib`.
- `configs/stage1.yaml`, `configs/smoke.yaml`.
- `Dockerfile`: layers project requirements onto the course base image and pre-bakes ViT-B/32 + ViT-L/14 weights to avoid first-run download races on compute pods.

Phase 3 + Phase 5 (configs) + Phase 6 (Dockerfile) are written. Reviewing in Phase 3.5, then Track A.

## Phase 4 — Track A local validation

**2026-05-02** —

- Installed `uv 0.11.8` to `~/.local/bin`.
- `uv venv --python 3.12` → CPython 3.12.3 in `.venv/`.
- `uv pip install --python .venv/bin/python -e . -r requirements.txt` succeeded. Notable installed versions: torch 2.11.0, open_clip_torch (latest), scikit-learn 1.8.0, transformers 5.7.0, timm 1.0.26.
- `pytest tests/` → **7 passed, 4 skipped** (the 4 skips are `tests/test_hateful_memes_data.py` cases that require the dataset on disk; expected since the dataset lives on cluster scratch).
- `python -m robust_meme_hate_detection.tests.smoke_synthetic` → **PASSED**:
  ```
  smoke_synthetic OK: forward, backward, and PGD gradient flow all verified.
  max delta=0.01569, mean delta=0.00871
  ```
  (random head, ε=4/255, 2 PGD steps; the non-zero delta confirms gradient flow back to the [0,1] image tensor — this is the contract that keeps the model attack-ready.)
- Generated `data/processed/splits/train_split.json` (deterministic, seed 0) by streaming the cluster's `train.jsonl` over SSH and running `scripts/make_train_split.py`. Result: **train=8000 (2871 pos, 35.9%), held_out=500 (179 pos, 35.8%)** — class balance preserved.

Track A done. Proceeding to Phase 3.5 (review agent) before pushing to the cluster.

## Phase 3.5 — Code review by general-purpose agent

**2026-05-02** — Spawned a `general-purpose` Agent over Phase 3 deliverables. Findings and fixes:

- **Real bug:** `scripts/make_train_split.py` wrote IDs as bare `str(r["id"])` while `HatefulMemesRecord.id` is normalised via `.zfill(5)`. Any ID with fewer than 5 digits would be silently dropped by the `train/stage1.py` split filter, corrupting the partition. Fix: zero-pad to 5 chars at split-write time and in the post-write summary count. After fix, all 8500 IDs are uniformly width=5.
- **Defensive fix:** `train/stage1.py` initialised `epoch = -1` before the training loop so that `epochs_run = epoch + 1` returns 0 cleanly when `epochs=0` (rather than `NameError`).
- Concerns flagged but intentionally not fixed: cosmetic re-imports, `_RecordsDataset` defined inside `main()` of `smoke_cluster.py` (works under Linux's default `fork`), `pgd.sign(0)=0` no-op (acceptable). All within the surgical-change scope I gave the agent.

After re-running the regeneration with the fixed script:
```
Wrote data/processed/splits/train_split.json: train=8000 (2871 pos, 0.359), held_out=500 (179 pos, 0.358)
total IDs: 8500 | sample: ['01235', '01236', '01243'] | widths: [5]
```

`pytest`: 7 passed, 4 skipped. `smoke_synthetic`: PASSED.

## Phase 6 — Custom Docker image

**2026-05-02/03** — Iterated through three image versions while debugging cluster integration:

- **v0.1 (initial)**: PEP-668 blocked `pip install` in the base image. Fix: added `--break-system-packages` to the pip command. Build also revealed that the base image only has `python3` (not `python`) on PATH; updated the weight pre-bake step to use `python3`.
- **v0.2 (UID fix)**: First cluster smoke run failed with `KeyError: 'getpwuid(): uid not found: 316498'` from `torch._dynamo.cache_dir` calling `getpass.getuser()`. Fix: added a `useradd`-equivalent for uid 316498 plus `ENV USER=guasch HOME=/home/guasch LOGNAME=guasch` so `getpass.getuser()` short-circuits via env vars.
- Both ViT-B/32 and ViT-L/14 weights baked into the image successfully (`baked ViT-B-32/laion2b_s34b_b79k`, `baked ViT-L-14/laion2b_s32b_b82k`).
- Push to `registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2` succeeded after one transient network retry on each push attempt (TCP closed by the corporate proxy mid-upload — expected; Docker push is resumable).

`cluster/state.local.env` records `ACTIVE_IMAGE=registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2`.

## Phase 7 — Cluster smoke job (B.4)

**2026-05-03** — Three iterations to reach a clean run:

1. First attempt failed because `python -m robust_meme_hate_detection.tests.smoke_cluster` couldn't find the package — the `src/`-layout requires PYTHONPATH. Replaced the inline `bash -lc "..."` command with a small launcher script `scripts/cluster_entrypoint.sh` that sets `PYTHONPATH=$REPO_DIR/src` and `cd`s before exec'ing python3.
2. Second attempt failed with `getpwuid()` (the v0.1 image issue) — fixed by Dockerfile v0.2.
3. Third attempt failed with `RuntimeError: unable to allocate shared memory` — Run:AI pods have a tiny default `/dev/shm` (multi-worker DataLoader needs more). Fixed by adding `--large-shm` to all `runai submit` invocations and reducing the smoke job's `num_workers` to 0 (smoke only does 4 batches).
4. **Fourth attempt: Succeeded.** Logs:
   ```
   device=cuda trainable_params=1053697 total_params=152331010 n_val=500
   batch=0 loss=0.5870
   batch=1 loss=0.7894
   batch=2 loss=2.1362
   batch=3 loss=1.8737
   smoke_cluster OK {'device':'cuda','elapsed_seconds':2.79,'trainable_params':1053697,'total_params':152331010,'n_val_seen':128,'val_accuracy':0.7031,'val_macro_f1':0.4128,'val_auroc':0.3706, ...}
   ```
   - 152.3M total params, 1.05M trainable — matches the model_architecture.md targets (1.3M was an upper bound for hidden=512; the actual fusion head is 1.05M).
   - 4 forward+backward batches in 2.8s on the A100. The whole pipeline (image, dataset mount, model build, optimizer step, eval) is now end-to-end working on the cluster.
   - `metrics.json` written to `/scratch/.../experiments/<job>/` and pulled to `cluster-results/<job>/`. AUROC of 0.37 on a 128-example random-head subset is fine — random classifier on a small sample.

DoD item 2 (Track B.4) ✅ closed.

## Phase 8 — Stage-1 training (3 seeds)

**2026-05-03** — Three seeds, all on `registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2`, configs/stage1.yaml, batch 128, bf16 AMP on A100 80 GB.

| Seed | AUROC | macro F1 | best epoch | wall clock |
|---:|---:|---:|---:|---:|
| 0 | 0.7432 | 0.6373 | 5 | 145.9 s |
| 1 | 0.7254 | 0.6509 | 4 | 136.0 s |
| 2 | 0.7329 | 0.6388 | 4 | 137.7 s |

All seeds early-stop at epoch 7-8 with patience 3. AUROC σ=0.0073 across seeds.
**All seeds clear the 0.72 acceptance bar.** Per the user's pre-confirmation we proceed to evaluation.

## Phase 9 + 10 — Threshold sweep + PGD attack-readiness (combined eval job, seed 0)

**2026-05-03** — Single Run:AI job `eval-seed0-20260503-165736` runs clean eval, threshold sweep, and PGD on the seed-0 checkpoint.

- Best τ on macro F1: **0.41** (lifts accuracy 0.65 → 0.68, recall 0.476 → 0.572).
- PGD (ε=4/255, α=1/255, 5 steps): clean-correct accuracy 0.68 → **0.008**, AUROC 0.7432 → **0.0004**, **ASR=0.988**. Confirms end-to-end gradient flow through Normalize and the encoders.

`cluster-results/eval-seed0-20260503-165736/eval.json` has the full threshold grid and per-attack metrics.

DoD items 3 + 4 ✅ closed.

## Phase 11 — Final report

Wrote `project_planning/Phase1-2_Completion_Report.md`. Project ready for Phase 3 of the roadmap (perturbation suite) and Phase 5 (Stage-3 robust training).
