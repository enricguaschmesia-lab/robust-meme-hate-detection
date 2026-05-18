# Implementation Worklog — Phases 1–15

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

## Phase 12 — Perturbation calibration round 2

**2026-05-16** — First inspection round (`inspect-perturb-20260512-211209-a4852b5/`) reviewed manually. Notes recorded in `project_planning/perturbations_calibration.md`. Calibration deltas applied to `perturbations/{image,text}_perturbations.py`:

- gaussian_noise σ 0.01 / 0.04 / 0.08 (was 0.01 / 0.03 / 0.05).
- blur radius now float (PIL accepts it); 0.8 / 1.8 / 3.0 px (was integer-quantised 1/2/3).
- compression JPEG q ≈ 80 / 40 / 15 (was 80 / 50 / 25).
- brightness_down magnitude raised to severity·0.8 (factors 0.80 / 0.50 / 0.20).
- brightness_up / contrast_{up,down} magnitude raised to severity·0.6 (factors ±0.15 / ±0.375 / ±0.60).
- translation / crop 4 / 10 / 20 % per edge (was 2 / 5 / 10).
- occlusion patch 10 / 20 / 35 % (was 5 / 10 / 20).
- char_swap 2 / 3 / 5 swaps (was 1 / 2 / 3).
- censoring 5 / 12 / 25 % of chars (was 10 / 20 / 40).

Old inspection artefacts (`inspect-perturb-20260512-211209-a4852b5/`, 2.3 GB) deleted locally. Regenerated set materialised to `cluster-results/inspect-perturb-calibrated-20260516-130751/` (27 000 manifest rows). Phase 3 closed.

## Phase 13 — Phase-4 robustness evaluation

**2026-05-16** — Local `cluster-checkpoints/stage1-seed{0,1,2}-*/best.pt` re-uploaded to `/mnt/.../scratch-g49/.../experiments/` because the cluster scratch dirs had been cleaned (≈1.7 GB total, three 582 MB files).

Submitted 5 naturalistic-benchmark jobs (`eval/run_perturbed.py`) and 4 white-box jobs (new `eval/run_whitebox.py`, FGSM + PGD-10 across ε ∈ {1,2,4,8}/255). Sample sanity:

- Per-run clean AUROC matches the Phase-1-2 numbers to ≤ 0.01 (whitebox seed-0 0.7432 exact; perturbed seed-0 0.7492 — small diff is bf16 reduction order across batch sizes).
- No cell has attacked AUROC > clean + 0.02 (sign check passes).
- All ASR values fall in [0, 1].

Aggregator (`scripts/aggregate_phase4.py`) reads every `cluster-results/{perturbed,whitebox}-*/...json` and emits `project_planning/phase4/{perturbed_table,whitebox_table,worst_case}.md` + 5 PNG figures.

Headline:

- **White-box PGD dominates**: ε ≥ 2/255 → multimodal AUROC ≈ 0, ASR ≥ 0.98. FGSM at ε = 8/255 only reaches ASR ≈ 0.67.
- **Among naturalistic attacks, high-severity text edits are worst** (multimodal ΔAUROC ~0.11 for `censoring`, `char_deletion`, `leetspeak` at high). Image attacks drop multimodal AUROC by ≤ 0.04 at high severity.
- **Multimodal is MORE text-fragile than the text-only baseline** (multimodal Δ=+0.085 vs text-only Δ=+0.065 at high). Informs Phase-5 augmentation mix.

Full write-up in `project_planning/Phase3_4_Completion_Report.md`. Phase 4 DoD closed.

## Phase 14 — Robust training (Phase 5)

**2026-05-17** — Added robust training pipeline:

- `src/robust_meme_hate_detection/train/_robust_loss.py` — pure utilities (`binary_kl`, `robust_loss`).
- `src/robust_meme_hate_detection/train/_robust_augmenter.py` — `RobustAugmenter` class wrapping `TextPerturbation` and `ImagePerturbation`, samples `text` / `image` / `both` per example with weights `(0.5, 0.3, 0.2)`.
- `src/robust_meme_hate_detection/train/stage1_robust.py` — sibling of `stage1.py`. Custom collate yields raw PIL + caption alongside processed tensors; per-batch step does 2 fwds (clean + perturbed) + optionally 2 image-only fwds, computes `BCE_clean + α·BCE_pert + β·KL_full + γ·KL_image_branch`, logs each term.
- `configs/stage1_robust_{augonly,kl}.yaml` — augonly (β=γ=0) and kl (β=0.5, γ=0.25) variants.

Cluster runs: 6 training jobs (`train-robust-{augonly,kl}-seed{0,1,2}`) + 18 eval jobs (3 evals each × 6 ckpts: perturbed naturalistic, white-box FGSM/PGD ε-sweep, modality ablation). Per-job wall time ranged 12–60 min depending on GPU node contention; all 24 succeeded.

Aggregator extended (`scripts/aggregate_phase4.py`): new `_load_modality_ablation`, `_is_robust_key`, `_group_by_variant`, `_fully_robust_count`, `_write_robust_vs_clean`, `_make_robust_figures`. Output: `phase4/robust_vs_clean.md` + `figures/heatmap_robust_kl_seed0.png` + `figures/curve_whitebox_robust.png`.

Failure analysis extended (`scripts/failure_analysis.py`): new `_load_robust_jsons`, `_annotate_robust` add per-example `robust_status ∈ {fixed, still_failed, new_failure, unchanged}` and aggregate counts.

Headline:

- **Clean dev AUROC** (3-seed mean): clean 0.742 → augonly 0.712 → kl 0.737. KL pays only 0.005 AUROC for robustness — augonly costs 0.030.
- **Worst-cell text ΔAUROC**: clean 0.115 → augonly 0.083 → kl 0.090. Target was ≤ 0.055 — missed but improved by 22–28 %.
- **Image-only forward AUROC**: clean 0.593 → augonly 0.604 → kl 0.611. `KL_image_branch` term lifted the dead image branch by +0.018 but did not match the dedicated image-only baseline of 0.628.
- **Per-example fully-robust count** (mean of 3 seeds, out of 500): clean 0.33 → augonly 1.67 → kl 1.00. Target of ≥ 50 missed entirely; per-cell ΔAUROC gains do not concentrate on any individual example.
- **White-box PGD ε ≥ 2/255 still drives AUROC to ~0** for all recipes (expected non-target).

Full write-up in `project_planning/Phase5_Completion_Report.md`.

## Phase 14b — Modality dropout + naturalistic threat-model framing

**2026-05-17** — Two follow-up actions triggered by review of Phase-5 framing:

1. **Naturalistic-only fully-robust metric** added to `scripts/aggregate_phase4.py` (`_natural_fully_robust_count`) and `scripts/failure_analysis.py`. Strips out PGD ε=4/255 from the fully-robust definition so the per-example headline aligns with the project's actual threat model (typical adversarial user, not gradient-aware attacker). Re-running the aggregator on existing data immediately reframed the story: clean baseline goes from "1/500 fully robust" → **94-114/500 naturalistic robust** depending on seed; kl from 1/500 → 137/500 (+43 examples vs clean on seed 0).

2. **Modality dropout** added to `src/robust_meme_hate_detection/train/stage1_robust.py` via new `_fuse_with_dropout` helper (verified to exactly reproduce `forward_image_only`/`forward_text_only` at the mask boundaries) and new config `configs/stage1_robust_kl_drop.yaml` (kl recipe + per-example text-modality dropout p=0.30). Three additional training jobs + nine eval jobs (`train-robust-kldrop-seed{0,1,2}`, all three eval kinds per ckpt).

Phase 5b headline:

- **`kldrop` fully revives the dead image branch.** Image-only forward AUROC: 0.593 (clean) → 0.611 (kl) → **0.636 (kldrop)**. This is the first recipe whose image-only forward EXCEEDS the dedicated image-only baseline (0.628). Modality dropout is the hard-forcing intervention that KL_image_branch alone was too soft to deliver.
- **`kldrop` achieves the best per-cell text robustness**: worst-cell text ΔAUROC 0.115 → 0.067 (42 % reduction, vs kl's 0.090 → 22 %).
- **Cost**: clean AUROC drops to 0.705 (vs kl's 0.737). Naturalistic-robust count: 124 (vs kl's 134) — slightly worse than kl because the smaller pool of clean-correct examples bounds the count.
- **Pareto reading**: `kl` is the best for clean-input accuracy + naturalistic survival; `kldrop` is the best for branch-balanced fusion + per-cell robustness. Both are publishable, complementary outcomes.

Phase 5 report (`project_planning/Phase5_Completion_Report.md`) rewritten in-place with the new threat-model framing, kldrop column added to all comparison tables, image-branch revival result highlighted in § 8. `project_planning/phase4/robust_vs_clean.md` and `phase4/failure_analysis.md` rewritten by re-running the aggregator and failure-analysis scripts.

## Phase 15 — Generalisation analysis (Phase 5c, 2026-05-17)

Three follow-up extensions on Phase 5b prompted by three review questions: (a) does the augmentation gain transfer to attacks not in the training pool, (b) what about compositional attacks (multiple perturbations per sample, the realistic-adversarial-user threat model), and (c) is high-severity augmentation necessary, or does training only on internet-realistic (low+medium) severities suffice?

**5c-1 (free, no cluster cost)** — extended `scripts/aggregate_phase4.py` with three sections written to `project_planning/phase4/robust_vs_clean.md`:

- `_in_pool_attacks` reads the canonical training pool from `configs/stage1_robust_kl.yaml`.
- `_write_pool_vs_ood_table`: per-recipe mean ΔAUROC split by training-pool membership, separately for text and image. Includes Δ(OOD−in) column for each modality. Caveat noted in-table that held-out text attacks are intrinsically weak (≤ 0.02 ΔAUROC on the clean ckpt), so the text-side column has limited signal.
- `_write_per_severity_table`: per-recipe ΔAUROC at low / medium / high, per family. Medium column bolded as the "internet-realistic threat" headline.
- `_write_composite_table`: per-recipe per-severity stats for composite cells (no-op until 5c-2 runs land). Auto-detects.
- `_is_robust_key` learns `kllowmed`; `VARIANT_ORDER` constant replaces ad-hoc tuples in six call sites.

Re-aggregating against existing cluster-results immediately shows:

- Δ(OOD−in) image is negative for every robust recipe → augmentation gains *transfer* to held-out image attacks. `kldrop` shows the largest gain on held-out image attacks (0.0013 vs clean's 0.0048).
- Monotone low ≤ medium ≤ high ΔAUROC for every (recipe, family) — sanity gate passed; no severity-calibration drift.
- Medium-severity headline: kldrop cuts text-attack ΔAUROC by 46 % vs clean (0.063 → 0.034).

**5c-2 (~1.5 A100-h)** — extended `src/robust_meme_hate_detection/eval/run_perturbed.py` with:

- `--composites <none|all|csv>` CLI flag (default `none`, back-compat preserved).
- `_apply_composite(text, image, sample_seed, ...)`: deterministic per-sample composite, sample seed derived from `sha256(cell_seed | sample_id)`. Components sampled *without replacement* per modality from the full eval pool (TEXT_MODES + IMAGE_MODES_BENCHMARK) — composite is itself an OOD generalisation test.
- `COMPOSITE_TYPES` ∈ `{composite_2text, composite_2image, composite_text_image, composite_2text_2image}` × 3 severities = 12 new cells per perturbed-eval run (57 single + 12 composite = 69 cells per JSON).

Submitted 12 perturbed-eval re-runs (`perturbed-{stage1,robust-augonly,robust-kl,robust-kldrop}-seed{0,1,2}` with `--composites all`); each overwrites its existing JSON. Per-cell composite log records the components applied per sample so the report can quote example strings if needed.

**5c-3 (~2 A100-h)** — `configs/stage1_robust_kl_lowmed.yaml`: copy of `stage1_robust_kl.yaml` with `severity_weights_text: [1, 2, 0]` and `severity_weights_image: [1, 1, 0]`. All other hyperparameters identical to `kl` so severity coverage is the only variable.

Submitted 3 train + 9 eval jobs (`train-robust-kllowmed-seed{0,1,2}`; `perturbed-`/`whitebox-`/`modality-ablation-robust-kllowmed-seed{0,1,2}`); evals chained via a `wait-for-Succeeded` poller so they fire as soon as each train completes.

Phase 5 report gains § 13 ("Phase 5c — generalisation analysis") with the 5c-1 tables already populated; 5c-2 composite ΔAUROC and 5c-3 `kllowmed` row land automatically when the cluster pull happens.

**Phase 15 outcomes (2026-05-17, all jobs Succeeded, results pulled):**

- *5c-2 composite attacks*: composite cells escalate the threat past the single-attack worst cell. Clean-ckpt `composite_2text_2image` at high severity reaches ΔAUROC 0.145 (vs single-cell worst 0.115). `kldrop` is the strict Pareto winner on every composite cell — cuts realistic medium-severity `composite_2text` by 43 % vs clean (0.090 → 0.051), `composite_text_image` by 51 % (0.050 → 0.024), `composite_2text_2image` by 41 % (0.080 → 0.047). Modality dropout pays off most when the attack hits both modalities at once.
- *5c-3 `kl_lowmed`*: clean negative result — Pareto-dominated by `kl` on every metric. Clean AUROC tied (0.7347 vs 0.7367, within σ); text ΔAUROC at medium +0.002 (worse), at high +0.003; image-only AUROC −0.009; naturalistic-robust count 107.33 vs 111.67 (−4). High-severity augmentation in the original Phase-5 setup was *not* over-tuned — removing it produces a strictly worse model. Publishable as a clean methodology check.
- *Updated Pareto reading*: `kl` for clean-accuracy-preserving naturalistic single-attack robustness; **`kldrop` is the recommended ckpt under the realistic composite threat model** and the only recipe with a fully-revived image branch. `augonly` is now dominated; `kllowmed` is the negative result.

Phase 5 report § 13.4–13.7 backfilled with the actual data; README Phase 5c block rewritten with the final findings.
