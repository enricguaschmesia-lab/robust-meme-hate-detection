---
name: epfl-runai-cluster
description: Use this skill when an agent needs to operate the EPFL Run:AI GPU cluster for robust-meme-hate-detection from this repo's canonical WSL/Linux checkout using the automatic Bash workflow in ./cluster/cluster.sh. Covers environment validation, code sync, smoke tests, interactive debugging, image rebuild and push decisions, real Stage-1 training via run-train, and artifact retrieval.
---

# EPFL Run:AI Cluster Automation — robust-meme-hate-detection

## Required Human Input

If the SSH agent is missing or locked, tell the human to run:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/epfl-ssh-key
```

Also make sure the EPFL VPN is connected; `ssh: Could not resolve hostname jumphost.rcp.epfl.ch: Name or service not known` means the VPN or DNS path is not available.
If Run:AI auth is expired, tell the human to run `ssh epfl-jumphost`, then `runai login`, then `runai config project course-ee-559-guasch`.

Ask for these confirmations before starting any cluster action:

- The intended goal: `doctor`, `smoke-test`, `interactive-debug`, `full-train`, `pull-results`, `rebuild-image`, or `adapt-workflow`.
- Confirmation that the agent is running from the canonical WSL/Linux repo copy at `robust-meme-hate-detection/`, not a Windows mirror.
- Confirmation that the shell already has an unlocked `ssh-agent` with the EPFL key for the `epfl-jumphost` host alias.
- Confirmation of what changed locally: code, `configs/*.yaml`, `Dockerfile`, `requirements.txt`, or nothing. (The dataset lives read-only on shared scratch and is not synced from here.)
- If a real or smoke job will be launched, an explicit job name only if the default timestamped name should be overridden.
- For `submit-train`/`run-train`, confirm the desired `--config` (defaults to `configs/stage1.yaml`) and `--seed` (defaults to `0`).

Do not start cluster automation until the SSH agent condition is true. If the check fails, remind the human to run the two commands above. The workflow uses `ssh -o BatchMode=yes` and is expected to fail fast when the key is unavailable.

## Canonical Sources

Read these files instead of paraphrasing the workflow from memory:

- `README.md`
- `docs/CLUSTER_DATA.md`
- `docs/DATA.md`
- `cluster/cluster.sh`
- `cluster/config.env`
- `cluster/config.env.example`
- `cluster/state.local.env`
- `scripts/cluster_entrypoint.sh`
- `scripts/runai_submit_template.sh`
- `configs/smoke.yaml`
- `configs/stage1.yaml`

## Repo Facts That Matter

- The only canonical command entrypoint is `./cluster/cluster.sh`. (Note: `cluster/`, not `scripts/`.)
- Versioned cluster defaults live in `cluster/config.env`; a template for new collaborators is `cluster/config.env.example`.
- The active built or pushed image is tracked in `cluster/state.local.env` and currently resolves to `registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:<tag>`.
- Run:AI project is `course-ee-559-guasch`; group scratch PVC is `course-ee-559-scratch-g49` mounted at `/scratch`; per-user `home` PVC mounts at `/home/guasch`.
- The dataset (Hateful Memes / `hate_meta`) is **read-only on the shared cluster scratch at `/scratch/datasets/hate_meta`**, staged by course staff. There is no `sync-data` step — never try to upload or overwrite it from local.
- Local code in `src/robust_meme_hate_detection/` is synced to `/home/guasch/robust-meme-hate-detection` on the jumphost via `sync-code` and reaches jobs through the `home` PVC.
- The in-job launcher is `scripts/cluster_entrypoint.sh`, which sets `PYTHONPATH=$REPO_DIR/src` and `exec`s `python3 -m <module>`.
- The synced **smoke entrypoint** is `robust_meme_hate_detection.tests.smoke_cluster` driven by `configs/smoke.yaml`.
- The synced **training entrypoint** is `robust_meme_hate_detection.train.stage1` driven by `configs/stage1.yaml`.
- Training outputs land in `$JOB_RESULTS_ROOT/<job-name>/` on shared scratch and at minimum contain `metrics.json` plus a checkpoint directory; `summarize-results` parses `metrics.json` and applies the Stage-1 acceptance bar of dev AUROC ≥ 0.72.
- Local artifact pulls go to `cluster-results/<job>/` and checkpoint-only pulls go to `cluster-checkpoints/<job>/`.

## Command Rules

- Use `./cluster/cluster.sh` unless you are debugging the wrapper itself.
- Treat the WSL/Linux working tree as canonical.
- Keep SSH non-interactive through `ssh-agent`; do not switch the workflow to password or prompt-based login.
- Prefer `pull-artifacts` (alias of `pull-results`) or `pull-checkpoints` over ad hoc `rsync`.
- Never stop at `submit-train` for a real run. Use `run-train`, or do the exact full sequence yourself: submit, wait, pull artifacts, pull checkpoints, summarize, then report.
- Use `build-image` and `push-image` only when dependencies or runtime behavior changed.
- Do **not** invent a `sync-data` step — the dataset is managed by course staff. If `/scratch/datasets/hate_meta` looks wrong, escalate to the human, do not try to reupload.

## Decision Rules

- If Python or YAML config source changed, run `sync-code`.
- If `Dockerfile` changed, `requirements.txt` changed, imports fail in-cluster, or the UID user mapping is wrong, rebuild and push the image.
- If runtime behavior is uncertain, run `submit-smoke` before any real training.
- If manual inspection is needed for mounts, imports, GPUs, or filesystem state, use `submit-interactive` and `interactive-bash`.
- If the dataset path inside a job resolves but is empty or unexpected, stop and ask — don't try to repopulate `/scratch/datasets/hate_meta`.

## Standard Command Order

Use this default order unless there is a concrete reason to skip a step:

```bash
./cluster/cluster.sh doctor
./cluster/cluster.sh sync-code
./cluster/cluster.sh submit-smoke <job>
./cluster/cluster.sh job-status <job>
./cluster/cluster.sh job-logs <job> --pod <pod> --tail 200
./cluster/cluster.sh pull-artifacts <job>
./cluster/cluster.sh run-train <job>
```

If the job name is omitted, the script generates a timestamped name based on the git SHA (`smoke-YYYYMMDD-HHMMSS-<sha>` or `train-YYYYMMDD-HHMMSS-<sha>`).

## Safe Operating Flows

### Environment Validation

```bash
./cluster/cluster.sh doctor
./cluster/cluster.sh remote-check
```

Stop if either command fails. Do not script around a failed environment.

### Code-Only Iteration

```bash
./cluster/cluster.sh sync-code
./cluster/cluster.sh submit-smoke <job>
./cluster/cluster.sh pull-artifacts <job>
```

### Image Rebuild

```bash
./cluster/cluster.sh build-image [tag]
./cluster/cluster.sh push-image [tag]
```

This updates `cluster/state.local.env`, which later job submissions use as the active image.

### Interactive Debugging

```bash
./cluster/cluster.sh submit-interactive <job>
./cluster/cluster.sh interactive-bash <job>
```

Use this to verify imports, mounts (`/scratch`, `/home/guasch`), GPU visibility, and that `/scratch/datasets/hate_meta` is readable inside the actual Run:AI runtime.

### Full Training

```bash
./cluster/cluster.sh run-train <job> [poll-seconds]
```

`run-train` is the preferred real-training command because it submits the job, waits for completion, streams new logs, pulls artifacts, pulls checkpoints, and summarizes `metrics.json` (Stage-1 acceptance bar: dev AUROC ≥ 0.72).

## Current Project Defaults

For this repo, the configured workflow currently uses:

- `submit-smoke`: `--config configs/smoke.yaml` (ViT-B-32 / laion2b_s34b_b79k, frozen encoders, a few mini-batches against `/scratch/datasets/hate_meta`)
- `submit-train`: `--config configs/stage1.yaml --seed 0` (frozen-encoder Stage-1, 10 epochs, batch size 128, bf16 AMP, early stop on macro F1)
- 1 GPU per job (`RUNAI_GPU=1`), large shared memory, `home` + `course-ee-559-scratch-g49` PVCs mounted
- Results root: `/scratch/robust-meme-hate-detection/experiments/<job>/`

Override per call with `--config <path>` / `--seed <n>` or via `SMOKE_CONFIG` / `TRAIN_CONFIG` env vars. Read `cluster/config.env` before assuming any specific path, image tag, or Run:AI project value.

## Failure Handling

- If `doctor` reports SSH batch mode failure, stop and ask the human to unlock the SSH key in the shell that launched the agent.
- If the registry push fails with `401`, stop and ask the human to authenticate Docker to `registry.rcp.epfl.ch`.
- If the job fails during image pull, verify that the intended tag was actually pushed and that `cluster/state.local.env` points at it.
- If the job fails on imports, patch `Dockerfile` or `requirements.txt`, then rebuild and push.
- If the job fails on UID or `getpwuid()` issues, add the runtime user to the image instead of changing Run:AI commands blindly.
- If the job fails because `/scratch/datasets/hate_meta` is missing or empty, do **not** attempt to upload data — escalate; the dataset is staged by course staff.

## Reporting Expectations

After any cluster action, report:

- what command was run
- whether it succeeded
- any job name created or reused
- where artifacts were pulled locally (`cluster-results/<job>/`, `cluster-checkpoints/<job>/`)
- the next recommended step

For completed training runs, include the final `summarize-results` conclusion (the dev AUROC line and the pass/fail verdict against the 0.72 Stage-1 bar), not just raw logs.
