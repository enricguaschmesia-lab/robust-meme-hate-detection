#!/usr/bin/env bash
# Manual Run:AI submit template — for collaborators who run cluster jobs
# without the cluster.sh automation.
#
# Workflow:
#   1. SSH to the jumphost.
#   2. Clone or `git pull` this repo into your $HOME (one-off + on update).
#   3. Edit the variables under "EDIT ME" below for your account
#      (or override them on the command line, e.g. `JOB=foo bash scripts/runai_submit_template.sh`).
#   4. Pick exactly one CMD preset under "COMMAND PRESETS" (uncomment its block).
#   5. Run:    bash scripts/runai_submit_template.sh
#   6. Watch:  runai logs $JOB        # add -f to follow
#   7. Status: runai describe job $JOB
#   8. Results land in $JOB_RESULTS_ROOT/$JOB on the shared /scratch.
#
# Prerequisites on the jumphost:
#   - You are logged in to runai under your project:
#         runai config project course-ee-559-<your-username>
#   - The image is pullable from the cluster. Either:
#       (a) use a teammate's pushed image (if your registry creds permit), or
#       (b) build/push your own once with `cluster.sh build-image` + `push-image`.
#   - The shared scratch PVC (course-ee-559-scratch-g49) is attached to your project.

set -euo pipefail

# ---------------- EDIT ME --------------------------------------------------

# A unique job name. Run:AI requires a-z0-9-, max 63 chars.
JOB="${JOB:-stage1-seed0-$(date +%Y%m%d-%H%M%S)}"

# Container image with open_clip + pre-baked CLIP weights.
IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2}"

# Your numeric uid on the cluster (find with `id -u` on the jumphost).
USER_UID="${USER_UID:-$(id -u 2>/dev/null || echo 0)}"

# Your home PVC mount path inside the job. Must match your jumphost $HOME.
HOME_PVC="${HOME_PVC:-home}"
HOME_MOUNT="${HOME_MOUNT:-$HOME}"

# Where this repo is checked out on the jumphost. The home PVC mounts the same
# path inside the job, so this works for both jumphost and pod.
REPO_DIR="${REPO_DIR:-$HOME_MOUNT/robust-meme-hate-detection}"

# Shared group scratch (do not change unless your group differs).
SCRATCH_PVC="${SCRATCH_PVC:-course-ee-559-scratch-g49}"
SCRATCH_MOUNT="${SCRATCH_MOUNT:-/scratch}"
JOB_RESULTS_ROOT="${JOB_RESULTS_ROOT:-/scratch/robust-meme-hate-detection/experiments}"

# ---------------- COMMAND PRESETS ------------------------------------------
# Uncomment exactly one CMD block.

# (a) Cluster smoke — 4 batches forward+backward, ~3 s on A100. Use to verify
# the image, mounts, and dataset are visible from a real GPU pod.
# CMD=(
#     "$REPO_DIR/scripts/cluster_entrypoint.sh"
#     robust_meme_hate_detection.tests.smoke_cluster
#     --config configs/smoke.yaml
#     --out "$JOB_RESULTS_ROOT/$JOB"
# )

# (b) Stage-1 training — frozen encoders, 10 epochs, ~140 s on A100.
CMD=(
    "$REPO_DIR/scripts/cluster_entrypoint.sh"
    robust_meme_hate_detection.train.stage1
    --config configs/stage1.yaml
    --out "$JOB_RESULTS_ROOT/$JOB"
)

# (c) Eval — clean metrics + threshold sweep + PGD smoke. Set CKPT first.
# CKPT="$JOB_RESULTS_ROOT/<your-train-job>/ckpt/best.pt"
# CMD=(
#     "$REPO_DIR/scripts/cluster_entrypoint.sh"
#     robust_meme_hate_detection.eval.run_eval
#     --ckpt "$CKPT"
#     --config configs/stage1.yaml
#     --out "$JOB_RESULTS_ROOT/$JOB"
# )

# ---------------- SUBMIT ---------------------------------------------------

runai submit \
    --name "$JOB" \
    --image "$IMAGE" \
    --gpu 1 \
    --run-as-uid "$USER_UID" \
    --large-shm \
    --existing-pvc "claimname=$SCRATCH_PVC,path=$SCRATCH_MOUNT" \
    --existing-pvc "claimname=$HOME_PVC,path=$HOME_MOUNT" \
    --environment REPO_DIR="$REPO_DIR" \
    --command -- "${CMD[@]}"

cat <<EOF

Submitted: $JOB
Image    : $IMAGE
Watch    : runai logs $JOB         (add -f to follow)
Status   : runai describe job $JOB
Results  : $JOB_RESULTS_ROOT/$JOB   (on shared /scratch)
EOF
