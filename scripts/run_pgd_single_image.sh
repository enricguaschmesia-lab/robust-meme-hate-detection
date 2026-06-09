#!/usr/bin/env bash
set -euo pipefail

# Run:AI submit helper — PGD adversarial attack on a single image.
# Runs run_pgd_single.py from the repo root; edit that file to change
# image path, text, epsilon, or output path.
# Override defaults with environment variables, e.g.:
#   JOB=pgd-single-01456 \
#   CKPT=/scratch/.../ckpt/best.pt \
#   bash scripts/run_pgd_single_image.sh

# ---------------- EDIT ME --------------------------------------------------

JOB="${JOB:-pgd-single-image}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2}"
USER_UID="${USER_UID:-$(id -u 2>/dev/null || echo 0)}"

HOME_PVC="${HOME_PVC:-home}"
HOME_MOUNT="${HOME_MOUNT:-$HOME}"

REPO_DIR="${REPO_DIR:-$HOME_MOUNT/robust-meme-hate-detection}"

SCRATCH_PVC="${SCRATCH_PVC:-course-ee-559-scratch-g49}"
SCRATCH_MOUNT="${SCRATCH_MOUNT:-/scratch}"

SHARED_RO_PVC="${SHARED_RO_PVC:-course-ee-559-shared-ro}"
SHARED_RW_PVC="${SHARED_RW_PVC:-course-ee-559-shared-rw}"

CKPT="${CKPT:-/scratch/robust-meme-hate-detection/experiments/stage1-seed0-20260503-163917/ckpt/best.pt}"

ENTRYPOINT_SCRIPT="$REPO_DIR/scripts/cluster_entrypoint.sh"

# ---------------- SUBMIT ---------------------------------------------------

runai submit \
  --name "$JOB" \
  --image "$IMAGE" \
  --gpu 1 \
  --run-as-uid "$USER_UID" \
  --large-shm \
  --existing-pvc "claimname=$SCRATCH_PVC,path=$SCRATCH_MOUNT" \
  --existing-pvc "claimname=$HOME_PVC,path=$HOME_MOUNT" \
  --existing-pvc "claimname=$SHARED_RO_PVC,path=/shared-ro" \
  --existing-pvc "claimname=$SHARED_RW_PVC,path=/shared-rw" \
  --environment REPO_DIR="$REPO_DIR" \
  --environment CKPT="$CKPT" \
  --command -- "$ENTRYPOINT_SCRIPT" run_pgd_single

cat <<EOF

Submitted: $JOB
Image    : $IMAGE
Watch    : runai logs $JOB         (add -f to follow)
Status   : runai describe job $JOB
EOF
