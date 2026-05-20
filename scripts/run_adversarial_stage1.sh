#!/usr/bin/env bash
set -euo pipefail

# Run:AI submit helper for Stage-1 adversarial training.
# Override defaults with environment variables, e.g.:
#   JOB=adversarial-baseline-stage1-seed0 \
#   CKPT=/scratch/.../ckpt/best.pt \
#   OUT_DIR=/scratch/robust-meme-hate-detection/experiments/adversarial-baseline-stage1-seed0 \
#   bash scripts/run_adversarial_stage1.sh

# ---------------- EDIT ME --------------------------------------------------

JOB="${JOB:-adversarial-balanced-stage1-seed0-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2}"
USER_UID="316497"

HOME_PVC="${HOME_PVC:-home}"
HOME_MOUNT="${HOME_MOUNT:-$HOME}"

REPO_DIR="${REPO_DIR:-$HOME_MOUNT/robust-meme-hate-detection}"

SCRATCH_PVC="${SCRATCH_PVC:-course-ee-559-scratch-g49}"
SCRATCH_MOUNT="${SCRATCH_MOUNT:-/scratch}"
JOB_RESULTS_ROOT="${JOB_RESULTS_ROOT:-/scratch/robust-meme-hate-detection/experiments}"

SHARED_RO_PVC="${SHARED_RO_PVC:-course-ee-559-shared-ro}"
SHARED_RW_PVC="${SHARED_RW_PVC:-course-ee-559-shared-rw}"

CKPT="${CKPT:-/scratch/robust-meme-hate-detection/experiments/stage1-seed0-20260520-111021/ckpt/best.pt}"
CONFIG="${CONFIG:-$REPO_DIR/configs/stage1_adversarial.yaml}"
OUT_DIR="${OUT_DIR:-$JOB_RESULTS_ROOT/$JOB}"

SEED="${SEED:-0}"

# Keep the adversarial settings in the YAML config. If you need a different
# epsilon schedule or PGD setup, edit configs/stage1_adversarial.yaml.

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
  --command -- "$ENTRYPOINT_SCRIPT" robust_meme_hate_detection.train.stage1_adversarial \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR" \
  --seed "$SEED"

cat <<EOF

Submitted: $JOB
Image    : $IMAGE
Watch    : runai logs $JOB         (add -f to follow)
Status   : runai describe job $JOB
Results  : $JOB_RESULTS_ROOT/$JOB   (on shared /scratch)
EOF
