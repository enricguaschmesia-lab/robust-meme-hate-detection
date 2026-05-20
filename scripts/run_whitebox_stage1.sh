#!/usr/bin/env bash
set -euo pipefail

# Run:AI submit helper for Phase-4 white-box eval on the Stage-1 checkpoint.
# Override defaults with environment variables, e.g.:
#   JOB=whitebox-stage1-seed0 \
#   CKPT=/mnt/.../ckpt/best.pt \
#   OUT_DIR=/scratch/robust-meme-hate-detection/whitebox-stage1-seed0 \
#   bash scripts/run_whitebox_stage1.sh

# ---------------- EDIT ME --------------------------------------------------

JOB="${JOB:-whitebox-with-balanced-adversarial-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2}"
USER_UID="${USER_UID:-$(id -u 2>/dev/null || echo 0)}"

HOME_PVC="${HOME_PVC:-home}"
HOME_MOUNT="${HOME_MOUNT:-$HOME}"

REPO_DIR="${REPO_DIR:-$HOME_MOUNT/robust-meme-hate-detection}"

SCRATCH_PVC="${SCRATCH_PVC:-course-ee-559-scratch-g49}"
SCRATCH_MOUNT="${SCRATCH_MOUNT:-/scratch}"
JOB_RESULTS_ROOT="${JOB_RESULTS_ROOT:-/scratch/robust-meme-hate-detection/experiments}"

SHARED_RO_PVC="${SHARED_RO_PVC:-course-ee-559-shared-ro}"
SHARED_RW_PVC="${SHARED_RW_PVC:-course-ee-559-shared-rw}"

CKPT="${CKPT:-/scratch/robust-meme-hate-detection/experiments/adversarial-balanced-stage1-seed0-20260520-161106/ckpt/best.pt}"
# CKPT="${CKPT:-/scratch/robust-meme-hate-detection/experiments/stage1-seed0-20260520-111021/ckpt/best.pt}"
# CKPT="${CKPT:-/scratch/robust-meme-hate-detection/experiments/adversarial-baseline-stage1-seed0-20260520-144814/ckpt/best.pt}"
CONFIG="${CONFIG:-$REPO_DIR/configs/stage1.yaml}"
OUT_DIR="${OUT_DIR:-$JOB_RESULTS_ROOT/$JOB}"

MODALITY="${MODALITY:-multimodal}"
ATTACKS="${ATTACKS:-fgsm,pgd}"
EPSILONS="${EPSILONS:-1,2,4,8}"
MAX_EPSILON="${MAX_EPSILON:-0}"
PGD_STEPS="${PGD_STEPS:-100}"
PGD_ALPHA_FRAC="${PGD_ALPHA_FRAC:-0.25}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-4}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SEED="${SEED:-24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NORM="linf"

# Do not create $OUT_DIR locally — results live on the cluster's shared
# /scratch mounted into the job. Creating `/scratch/...` on the jumphost
# can fail with permission errors, so leave directory creation to the
# container/pod (the job will write into $JOB_RESULTS_ROOT/$JOB).

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
  --command -- "$ENTRYPOINT_SCRIPT" robust_meme_hate_detection.eval.run_whitebox \
  --ckpt "$CKPT" \
  --config "$CONFIG" \
  --out "$OUT_DIR" \
  --modality "$MODALITY" \
  --attacks "$ATTACKS" \
  --epsilons "$EPSILONS" \
  --max-epsilon "$MAX_EPSILON" \
  --pgd-steps "$PGD_STEPS" \
  --pgd-alpha-frac "$PGD_ALPHA_FRAC" \
  --sample-limit "$SAMPLE_LIMIT" \
  --max-batches "$MAX_BATCHES" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --norm "$NORM"

cat <<EOF

Submitted: $JOB
Image    : $IMAGE
Watch    : runai logs $JOB         (add -f to follow)
Status   : runai describe job $JOB
Results  : $JOB_RESULTS_ROOT/$JOB   (on shared /scratch)
EOF
