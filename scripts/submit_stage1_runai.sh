#!/usr/bin/env bash
set -euo pipefail

# submit_stage1_runai.sh
# Helper to submit `stage1.py` via `runai submit` with configurable python args.
# Usage example (dry-run first):
# ./scripts/submit_stage1_runai.sh --uid 316497 --user hgruber --job stage1-seed0 \
#   --config /home/hgruber/robust-meme-hate-detection/configs/stage1.yaml \
#   --seed 0 --out /home/hgruber/robust-meme-hate-detection/data/models/stage1-seed0 --dry-run

print_help() {
  cat <<'USAGE'
Usage: submit_stage1_runai.sh [options]

Options:
  --uid UID             Numeric UID to run the container as (required)
  --user USERNAME       Username (required)
  --job JOB_NAME        RunAI job name (default: job-<timestamp>)
  --image IMAGE         Container image (default registry.rcp.epfl.ch/ee559/environment-with-packages:latest)
  --gpu N               GPUs to request (default: 1)
  --scratch-pvc NAME    Scratch PVC claimname (default: course-ee-559-scratch-g49)
  --home-pvc NAME       Home PVC claimname (default: home)
  --shared-ro NAME      Shared read-only PVC claimname (default: course-ee-559-shared-ro)
  --shared-rw NAME      Shared read-write PVC claimname (default: course-ee-559-shared-rw)
  --config PATH         Python --config arg passed to stage1.py (required)
  --seed N              Python --seed arg (default: 0)
  --out PATH            Python --out arg (required)
  --dry-run             Print the runai command and exit
  -h, --help            Show this help
USAGE
}

# Defaults
IMAGE_DEFAULT="registry.rcp.epfl.ch/ee-559-guasch/robust-meme-hate-detection:v0.2"
GPU_DEFAULT=1
SCRATCH_PVC_DEFAULT="course-ee-559-scratch-g49"
HOME_PVC_DEFAULT="home"
SHARED_RO_PVC_DEFAULT="course-ee-559-shared-ro"
SHARED_RW_PVC_DEFAULT="course-ee-559-shared-rw"
REPO_DIR_DEFAULT="/home/hgruber/robust-meme-hate-detection"

RUN_UID="316497"
USERNAME="hgruber"
JOB_NAME="job-$(date +%Y%m%d-%H%M%S)"
IMAGE="$IMAGE_DEFAULT"
GPU="$GPU_DEFAULT"
SCRATCH_PVC="$SCRATCH_PVC_DEFAULT"
HOME_PVC="$HOME_PVC_DEFAULT"
SHARED_RO_PVC="$SHARED_RO_PVC_DEFAULT"
SHARED_RW_PVC="$SHARED_RW_PVC_DEFAULT"
REPO_DIR="$REPO_DIR_DEFAULT"
PY_CONFIG="/home/hgruber/robust-meme-hate-detection/configs/stage1.yaml"
PY_SEED=0
PY_OUT="/home/hgruber/robust-meme-hate-detection/data/models"
DRY_RUN=0

if [[ -z "$RUN_UID" || -z "$USERNAME" ]]; then
  echo "ERROR: --uid and --user are required" >&2
  print_help
  exit 2
fi
if [[ -z "$PY_CONFIG" || -z "$PY_OUT" ]]; then
  echo "ERROR: --config and --out are required" >&2
  print_help
  exit 2
fi

# Path to the python script inside the container (mounted at /home/<username>)
ENTRYPOINT_SCRIPT="${REPO_DIR}/scripts/cluster_entrypoint.sh"

# Build PVC flags
PVC_SCRATCH=(--existing-pvc "claimname=${SCRATCH_PVC},path=/scratch")
PVC_HOME=(--existing-pvc "claimname=${HOME_PVC},path=/home/${USERNAME}")
PVC_SHARED_RO=(--existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro")
PVC_SHARED_RW=(--existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw")

# Compose the runai command as an array for safety
RUNAI_CMD=(runai submit
  --name "${JOB_NAME}"
  --run-as-uid "${RUN_UID}"
  --image "${IMAGE}"
  --gpu "${GPU}"
  --environment REPO_DIR="${REPO_DIR}"
  "${PVC_SCRATCH[@]}"
  "${PVC_HOME[@]}"
  "${PVC_SHARED_RO[@]}"
  "${PVC_SHARED_RW[@]}"
  --command -- "${ENTRYPOINT_SCRIPT}" robust_meme_hate_detection.train.stage1 --config "${PY_CONFIG}" --seed "${PY_SEED}" --out "${PY_OUT}"
)

echo "Prepared runai command for job: ${JOB_NAME}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${RUNAI_CMD[@]}"
  echo
  exit 0
fi

# Verify runai exists
if ! command -v runai >/dev/null 2>&1; then
  echo "ERROR: runai CLI not found in PATH. Please install or run this on the jumphost where runai is available." >&2
  exit 3
fi

echo "Submitting job..."
"${RUNAI_CMD[@]}"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "runai submit failed with exit code $RC" >&2
  exit $RC
fi

echo "Submitted ${JOB_NAME} successfully."

cat <<EOF

Submitted: $JOB_NAME
Image    : $IMAGE
Watch    : runai logs $JOB_NAME         (add -f to follow)
Status   : runai describe job $JOB_NAME
EOF

exit 0
