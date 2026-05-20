#!/usr/bin/env bash
set -euo pipefail

# submit_runai.sh - convenience wrapper for runai submit
# Usage: ./submit_runai.sh [options]
# Options:
#   -u UID                Numeric UID to run as (required)
#   -U USERNAME           username (required)
#   -g GROUP_NUMBER       group number (default: 49)
#   -i IMAGE              container image (default registry.rcp.epfl.ch/ee559/environment-with-packages:latest)
#   -G GPU                number of GPUs (default: 1)
#   -s SCRATCH_PVC        scratch PVC name (default: course-ee-559-scratch-g49)
#   -H HOME_PVC           home PVC name (default: home)
#   -r SHARED_RO_PVC      shared read-only PVC (default: course-ee-559-shared-ro)
#   -w SHARED_RW_PVC      shared read-write PVC (default: course-ee-559-shared-rw)
#   -n JOB_NAME           runai job name (default: job-<timestamp>)
#   -c CMD                command to run (default: python3 <HOME>/practice_3_repository/practice_3_simplified.py)
#   -d DATASET_PATH       dataset path inside job (default: /home/<USERNAME>/)
#   -o RESULTS_PATH       results path inside job (default: /home/<USERNAME>/practice_3_repository/results/)
#   --dry-run             print the final runai command and exit
#   -h                    show this help

IMAGE_DEFAULT="registry.rcp.epfl.ch/ee559/environment-with-packages:latest"
GPU_DEFAULT=1
SCRATCH_PVC_DEFAULT="course-ee-559-scratch-g49"
HOME_PVC_DEFAULT="home"
SHARED_RO_PVC_DEFAULT="course-ee-559-shared-ro"
SHARED_RW_PVC_DEFAULT="course-ee-559-shared-rw"
GROUP_NUMBER_DEFAULT=49

print_help() {
  sed -n '1,200p' "$0" | sed -n '1,200p' >/dev/stderr
}

# defaults
UID="316497"
USERNAME="hgruber"
GROUP_NUMBER="$GROUP_NUMBER_DEFAULT"
IMAGE="$IMAGE_DEFAULT"
GPU="$GPU_DEFAULT"
SCRATCH_PVC="$SCRATCH_PVC_DEFAULT"
HOME_PVC="$HOME_PVC_DEFAULT"
SHARED_RO_PVC="$SHARED_RO_PVC_DEFAULT"
SHARED_RW_PVC="$SHARED_RW_PVC_DEFAULT"
JOB_NAME="job-$(date +%Y%m%d-%H%M%S)"
CMD=""
DATASET_PATH=""
RESULTS_PATH=""
DRY_RUN=0

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -u) UID="$2"; shift 2 ;;
    -U) USERNAME="$2"; shift 2 ;;
    -g) GROUP_NUMBER="$2"; shift 2 ;;
    -i) IMAGE="$2"; shift 2 ;;
    -G) GPU="$2"; shift 2 ;;
    -s) SCRATCH_PVC="$2"; shift 2 ;;
    -H) HOME_PVC="$2"; shift 2 ;;
    -r) SHARED_RO_PVC="$2"; shift 2 ;;
    -w) SHARED_RW_PVC="$2"; shift 2 ;;
    -n) JOB_NAME="$2"; shift 2 ;;
    -c) CMD="$2"; shift 2 ;;
    -d) DATASET_PATH="$2"; shift 2 ;;
    -o) RESULTS_PATH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; print_help; exit 1 ;;
  esac
done

# required checks
if [[ -z "$UID" || -z "$USERNAME" ]]; then
  echo "ERROR: -u UID and -U USERNAME are required" >&2
  print_help
  exit 2
fi

# sensible defaults for CMD/DATA/RESULTS if not provided
if [[ -z "$CMD" ]]; then
  CMD="python3 /home/${USERNAME}/practice_3_repository/practice_3_simplified.py"
fi
if [[ -z "$DATASET_PATH" ]]; then
  DATASET_PATH="/home/${USERNAME}/"
fi
if [[ -z "$RESULTS_PATH" ]]; then
  RESULTS_PATH="/home/${USERNAME}/practice_3_repository/results/"
fi

# check runai is available
if ! command -v runai >/dev/null 2>&1; then
  echo "ERROR: runai CLI not found in PATH. Install/ensure runai is available." >&2
  exit 3
fi

# build PVC options
PVC_SCRATCH="--existing-pvc claimname=${SCRATCH_PVC},path=/scratch"
PVC_HOME="--existing-pvc claimname=${HOME_PVC},path=/home/${USERNAME}"
PVC_SHARED_RO="--existing-pvc claimname=${SHARED_RO_PVC},path=/shared-ro"
PVC_SHARED_RW="--existing-pvc claimname=${SHARED_RW_PVC},path=/shared-rw"

# Build the full command array to avoid quoting headaches
# Using --command -- <cmd> so runai interprets the rest as the command
RUNAI_CMD=(runai submit
  --run-as-uid "${UID}"
  --image "${IMAGE}"
  --gpu "${GPU}"
  ${PVC_SCRATCH}
  ${PVC_HOME}
  ${PVC_SHARED_RO}
  ${PVC_SHARED_RW}
  --command --)

# Append the command and its args
# Split CMD into words safely using eval and array assignment
# Note: we guard by using a subshell; this is standard for simple command strings.
# If you prefer to pass a fixed program and args, adjust invocation accordingly.
read -r -a CMD_ARR <<< "$(printf '%s\n' $CMD)"
for part in "${CMD_ARR[@]}"; do
  RUNAI_CMD+=("$part")
done

# Environment variables to mount in the job (optional). Example: pass dataset/results via env
# If you want these set inside the container, append: --env KEY=VALUE
RUNAI_CMD+=(--env "DATASET_PATH=${DATASET_PATH}" --env "RESULTS_PATH=${RESULTS_PATH}")

# Final command output for dry-run or execution
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== DRY RUN: final runai command ==="
  printf '%q ' "${RUNAI_CMD[@]}"
  echo
  exit 0
fi

# Execute
echo "Submitting runai job '${JOB_NAME}'..."
printf '%s ' "${RUNAI_CMD[@]}"
echo
# submit and capture exit code
"${RUNAI_CMD[@]}"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "runai submit failed with exit code $RC" >&2
  exit $RC
fi

echo "runai submit succeeded."
exit 0