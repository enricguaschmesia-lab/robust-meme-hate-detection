#!/usr/bin/env bash
# Launcher used by Run:AI jobs. Receives a python module and its args,
# sets PYTHONPATH for the src/-layout package, cd's to the repo root, and
# execs python3.
#
# Usage:
#   cluster_entrypoint.sh <module-name> [args...]
# Example:
#   cluster_entrypoint.sh robust_meme_hate_detection.train.stage1 \
#       --config configs/stage1.yaml --seed 0 --out /scratch/.../experiments/$JOB

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/guasch/robust-meme-hate-detection}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODULE="${1:?Missing module name as first argument}"
shift
exec python3 -m "$MODULE" "$@"
