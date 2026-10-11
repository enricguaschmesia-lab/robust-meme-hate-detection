#!/usr/bin/env bash
# Drive the unified-robust train + test matrix on the Run:AI cluster.
#
# Modes  : nat | adv | both   (configs/robust_{nat,adv,both}.yaml)
# Seeds  : 0 1 2
# Splits : dev test_seen test_unseen
#
# Each job is submitted via cluster/cluster.sh, waited on, and its artifacts
# (and, for training, checkpoints) pulled back. Jobs run sequentially because
# the project has a single GPU allotment, so this also serialises the queue.
#
# Usage:
#   scripts/run_unified_matrix.sh smoke                 # cluster smoke (both mode)
#   scripts/run_unified_matrix.sh train [mode] [seed]   # all, or one cell
#   scripts/run_unified_matrix.sh eval  [mode] [seed] [split]
#   scripts/run_unified_matrix.sh all                   # train all, then eval all
# NOTE: deliberately no `set -e`. These jobs poll the cluster over SSH for many
# minutes; a single transient SSH blip must not abort a 50-job sweep. We handle
# errors explicitly and keep going (the skip-guard makes the sweep resumable).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="$HERE/cluster/cluster.sh"
SSH_HOST="${SSH_HOST:-epfl-jumphost}"
ENTRY=/home/guasch/robust-meme-hate-detection/scripts/cluster_entrypoint.sh
EXP=/scratch/robust-meme-hate-detection/experiments
POLL="${POLL:-15}"

# Robust terminal-state wait: polls Status over SSH and only returns once the
# job is terminal. Transient SSH/describe failures are swallowed (|| true) so a
# network blip during a long job cannot kill the sweep.
wait_terminal() {
  local job="$1" st
  while true; do
    st="$(ssh -o BatchMode=yes "$SSH_HOST" \
      "SUPPRESS_DEPRECATION_MESSAGE=true runai describe job $job 2>/dev/null | awk -F': ' '/^Status:/{print \$2; exit}'" \
      2>/dev/null || true)"
    case "$st" in
      Succeeded|Failed|Cancelled|Deleted) echo "WAIT ${job} -> ${st}"; return 0 ;;
    esac
    sleep "$POLL"
  done
}

declare -A CONFIG=(
  [nat]=configs/robust_nat.yaml
  [adv]=configs/robust_adv.yaml
  [both]=configs/robust_both.yaml
  [advfrozen]=configs/robust_adv_frozen.yaml
  [bothfrozen]=configs/robust_both_frozen.yaml
)
ALL_MODES=(nat adv both)
ALL_SEEDS=(0 1 2)
ALL_SPLITS=(dev test_seen test_unseen)

train_one() {
  local mode="$1" seed="$2"
  local job="train-unified-${mode}-seed${seed}"
  echo ">>> TRAIN ${job}"
  "$CLUSTER" submit-cmd "$job" -- "$ENTRY" \
    robust_meme_hate_detection.train.stage1_robust_unified \
    --config "${CONFIG[$mode]}" --seed "$seed" --out "$EXP/$job"
  wait_terminal "$job"
  "$CLUSTER" pull-artifacts "$job" || true
  "$CLUSTER" pull-checkpoints "$job" || true
}

eval_one() {
  local mode="$1" seed="$2" split="$3"
  local train_job="train-unified-${mode}-seed${seed}"
  local ckpt="$EXP/$train_job/ckpt/best.pt"
  local cfg="${CONFIG[$mode]}"
  # Run:AI job names must be lower-case alnum/'-' only, so the split tag in the
  # job name uses dashes (test_seen -> test-seen); the real split (with the
  # underscore) is still passed to --split.
  local tag="${split//_/-}"

  # The default mirror (/scratch/datasets/hate_meta) only has train/dev. The
  # held-out labelled test_seen/test_unseen sets live in a separate mirror, so
  # override --dataset-root for those splits (images resolve inside it too).
  local extra=()
  if [[ "$split" == test_* ]]; then
    extra=(--dataset-root /scratch/robust-meme-hate-detection/data/test_labels)
  fi

  local pj="perturbed-unified-${mode}-seed${seed}-${tag}"
  if [[ -f "$HERE/cluster-results/$pj/perturbed_eval.json" ]]; then
    echo "=== SKIP ${pj} (already pulled)"
  else
    echo ">>> PERTURBED ${pj}"
    "$CLUSTER" submit-cmd "$pj" -- "$ENTRY" \
      robust_meme_hate_detection.eval.run_perturbed \
      --ckpt "$ckpt" --config "$cfg" --split "$split" "${extra[@]}" --out "$EXP/$pj"
    wait_terminal "$pj"
    "$CLUSTER" pull-artifacts "$pj" || true
  fi

  local wj="whitebox-unified-${mode}-seed${seed}-${tag}"
  if [[ -f "$HERE/cluster-results/$wj/whitebox_eval.json" ]]; then
    echo "=== SKIP ${wj} (already pulled)"
  else
    echo ">>> WHITEBOX ${wj}"
    "$CLUSTER" submit-cmd "$wj" -- "$ENTRY" \
      robust_meme_hate_detection.eval.run_whitebox \
      --ckpt "$ckpt" --config "$cfg" --split "$split" "${extra[@]}" --out "$EXP/$wj"
    wait_terminal "$wj"
    "$CLUSTER" pull-artifacts "$wj" || true
  fi
}

cmd_smoke() {
  local job="smoke-unified-both"
  echo ">>> SMOKE ${job}"
  "$CLUSTER" submit-cmd "$job" -- "$ENTRY" \
    robust_meme_hate_detection.train.stage1_robust_unified \
    --config configs/smoke_robust.yaml --seed 0 --out "$EXP/$job"
  wait_terminal "$job"
  "$CLUSTER" pull-artifacts "$job" || true
}

cmd_train() {
  local modes=("${ALL_MODES[@]}") seeds=("${ALL_SEEDS[@]}")
  [[ $# -ge 1 ]] && modes=("$1")
  [[ $# -ge 2 ]] && seeds=("$2")
  for m in "${modes[@]}"; do for s in "${seeds[@]}"; do train_one "$m" "$s"; done; done
}

cmd_eval() {
  local modes=("${ALL_MODES[@]}") seeds=("${ALL_SEEDS[@]}") splits=("${ALL_SPLITS[@]}")
  [[ $# -ge 1 ]] && modes=("$1")
  [[ $# -ge 2 ]] && seeds=("$2")
  [[ $# -ge 3 ]] && splits=("$3")
  for m in "${modes[@]}"; do for s in "${seeds[@]}"; do for sp in "${splits[@]}"; do eval_one "$m" "$s" "$sp"; done; done; done
}

case "${1:-}" in
  smoke) shift; cmd_smoke "$@" ;;
  train) shift; cmd_train "$@" ;;
  eval)  shift; cmd_eval "$@" ;;
  all)   cmd_train; cmd_eval ;;
  *) echo "Usage: $0 {smoke|train [mode] [seed]|eval [mode] [seed] [split]|all}"; exit 1 ;;
esac
