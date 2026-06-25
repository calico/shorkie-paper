#!/bin/bash
# Shorkie_LM pretraining: masked DNA language model (hound_train.py, single fold)
# on the 165_Saccharomycetales corpus. Produces the --restore checkpoint that
# Shorkie_finetuned starts from.
#
# Run on a GPU node, e.g.:
#   scripts/common/submit.sh --profile gpu scripts/02_train/shorkie_lm/train.sh
# Add --dry-run to print the fully-resolved command without running it.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# 165_Saccharomycetales tier dir (tfrecords/ + statistics.json) the LM trains on.
LM_DATA="$(cfg datasets.lm_train_dir)"

CMD=(python "${BASKERVILLE_SCRIPTS}/hound_train.py"
  --eval_dir "${LM_DATA}/"
  -o train/
  params.json
  "${LM_DATA}")

if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

mkdir -p train
"${CMD[@]}" 1>train/train.out 2>train/train.err
