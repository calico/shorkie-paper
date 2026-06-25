#!/bin/bash

#SBATCH --job-name=shorkie_finetuned
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=20g
#SBATCH -o shorkie_finetuned.%j.out
#SBATCH --mail-type=start,end

# Shorkie_finetuned: 8-fold supervised ensemble, fine-tuned FROM the Shorkie_LM
# checkpoint (westminster_train_folds.py --restore). The ONLY differences vs
# Shorkie_scratch are this --restore flag and a few params.json[train] fields
# (see scripts/02_train/README.md).
#
# Submit with `sbatch make_model.sh`, or run portably (no scheduler) via
#   scripts/common/submit.sh --profile gpu scripts/02_train/shorkie_finetuned/make_model.sh
# Add --dry-run to print the fully-resolved command without launching.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

RESTORE="$(cfg models.shorkie_lm_checkpoint)"   # Shorkie_LM model_best.h5 (the --restore target)
EVAL_DIR="$(cfg datasets.lm_corpus_split_root)"
DATA_DIR="$(cfg datasets.supervised_data)"      # 8-fold supervised TFRecords

CMD=(python "${WESTMINSTER_SCRIPTS}/westminster_train_folds.py"
  --restart
  -f 8
  -e yeast_ml
  --restore "$RESTORE"
  --eval_dir "${EVAL_DIR}/"
  -o train
  -q a100
  --rc
  --shifts "0,1"
  params.json
  "$DATA_DIR")

if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

"${CMD[@]}" 1>train.out 2>train.err
