#!/bin/bash
# Revision 10 step 6 — fine-tune the control LM on the supervised tracks and score it
#  , OPTIONAL.
#
# Perplexity (step 5) answers the scope question directly. This step answers the
# question the paper actually cares about: does the corpus choice change DOWNSTREAM
# expression prediction, or only the language-model objective? The published Figure 1
# analysis found the two rankings agreed, so a disagreement here would itself be a result.
#
# It is the expensive step — a full 8-fold supervised fine-tune, the same cost as training
# Shorkie itself — so run it only if step 5 leaves the question open.
#
# The recipe is the published one: westminster_train_folds.py --restore <LM checkpoint>
# with scripts/02_train/shorkie_finetuned/params.json, unchanged, so the result is directly
# comparable to Shorkie. Score it with experiment 01's machinery, which gives the same
# metrics with cross-fold intervals:
#     python../01_headline_uncertainty/1_collect_fold_metrics.py \
#         --supervised_root <this results dir>
#
# Usage:
#   scripts/common/submit.sh --profile gpu 6_finetune_and_eval.sh
#   bash 6_finetune_and_eval.sh --dry-run
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

MODEL_DIR="${MODEL_DIR:-$HERE/results/lm_fungi165_matched_unet_small}"
CKPT="$MODEL_DIR/train/model_best.h5"
SUP_DATA="$(cfg datasets.supervised_data)"
PARAMS="$REPO_ROOT/scripts/02_train/shorkie_finetuned/params.json"
# Same <model-tree>/train suffix as experiment 11, so experiment 01's collector can score it.
OUT_DIR="${OUT_DIR:-$HERE/results/finetuned_from_matched_control/self_supervised_unet_small_bert_drop/train}"

if [[ ! -f "$CKPT" ]]; then
  echo "error: no checkpoint at $CKPT -- run 4_train_matched_lm.sh first" >&2
  exit 1
fi

# NOTE: --eval_dir is NOT an option of the pinned westminster (external/westminster at
# v0.0.1 / 735a45b); passing it aborts with "no such option". Without it westminster
# evaluates on the data's own held-out test fold, which is what we want here anyway —
# it is the same test split the published Shorkie and Shorkie_Random_Init numbers use,
# so the arms stay directly comparable.
CMD=(python "${WESTMINSTER_SCRIPTS}/westminster_train_folds.py"
  -f 8 -e yeast_ml
  --restart
  --restore "$CKPT"
  -o "$OUT_DIR"
  --rc --shifts "0,1"
  "$PARAMS" "$SUP_DATA")

echo "restore : $CKPT"
echo "data    : $SUP_DATA"
echo "output  : $OUT_DIR"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$(dirname "$OUT_DIR")"
"${CMD[@]}"
echo "=== done. Score with../01_headline_uncertainty/1_collect_fold_metrics.py ==="
