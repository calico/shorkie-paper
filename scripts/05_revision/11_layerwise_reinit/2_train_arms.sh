#!/bin/bash
# Revision 11 step 2 — fine-tune each partially re-initialised arm with the published
# supervised recipe.
#
# Every arm uses scripts/02_train/shorkie_finetuned/params.json UNCHANGED and the same
# 8-fold westminster_train_folds.py invocation as the published Shorkie, so the only
# difference between arms is which part of the LM checkpoint was reset. Two anchors already
# exist and must NOT be retrained: full-LM init is Shorkie itself, and all-random is
# Shorkie_Random_Init — both are released.
#
#   arm                published equivalent / what it isolates
#   -----------------  ------------------------------------------------------------------
#   (none)             Shorkie                — all LM weights kept          [released]
#   conv_reset         conv tower re-initialised, transformer + decoder kept
#   transformer_reset  transformer re-initialised, conv tower + decoder kept
#   decoder_reset      U-Net decoder re-initialised, conv + transformer kept
#   (none)             Shorkie_Random_Init    — nothing kept                 [released]
#
# Usage:
#   python 1_make_reinit_checkpoints.py
#   scripts/common/submit.sh --profile gpu 2_train_arms.sh conv_reset
#   bash 2_train_arms.sh transformer_reset --dry-run
#
# Cost: one full 8-fold supervised fine-tune per arm — the same cost as training Shorkie.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

ARM="${1:-conv_reset}"
DRY_RUN=0
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY_RUN=1; done

case "$ARM" in
  conv_reset|transformer_reset|decoder_reset) ;;
  *) echo "unknown arm '$ARM' (conv_reset|transformer_reset|decoder_reset)" >&2; exit 2;;
esac

CKPT="$HERE/results/checkpoints/${ARM}.h5"
if [[ ! -f "$CKPT" ]]; then
  echo "error: no checkpoint at $CKPT" >&2
  echo "       run: python $HERE/1_make_reinit_checkpoints.py" >&2
  exit 1
fi

SUP_DATA="$(cfg datasets.supervised_data)"
PARAMS="$REPO_ROOT/scripts/02_train/shorkie_finetuned/params.json"
# The trailing <model-tree>/train is load-bearing: westminster writes <out_dir>/f{N}c{M}/,
# while experiment 01's collector (which step 3 reuses to score the arms) expects
# <root>/<model-tree>/train/f{N}c0/eval/acc.txt — the layout the published trees have
# because westminster was run there with `-o train` from inside the model directory.
OUT_DIR="${OUT_DIR:-$HERE/results/arms/$ARM/self_supervised_unet_small_bert_drop/train}"

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

echo "arm     : $ARM"
echo "restore : $CKPT"
echo "output  : $OUT_DIR"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$(dirname "$OUT_DIR")"
"${CMD[@]}"
echo "=== done. Score all arms with 3_eval_arms.py ==="
