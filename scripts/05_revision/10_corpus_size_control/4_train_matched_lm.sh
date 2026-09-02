#!/bin/bash
# Revision 10 step 4 — pretrain the masked language model on the window-matched
# 165-genome fungal control corpus.
#
# The architecture is deliberately `unet_small`, NOT the released `unet_small_bert_drop`:
# the corpus-scaling comparison in question (Figure 1F/G) was run at
# `unet_small`, so the new point only lands on that figure if it uses the same architecture.
# `params_unet_small.json` next to this script is the exact config the published
# 165_Saccharomycetales `unet_small` run used, committed here so the control is
# self-contained.
#
# The control corpus has 165 genomes, so num_features = 4 (DNA) + 165 + 1 = 170 — identical
# to the 165_Saccharomycetales tier. Architecture, capacity, optimiser, schedule and
# held-out split are all held fixed; the ONLY difference is which 165 genomes the model
# reads. That is what makes it a control for phylogenetic scope.
#
# Usage:
#   scripts/common/submit.sh --profile gpu 4_train_matched_lm.sh
#   bash 4_train_matched_lm.sh --dry-run
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SUFFIX="_fungi165_matched_gtf"
SPLIT_ROOT="${SPLIT_ROOT:-$(cfg datasets.lm_corpus_split_root)}"
LM_DATA="$SPLIT_ROOT/data${SUFFIX}"
OUT_DIR="${OUT_DIR:-$HERE/results/lm_fungi165_matched_unet_small}"

if [[ ! -d "$LM_DATA" ]]; then
  echo "error: control corpus not found at $LM_DATA" >&2
  echo "       run 2_sample_matched_corpus.py then 3_build_corpus.sh first" >&2
  exit 1
fi

CMD=(python "${BASKERVILLE_SCRIPTS}/hound_train.py"
  --eval_dir "${LM_DATA}/"
  -o "${OUT_DIR}/train"
  "$HERE/params_unet_small.json"
  "${LM_DATA}")

echo "corpus  : $LM_DATA"
echo "output  : $OUT_DIR"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$OUT_DIR/train"
"${CMD[@]}" 1>"$OUT_DIR/train/train.out" 2>"$OUT_DIR/train/train.err"
echo "=== done. Next: 5_eval_perplexity.sh ==="
