#!/bin/bash
# Revision 10 step 5 — held-out perplexity for the control LM, on the same S. cerevisiae
# test split as every published tier.
#
# This is the measurement that decides the question. Figure 1G reports test
# perplexity on R64 chrXII/chrXIV/chrXVI for each corpus tier; the control corpus was built
# with that identical held-out split and left untouched by the window subsampling, so its
# perplexity drops straight onto that axis.
#
# Reading the result:
#   control ~ 1341_Fungus   -> the broad corpus underperforms because of its SCOPE, and the
#                              window count was never the explanation
#   control ~ 165_Sacc      -> matching the data volume closes the gap, and the "sweet spot"
#                              claim should be weakened to a statement about corpus size
#   control between         -> both contribute; report the decomposition rather than a
#                              single cause
#
# Usage:
#   scripts/common/submit.sh --profile gpu 5_eval_perplexity.sh
#   bash 5_eval_perplexity.sh --dry-run
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
MODEL_DIR="${MODEL_DIR:-$HERE/results/lm_fungi165_matched_unet_small}"
CKPT="$MODEL_DIR/train/model_best.h5"
OUT_DIR="$MODEL_DIR/test_testset_perplexity_region"

if [[ ! -f "$CKPT" ]]; then
  echo "error: no checkpoint at $CKPT -- run 4_train_matched_lm.sh first" >&2
  exit 1
fi

# Same invocation as scripts/03_eval/lm/lm_model_eval/, so the number is directly
# comparable to the published Figure 1G perplexities.
CMD=(python "${BASKERVILLE_SCRIPTS}/hound_eval_mlm_perplexity_region.py"
  -o "$OUT_DIR" --rc --save --split test
  "$HERE/params_unet_small.json" "$CKPT" "$LM_DATA")

echo "checkpoint : $CKPT"
echo "output     : $OUT_DIR"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$OUT_DIR"
"${CMD[@]}" 1>"$OUT_DIR/test_testset_perplexity_region.out" \
            2>"$OUT_DIR/test_testset_perplexity_region.err"
echo "=== done. Compare against the published tiers with"
echo "    reproduction/figure_01/recheck/recompute_fig01.py ==="
