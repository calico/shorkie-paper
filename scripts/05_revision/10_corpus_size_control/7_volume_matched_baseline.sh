#!/bin/bash
# Revision 10 step 7 — the OTHER half of the sweet-spot control: 165_Saccharomycetales
# downsampled to the 80_strains window count.
#
# The "sweet spot" is a peak, and the data-volume confound runs in OPPOSITE directions on
# its two sides:
#
#   upper side  Saccharomycetales (385,551 windows) beats 1341_Fungus (625,355)
#               -> the broad corpus has 1.62x MORE data, so volume cannot explain it.
#                  Steps 2-5 build the corpus-matched control for this side.
#
#   lower side  Saccharomycetales (385,551) beats 80_strains (102,315)
#               -> Saccharomycetales has 3.77x MORE data, so volume COULD explain all of it.
#                  Nothing in the paper, nor the corpus-matched control, tests this.
#
# This arm closes the lower side, and it is the cheaper of the two: no corpus build, no new
# genomes. It reuses the existing Saccharomycetales tier, subsamples its training windows to
# 102,315, rebuilds only the TFRecords, and retrains.
#
# It also matches the TRAINING SCHEDULE to 80_strains, which turns out to matter: the four
# published tiers were NOT trained on a common schedule (see 1_training_dynamics.py) --
# R64/80_strains used train_epochs_max=500 / patience=50 while Saccharomycetales/1341_Fungus
# used 10000 / 1000. Matching only the window count would leave that 20x difference standing
# and the arm would answer nothing. Both are matched here; set MATCH_SCHEDULE=0 to keep the
# Saccharomycetales schedule and isolate volume alone.
#
# Read-out:
#   still beats 80_strains  -> phylogenetic scope is doing the work on BOTH sides of the peak
#   ties with 80_strains    -> the lower half of the "sweet spot" is a data-volume/schedule
#                              artifact and the claim must be restated
#
# Usage:
#   scripts/common/submit.sh --profile gpu 7_volume_matched_baseline.sh
#   bash 7_volume_matched_baseline.sh --dry-run
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM="$REPO_ROOT/scripts/01_data_build/lm_corpus"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
run() { echo "+ $*"; [[ "$DRY_RUN" == 1 ]] || "$@"; }

TARGET_TRAIN="${TARGET_TRAIN:-102315}"      # the 80_strains training-window count
MATCH_SCHEDULE="${MATCH_SCHEDULE:-1}"
SRC_SUFFIX="_saccharomycetales_gtf"
DST_SUFFIX="_saccharomycetales_vol${TARGET_TRAIN}_gtf"
SPLIT_ROOT="${SPLIT_ROOT:-$(cfg datasets.lm_corpus_split_root)}"
SRC="$SPLIT_ROOT/data${SRC_SUFFIX}"
DST="$SPLIT_ROOT/data${DST_SUFFIX}"
OUT_DIR="${OUT_DIR:-$HERE/results/lm_saccharomycetales_vol${TARGET_TRAIN}_unet_small}"

if [[ ! -d "$SRC" ]]; then
  echo "error: source tier not found at $SRC" >&2
  echo "       fetch it with: bash data/download.sh --lm-corpus 165_Saccharomycetales -u PROJECT" >&2
  exit 1
fi

echo "=== volume-matched Saccharomycetales arm ==="
echo "    source tier : $SRC"
echo "    new tier    : $DST"
echo "    train windows -> $TARGET_TRAIN (from 385,551)"
echo "    schedule matched to 80_strains: $MATCH_SCHEDULE"

# The source tier is ~50 GB, almost all of it read-only FASTA/GTF, so link rather than copy;
# only the small per-split BEDs and statistics.json are duplicated, because those are what
# the subsampling rewrites.
run mkdir -p "$DST"
for d in fasta gtf extracted_fasta; do
  [[ -e "$SRC/$d" ]] && run ln -sfn "$SRC/$d" "$DST/$d"
done
for f in sequences.bed sequences_train.bed sequences_train.cleaned.bed \
         sequences_valid.bed sequences_valid.cleaned.bed \
         sequences_test.bed sequences_test.cleaned.bed statistics.json targets.txt; do
  [[ -e "$SRC/$f" ]] && run cp -f "$SRC/$f" "$DST/$f"
done

# Subsample the TRAINING split only. valid/test come from S. cerevisiae R64 and are identical
# across every tier by construction — if they moved, perplexity would stop being comparable
# to the published Figure 1G.
run python "$HERE/3_subsample_windows.py" \
    --bed "$DST/sequences_train.cleaned.bed" --target "$TARGET_TRAIN"

echo "--- Stage 4: TFRecord generation (ZLIB) for the reduced tier ---"
( run cd "$LM/4_tf_data_generation"
  run python 1_write_data_multi.py --save_suffix "$DST_SUFFIX" --out_dir "$SPLIT_ROOT/" \
      --run_local --processes 8 --use_gtf )

# Derive the params at run time rather than committing a second near-duplicate config; the
# only difference from params_unet_small.json is the schedule.
PARAMS="$OUT_DIR/params.json"
run mkdir -p "$OUT_DIR"
if [[ "$DRY_RUN" == 0 ]]; then
  python - "$HERE/params_unet_small.json" "$PARAMS" "$MATCH_SCHEDULE" <<'PY'
import json, sys
src, dst, match = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
p = json.load(open(src))
if match:
    # 80_strains schedule, so the arm differs from 80_strains only in phylogenetic scope.
    p["train"].update(train_epochs_max=500, train_epochs_min=50, patience=50)
json.dump(p, open(dst, "w"), indent=4)
print(f"wrote {dst} (schedule matched: {match})")
PY
fi

CMD=(python "${BASKERVILLE_SCRIPTS}/hound_train.py"
  --eval_dir "${DST}/"
  -o "${OUT_DIR}/train"
  "$PARAMS"
  "$DST")

echo "output  : $OUT_DIR"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$OUT_DIR/train"
"${CMD[@]}" 1>"$OUT_DIR/train/train.out" 2>"$OUT_DIR/train/train.err"
echo "=== done. Evaluate with:"
echo "    MODEL_DIR=$OUT_DIR SPLIT_ROOT=$SPLIT_ROOT bash $HERE/5_eval_perplexity.sh"
echo "    then compare against the 80_strains perplexity in Figure 1G ==="
