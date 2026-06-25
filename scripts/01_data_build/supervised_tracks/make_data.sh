#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Supervised-track TFRecord build (hound_data.py)
# ---------------------------------------------------------------------------
# Turns the 5,215-track targets sheet + R64 genome into the 8-fold TFRecord
# dataset that Shorkie (finetuned and scratch) trains on. This is the final
# stage of the supervised pipeline; the FASTQ -> BAM -> BigWig stages that
# produce the tracks listed in the sheet are documented in README.md.
#
# All site-specific inputs come from config/paths.yaml via scripts/common/env.sh
# (genome, blacklist/gaps BEDs, targets sheet, baskerville scripts). The exact
# hound_data.py flags below reproduce the released dataset.
#
# Usage:
#   scripts/01_data_build/supervised_tracks/make_data.sh [--out-dir DIR] [--dry-run]
#
#   --out-dir   where to write the TFRecord dataset (default: <supervised_root>/data)
#   --dry-run   print the exact hound_data.py command and exit (no build)
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"

DRY_RUN=0; OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1;    shift;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

FASTA="$(cfg genome.r64_supervised_fasta)"   # SGD S288C R64-3-1
SHEET="$(cfg datasets.targets_sheet)"         # cleaned_sheet.txt (5215 tracks)
MASK="$(cfg genome.mask_bed)"                 # mask_rossi.bed (blacklist + unmappable)
GAPS="$(cfg genome.gaps_bed)"                 # assembly_gaps.bed
SUP_ROOT="$(cfg datasets.supervised_root)"
[[ -n "$OUT_DIR" ]] || OUT_DIR="$SUP_ROOT/data"
HOUND_DATA="${BASKERVILLE_SCRIPTS}/hound_data.py"

# hound_data.py flags (verbatim from the released build):
#   --local         run locally (no SLURM)        -p 200   200 parallel procs
#   -l 16384        sequence length               -w 16    16 bp pooling/bins
#   -c 1024         crop 1024 bp off each end      -f 8     8-fold CV split
#   -r 256          256 sequences per TFRecord     --stride 6165  train stride (bp)
#   -b <mask>       blacklist -> baseline value    -u <mask>      unmappable regions
#   --umap_clip 0.5 clip unmappable to 50th pct    -g <gaps>      assembly-gap exclusion
#   -o <out>        output dir (relative to CWD)
ARGS=(
  --local
  -p 200
  -l 16384
  -w 16
  -c 1024
  -f 8
  -r 256
  --stride 6165
  -b "$MASK"
  -u "$MASK"
  --umap_clip 0.5
  -g "$GAPS"
  -o "$OUT_DIR"
  "$FASTA" "$SHEET"
)

echo "hound_data.py : $HOUND_DATA"
echo "genome FASTA  : $FASTA"
echo "targets sheet : $SHEET"
echo "blacklist/umap: $MASK"
echo "assembly gaps : $GAPS"
echo "output dir    : $OUT_DIR"
echo
echo "+ python $HOUND_DATA ${ARGS[*]}"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "(dry run — not executing)"
  exit 0
fi

for f in "$HOUND_DATA" "$FASTA" "$SHEET" "$MASK" "$GAPS"; do
  [[ -e "$f" ]] || { echo "MISSING input: $f" >&2; exit 1; }
done
exec python "$HOUND_DATA" "${ARGS[@]}"
