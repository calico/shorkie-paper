#!/bin/bash
# Figure 1C — MUMmer genome dot plots.
# Full recompute of nucmer alignments of the R64 reference (x-axis) against one
# representative genome per dataset (y-axis), then show-coords -> .coords text
# that the reproduction notebook renders as dot plots with matplotlib.
#
# Faithful to scripts/03_eval/lm/genome_evaluation/3_genome_dist/{1_nucmer,2_show_coords}.sh,
# but pointed at the on-disk corpus (datasets.lm_corpus_split_root) instead of the
# legacy corpus_build_data_root, and emitting .coords (mummerplot/gnuplot replaced
# by matplotlib in the notebook for portability). See figure_01/README.md.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

# MUMmer (nucmer/show-coords) lives in the conda *base* env on this cluster, which
# activating yeast_ml drops from PATH. Append base/bin so the tools resolve while
# keeping yeast_ml's python (for shorkie) first. Edit/override for your site.
if ! command -v nucmer >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
  export PATH="$PATH:$CONDA_BASE/bin"
fi
command -v nucmer >/dev/null || { echo "ERROR: nucmer not found on PATH"; exit 127; }

LCS="$(cfg datasets.lm_corpus_split_root)"
OUT="reproduction/figure_01/reproduced/panelC_mummer"
mkdir -p "$OUT"

REF="$LCS/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"

# Representative target per dataset: "label|tier_dir|accession_basename"
TARGETS=(
  "R64-1.1 (species)|data_r64_gtf|GCA_000146045_2"
  "YJM1078 (strain, repr.)|data_strains_gtf|GCA_000975645_3"
  "N. glabratus CBS138 (order)|data_saccharomycetales_gtf|GCA_000002545_2"
  "C. albicans SC5314 (order)|data_saccharomycetales_gtf|GCA_000182965_3"
  "N. crassa OR74A (kingdom)|data_fungi_1385_gtf|GCA_000182925_2"
  "S. pombe 972h (kingdom)|data_fungi_1385_gtf|GCA_000002945_2"
)

echo "REF = $REF"
for entry in "${TARGETS[@]}"; do
  IFS='|' read -r label tier acc <<< "$entry"
  tgt="$LCS/$tier/fasta/${acc}.cleaned.fasta"
  if [[ ! -f "$tgt" ]]; then echo "[skip] missing $tgt"; continue; fi
  pfx="$OUT/$acc"
  echo "[nucmer] R64 vs $label  ($acc)"
  nucmer -t 8 -p "$pfx" "$REF" "$tgt"
  # tab-delimited, with seq lengths; -r sort by ref. Header rows start with non-numeric.
  show-coords -r -l -T "$pfx.delta" > "$pfx.coords"
  echo "  -> $pfx.coords ($(wc -l < "$pfx.coords") lines)"
done
echo "[done] panel C coords in $OUT"
