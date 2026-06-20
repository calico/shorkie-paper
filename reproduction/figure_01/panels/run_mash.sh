#!/bin/bash
# Figure 1D — Mash distance of the R64 reference vs every genome in the
# 165_Saccharomycetales and 80_Strains datasets. Full recompute via `mash dist`.
# Faithful to scripts/03_eval/lm/genome_evaluation/3_genome_dist/mash/1_mash_genome.sh
# (one ref vs all targets), pointed at the on-disk corpus. Output tables are parsed
# + plotted (sorted ascending bars) by reproduce_figure_01.ipynb.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

# mash lives in the yeast_ml env (installed via bioconda); fall back to base.
if ! command -v mash >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
  export PATH="$PATH:$CONDA_BASE/bin"
fi
command -v mash >/dev/null || { echo "ERROR: mash not found on PATH"; exit 127; }

LCS="$(cfg datasets.lm_corpus_split_root)"
OUT="reproduction/figure_01/reproduced/panelD_mash"
mkdir -p "$OUT"
REF="$LCS/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"
echo "REF = $REF ; mash $(mash --version)"

for tier in saccharomycetales strains; do
  fdir="$LCS/data_${tier}_gtf/fasta"
  out="$OUT/${tier}_dist.tab"
  echo "[mash dist] R64 vs $(ls "$fdir"/*.cleaned.fasta | wc -l) $tier genomes"
  # columns: ref-id  query-id  distance  p-value  shared-hashes
  mash dist "$REF" "$fdir"/*.cleaned.fasta > "$out"
  echo "  -> $out ($(wc -l < "$out") rows)"
done
echo "[done] panel D distance tables in $OUT"
