#!/bin/bash
# Figure 1C — step 3/3: render a genome dot plot per target from each nucmer .delta
# (mummerplot -png -> gnuplot). Shared paths/tools come from _genome_dist_env.sh.
# See README.md. (The figure-reproduction notebook renders the same .delta/.coords
# with matplotlib instead — reproduction/figure_01/panels/run_mummer.sh — for a
# scheduler-/gnuplot-free path.)
#
#   bash 3_mummerplot.sh <data_type> [--dry-run]
source "$(dirname "${BASH_SOURCE[0]}")/_genome_dist_env.sh"
gd_init "${1:-}" mummer "${@:2}"
gd_need mummerplot
gd_need gnuplot

for tgt in "$FASTA_DIR"/*.cleaned.fasta; do
  base="$(basename "$tgt" .cleaned.fasta)"
  delta="$OUTPUT_DIR/nucmer_aln_${DATA_TYPE}_${ref_base_name}_${base}.delta"
  pfx="$OUTPUT_DIR/3_mummerplot_${DATA_TYPE}_${ref_base_name}_${base}"
  [[ -f "$delta" ]] || { echo "[skip] missing $delta (run 1_nucmer_aln_genome.sh first)"; continue; }
  echo "[mummerplot] $base"
  gd_run mummerplot -png "$delta" -R "$REF_FASTA" -Q "$tgt" -p "$pfx"
  gd_run gnuplot "$pfx.gp"
done
echo "[done] dot-plot .png files in $OUTPUT_DIR"
