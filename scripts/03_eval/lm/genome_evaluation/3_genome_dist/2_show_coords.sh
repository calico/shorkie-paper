#!/bin/bash
# Figure 1C — step 2/3: extract human-readable alignment coordinates from each
# nucmer .delta produced by step 1 (show-coords -lcr -> one .txt per target).
# Shared paths/tools come from _genome_dist_env.sh. See README.md.
#
#   bash 2_show_coords.sh <data_type> [--dry-run]
source "$(dirname "${BASH_SOURCE[0]}")/_genome_dist_env.sh"
gd_init "${1:-}" mummer "${@:2}"
gd_need show-coords

for tgt in "$FASTA_DIR"/*.cleaned.fasta; do
  base="$(basename "$tgt" .cleaned.fasta)"
  delta="$OUTPUT_DIR/nucmer_aln_${DATA_TYPE}_${ref_base_name}_${base}.delta"
  out="$OUTPUT_DIR/2_show_coords_${DATA_TYPE}_${ref_base_name}_${base}.txt"
  [[ -f "$delta" ]] || { echo "[skip] missing $delta (run 1_nucmer_aln_genome.sh first)"; continue; }
  echo "[show-coords] $base"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "  [dry-run] show-coords -lcr $delta > $out"
  else
    show-coords -lcr "$delta" > "$out"
  fi
done
echo "[done] coords .txt files in $OUTPUT_DIR"
