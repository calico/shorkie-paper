#!/bin/bash
# Figure 1C — step 1/3: nucmer alignment of the R64 reference against every genome
# in <data_type> (writes one .delta per target). Shared paths/tools come from
# _genome_dist_env.sh. See README.md.
#
#   bash 1_nucmer_aln_genome.sh <data_type> [--dry-run]
#   e.g.  bash 1_nucmer_aln_genome.sh strains_gtf
source "$(dirname "${BASH_SOURCE[0]}")/_genome_dist_env.sh"
gd_init "${1:-}" mummer "${@:2}"
gd_need nucmer

for tgt in "$FASTA_DIR"/*.cleaned.fasta; do
  base="$(basename "$tgt" .cleaned.fasta)"
  pfx="$OUTPUT_DIR/nucmer_aln_${DATA_TYPE}_${ref_base_name}_${base}"
  echo "[nucmer] R64 vs $base"
  gd_run nucmer -t "$GD_THREADS" -p "$pfx" "$REF_FASTA" "$tgt"
done
echo "[done] nucmer .delta files in $OUTPUT_DIR"
