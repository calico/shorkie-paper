#!/bin/bash
# Figure 1D — mash distance from the R64 reference to every genome in <data_type>
# (one row per target: ref query distance p-value shared-hashes). Shared paths/tools
# come from ../_genome_dist_env.sh. See ../README.md.
#
#   bash mash/1_mash_genome.sh <data_type> [--dry-run]
#   e.g.  bash mash/1_mash_genome.sh strains_gtf
source "$(dirname "${BASH_SOURCE[0]}")/../_genome_dist_env.sh"
gd_init "${1:-}" mash "${@:2}"
gd_need mash

for tgt in "$FASTA_DIR"/*.cleaned.fasta; do
  base="$(basename "$tgt" .cleaned.fasta)"
  out="$OUTPUT_DIR/${DATA_TYPE}_${ref_base_name}_${base}.txt"
  echo "[mash] R64 vs $base"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "  [dry-run] mash dist $REF_FASTA $tgt > $out"
  else
    mash dist "$REF_FASTA" "$tgt" > "$out"
  fi
done
echo "[done] mash distances in $OUTPUT_DIR"
