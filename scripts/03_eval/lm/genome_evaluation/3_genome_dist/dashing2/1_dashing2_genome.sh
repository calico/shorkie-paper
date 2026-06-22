#!/bin/bash
# Figure 1D (alternative metric) — dashing2 genome-sketch similarity from the R64
# reference to every genome in <data_type>. Shared paths come from
# ../_genome_dist_env.sh; the dashing2 binary is tools.dashing2_bin (dashing2 is not
# on bioconda — build it and point the config key at your copy). See ../README.md.
#
#   bash dashing2/1_dashing2_genome.sh <data_type> [--dry-run]
source "$(dirname "${BASH_SOURCE[0]}")/../_genome_dist_env.sh"
gd_init "${1:-}" dashing2 "${@:2}"

DASHING2_BIN="$(cfg tools.dashing2_bin)"
[[ -n "$DASHING2_BIN" ]] || { echo "ERROR: tools.dashing2_bin not set in config/paths.yaml" >&2; exit 1; }

for tgt in "$FASTA_DIR"/*.cleaned.fasta; do
  base="$(basename "$tgt" .cleaned.fasta)"
  out="$OUTPUT_DIR/${DATA_TYPE}_${ref_base_name}_${base}.txt"
  echo "[dashing2] R64 vs $base"
  gd_run "$DASHING2_BIN" sketch "$REF_FASTA" "$tgt" --cmpout "$out"
done
echo "[done] dashing2 similarities in $OUTPUT_DIR"
