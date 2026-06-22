#!/bin/bash
# ---------------------------------------------------------------------------
# Shared environment for the genome-distance pipeline (Figure 1C/1D).
# ---------------------------------------------------------------------------
# Sourced by the numbered step scripts in this directory (1_nucmer / 2_show_coords
# / 3_mummerplot and mash/ + dashing2/). Resolves every path through
# config/paths.yaml (no hardcoded filesystem fragments) and puts the MUMmer / mash
# tools on PATH.
#
#   inputs  <- datasets.lm_corpus_split_root      (the released, on-disk LM corpus
#              split: data_{r64,strains,saccharomycetales,fungi_1385}_gtf/fasta/*.cleaned.fasta)
#   outputs -> ${GD_OUTPUT_ROOT:-corpus_build_results_root}/ensembl_fungi_59/<data_type>/genome_dist/<tool>
#              (corpus_build_results_root is a legacy scratch root — override it in
#               config/paths.yaml, or export GD_OUTPUT_ROOT, for your machine)
#
# Knobs (environment):
#   GD_THREADS       nucmer -t value (default 8)
#   GD_OUTPUT_ROOT   override the results root (handy for testing / portability)
#
# Usage (from a step script):
#   source "$(dirname "${BASH_SOURCE[0]}")/_genome_dist_env.sh"
#   gd_init "<data_type>" "<tool>" "$@"        # data_type e.g. strains_gtf ; tool e.g. mummer
#   gd_need nucmer                              # ensure a tool is on PATH (appends conda base/bin)
#   for tgt in "$FASTA_DIR"/*.cleaned.fasta; do
#       base="$(basename "$tgt" .cleaned.fasta)"
#       gd_run some_command ... "$REF_FASTA" "$tgt"
#   done
# ---------------------------------------------------------------------------
set -euo pipefail

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

# Exported by gd_init:
DATA_TYPE="" ; TOOL="" ; DRY_RUN=0
LCS="" ; RESULTS_ROOT="" ; REF_FASTA="" ; ref_base_name="" ; FASTA_DIR="" ; OUTPUT_DIR=""
GD_THREADS="${GD_THREADS:-8}"

# MUMmer (nucmer/show-coords/mummerplot) + mash live in the conda *base* env on this
# cluster; activating yeast_ml drops base/bin from PATH. Append it (after yeast_ml so
# shorkie's python stays first) only if the requested tool is missing. Override per site.
gd_need() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    local base; base="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
    export PATH="$PATH:$base/bin"
  fi
  command -v "$tool" >/dev/null || { echo "ERROR: '$tool' not found on PATH" >&2; exit 127; }
}

gd_init() {
  DATA_TYPE="${1:?usage: <script> <data_type> [--dry-run]  (data_type e.g. strains_gtf)}"
  TOOL="${2:?gd_init needs a tool name (mummer|mash|dashing2)}"
  shift 2 || true
  for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY_RUN=1; done

  LCS="$(cfg datasets.lm_corpus_split_root)"
  RESULTS_ROOT="${GD_OUTPUT_ROOT:-$(cfg corpus_build_results_root)}"
  REF_FASTA="$LCS/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"
  ref_base_name="$(basename "$REF_FASTA" .cleaned.fasta)"
  FASTA_DIR="$LCS/data_${DATA_TYPE}/fasta"
  OUTPUT_DIR="$RESULTS_ROOT/ensembl_fungi_59/${DATA_TYPE}/genome_dist/${TOOL}"

  [[ -f "$REF_FASTA" ]] || { echo "ERROR: reference FASTA not found: $REF_FASTA" >&2; exit 1; }
  [[ -d "$FASTA_DIR"  ]] || { echo "ERROR: target FASTA dir not found: $FASTA_DIR" >&2; exit 1; }
  if [[ "$DRY_RUN" == 0 ]]; then mkdir -p "$OUTPUT_DIR"; fi

  echo "[genome_dist] data_type=$DATA_TYPE tool=$TOOL threads=$GD_THREADS dry_run=$DRY_RUN"
  echo "  REF_FASTA  = $REF_FASTA"
  echo "  FASTA_DIR  = $FASTA_DIR  ($(ls "$FASTA_DIR"/*.cleaned.fasta 2>/dev/null | wc -l) genomes)"
  echo "  OUTPUT_DIR = $OUTPUT_DIR"
}

# Run a command, or just echo it under --dry-run.
gd_run() {
  if [[ "$DRY_RUN" == 1 ]]; then echo "  [dry-run] $*"; else "$@"; fi
}
