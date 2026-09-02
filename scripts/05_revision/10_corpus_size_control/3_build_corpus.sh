#!/bin/bash
# Revision 10 step 3 — build the 165-genome fungal control corpus, window-matched to
# 165_Saccharomycetales.
#
# The four corpus-build stages share a uniform --save_suffix / --out_dir interface (see
# scripts/01_data_build/lm_corpus/run_pipeline.sh), so this drives them directly with a new
# tier suffix rather than modifying that orchestrator's hardcoded tier table.
#
# Stages 1-3 build the sequence BEDs, then the training windows are subsampled to exactly
# 385,551 (the 165_Saccharomycetales count) BEFORE stage 4 turns them into TFRecords —
# after that the count is baked in. The valid and test splits are left untouched: they come
# from S. cerevisiae R64 only and are identical across every tier, which is what makes
# perplexity comparable to the published Figure 1G.
#
# Usage:
#   python 2_sample_matched_corpus.py --write_species_list      # once, first
#   scripts/common/submit.sh --profile cpu 3_build_corpus.sh    # or run directly
#   bash 3_build_corpus.sh --dry-run                            # print every command
#
# Prerequisites: the corpus build downloads genomes from EnsemblFungi release-59 and needs
# network access; stage 2 downloads Ensembl's pre-soft-masked (dna_sm) assemblies rather
# than running RepeatModeler.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM="$REPO_ROOT/scripts/01_data_build/lm_corpus"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

SUFFIX="_fungi165_matched_gtf"
TARGET_TRAIN=385551
OUT_DIR="${OUT_DIR:-$(cfg datasets.lm_corpus_split_root)}"
DATA_DIR="$OUT_DIR/data${SUFFIX}"
SPECIES_CSV="$REPO_ROOT/data/species_lists/species${SUFFIX}.cleaned.csv"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
run() { echo "+ $*"; [[ "$DRY_RUN" == 1 ]] || "$@"; }

if [[ ! -f "$SPECIES_CSV" ]]; then
  echo "MISSING species list: $SPECIES_CSV" >&2
  echo "  run: python $HERE/2_sample_matched_corpus.py --write_species_list" >&2
  exit 1
fi
echo "=== control corpus build ==="
echo "    species list : $SPECIES_CSV"
echo "    build root   : $OUT_DIR"
echo "    output tier  : $DATA_DIR"

run mkdir -p "$OUT_DIR"
run cp "$SPECIES_CSV" "$OUT_DIR/species${SUFFIX}.cleaned.csv"
run cp "$SPECIES_CSV" "$OUT_DIR/species${SUFFIX}.csv"

echo "--- Stage 1: download + clean FASTA, download + split GTF ---"
( run cd "$LM/1_data_download"
  run python 1_download_fasta.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/"
  run python 2_download_gtf.py   --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/"
  run python 3_clean_fasta.py    --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" \
      --assembly_level chromosome --min_length 32768
  run python 4_split_gtf.py      --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" )

echo "--- Stage 2: repeat soft-masking (Ensembl dna_sm) ---"
( run cd "$LM/2_repeat_region_masking"
  run python 3_download_masked_fasta.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" )

echo "--- Stage 3: filtering -> sequence BEDs + statistics.json ---"
( run cd "$LM/3_data_filtering"
  run python 1_generate_sequences_bed.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" )

echo "--- Window matching: subsample training windows to ${TARGET_TRAIN} ---"
# sequences_train.cleaned.bed is the operative file — its line count is what
# statistics.json reports as train_seqs and what the TFRecord stage consumes.
run python "$HERE/3_subsample_windows.py" \
    --bed "$DATA_DIR/sequences_train.cleaned.bed" --target "$TARGET_TRAIN"

echo "--- Stage 4: TFRecord generation (ZLIB) ---"
echo "    NOTE: 1_write_data_multi.py ships with its label loop set to ['test'] only;"
echo "          build all splits before training (see run_pipeline.sh stage 4)."
( run cd "$LM/4_tf_data_generation"
  run python 1_write_data_multi.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" \
      --run_local --processes 8 --use_gtf )

echo "=== done. Tier built at $DATA_DIR ==="
echo "    Next: 4_train_matched_lm.sh"
