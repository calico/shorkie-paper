#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LM genome-corpus build orchestrator
# ---------------------------------------------------------------------------
# Chains the 4 numbered stages in this directory to (re)build one LM-corpus
# tier end-to-end, from EnsemblFungi r59 genomes to ZLIB TFRecords:
#
#   1_data_download/      download + clean FASTA, download + split GTF
#   2_repeat_region_masking/  soft-mask repeats (RepeatModeler/Masker/DUST,
#                             OR download Ensembl's pre-soft-masked genomes)
#   3_data_filtering/     generate train/valid/test sequence BEDs + statistics.json
#   4_tf_data_generation/ write one-hot sequences to TFRecords
#
# Each stage script shares a uniform interface: --save_suffix <tier-suffix>
# --out_dir <split-root>. This driver fills those in from config/paths.yaml and
# the committed species lists in data/species_lists/.
#
# The PRACTICAL way to obtain a corpus is to download the prebuilt TFRecords:
#     data/download.sh --lm-corpus <tier>
# This script is for reproducing the build from raw genomes. Stage 2 (de-novo
# RepeatModeler/RepeatMasker) needs heavy external tools and is best replaced by
# downloading Ensembl's pre-masked (dna_sm) assemblies (the default below).
#
# Usage:
#   scripts/01_data_build/lm_corpus/run_pipeline.sh \
#       --tier {r64|strains|saccharomycetales|fungi} [--out-dir DIR]
#       [--stages 1,2,3,4] [--dry-run] [--verify]
#
#   --tier      which corpus tier to build (default: r64 — smallest, smoke test)
#   --out-dir   build root (default: config datasets.lm_corpus_split_root)
#   --stages    comma list of stages to run (default: 1,2,3,4)
#   --dry-run   print every command without executing
#   --verify    after building, diff the tier's statistics.json against the
#               committed expected values (does not run a build)
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"

TIER=r64; OUT_DIR=""; STAGES="1,2,3,4"; DRY_RUN=0; VERIFY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)     TIER="$2";    shift 2;;
    --out-dir)  OUT_DIR="$2"; shift 2;;
    --stages)   STAGES="$2";  shift 2;;
    --dry-run)  DRY_RUN=1;    shift;;
    --verify)   VERIFY=1;     shift;;
    -h|--help)  sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Read a dotted key from config/paths.yaml.
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

# Map the friendly --tier to the dataset save_suffix (the stage scripts read
# species<suffix>.cleaned.csv and write data<suffix>/).
case "$TIER" in
  r64)               SUFFIX="_r64_gtf";               EXP_SPECIES=1;    EXP_TRAIN=1201;   ;;
  strains)           SUFFIX="_strains_gtf";           EXP_SPECIES=80;   EXP_TRAIN=102315; ;;
  saccharomycetales) SUFFIX="_saccharomycetales_gtf"; EXP_SPECIES=165;  EXP_TRAIN=385551; ;;
  fungi)             SUFFIX="_fungi_1385_gtf";         EXP_SPECIES=1361; EXP_TRAIN=625355; ;;
  *) echo "unknown --tier '$TIER' (r64|strains|saccharomycetales|fungi)" >&2; exit 2;;
esac
EXP_TEST=528; EXP_VALID=518; SEQ_LEN=16384

[[ -n "$OUT_DIR" ]] || OUT_DIR="$(cfg datasets.lm_corpus_split_root)"
OUT_DIR="${OUT_DIR%/}"
SPECIES_CSV="$REPO_ROOT/data/species_lists/species${SUFFIX}.cleaned.csv"
DATA_DIR="$OUT_DIR/data${SUFFIX}"
STATS="$DATA_DIR/statistics.json"

run() {  # echo + (maybe) execute a command, honoring --dry-run
  echo "+ $*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

# ----- --verify: just compare statistics.json, no build -----
if [[ "$VERIFY" == 1 ]]; then
  echo "Verifying $STATS against expected (tier=$TIER)"
  [[ -f "$STATS" ]] || { echo "MISSING: $STATS — build it first" >&2; exit 1; }
  python - "$STATS" "$SEQ_LEN" "$EXP_TEST" "$EXP_VALID" "$EXP_TRAIN" "$EXP_SPECIES" <<'PY'
import json, sys
f, *exp = sys.argv[1:]
seq_len, test, valid, train, species = map(int, exp)
s = json.load(open(f))
want = {"seq_length": seq_len, "test_seqs": test, "valid_seqs": valid,
        "train_seqs": train, "num_species": species}
bad = {k: (s.get(k), v) for k, v in want.items() if s.get(k) != v}
if bad:
    print("MISMATCH (got, want):", bad); sys.exit(1)
print("OK — statistics.json matches expected:", want)
PY
  exit 0
fi

echo "=== LM corpus build: tier=$TIER suffix=$SUFFIX ==="
echo "    species list : $SPECIES_CSV ($EXP_SPECIES species)"
echo "    build root   : $OUT_DIR"
echo "    output tier  : $DATA_DIR"
[[ "$DRY_RUN" == 1 ]] && echo "    (dry run — commands printed, not executed)"
[[ -f "$SPECIES_CSV" ]] || { echo "MISSING species list: $SPECIES_CSV" >&2; exit 1; }

# Seed the committed species list into the build root under both the raw
# (species<suffix>.csv, read by the download stage) and cleaned
# (species<suffix>.cleaned.csv, read by the filter stage) names.
run mkdir -p "$OUT_DIR"
run cp "$SPECIES_CSV" "$OUT_DIR/species${SUFFIX}.cleaned.csv"
run cp "$SPECIES_CSV" "$OUT_DIR/species${SUFFIX}.csv"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# ----- Stage 1: download + clean FASTA, download + split GTF -----
if has_stage 1; then
  echo "--- Stage 1: data download (EnsemblFungi release-59) ---"
  ( run cd "$HERE/1_data_download"
    run python 1_download_fasta.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/"
    run python 2_download_gtf.py   --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/"
    run python 3_clean_fasta.py    --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" \
        --assembly_level chromosome --min_length 32768
    run python 4_split_gtf.py      --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" )
fi

# ----- Stage 2: repeat soft-masking -----
if has_stage 2; then
  echo "--- Stage 2: repeat soft-masking ---"
  echo "    DEFAULT: download Ensembl's pre-soft-masked (dna_sm) assemblies."
  echo "    From-scratch alternative (heavy external tools), per genome:"
  echo "      0_rerun_repeatModeler.sh  # BuildDatabase + RepeatModeler -LTRStruct"
  echo "      1_repeatMasker.sh         # RepeatMasker -xsmall -e rmblast"
  echo "      2_rerun_dust.sh           # DUST low-complexity soft-mask"
  ( run cd "$HERE/2_repeat_region_masking"
    run python 3_download_masked_fasta.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" )
fi

# ----- Stage 3: filtering -> sequence BEDs + statistics.json -----
if has_stage 3; then
  echo "--- Stage 3: data filtering (chr-holdout split + biotype/repeat thresholds) ---"
  echo "    split: valid=R64 chrXI/XIII/XV, test=R64 chrXII/XIV/XVI, chrXI-XVI excluded from train"
  echo "    thresholds: seq_len 16384, stride 4096, record_size 32, max_rm_frac 0.001,"
  echo "                contig_pad 512; exon caps ''=0.925 protein_coding=0.875 rRNA/tRNA=0.03125"
  ( run cd "$HERE/3_data_filtering"
    run python 1_generate_sequences_bed.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" )
fi

# ----- Stage 4: TFRecord generation -----
if has_stage 4; then
  echo "--- Stage 4: TFRecord generation (ZLIB) ---"
  echo "    NOTE: 1_write_data_multi.py ships with its label loop set to ['test'] only."
  echo "          To build all splits, edit it to loop ['train','valid','test'] (line ~68)."
  ( run cd "$HERE/4_tf_data_generation"
    run python 1_write_data_multi.py --save_suffix "$SUFFIX" --out_dir "$OUT_DIR/" \
        --run_local --processes 8 --use_gtf )
fi

echo "=== done. Verify with: $0 --tier $TIER --verify ==="
