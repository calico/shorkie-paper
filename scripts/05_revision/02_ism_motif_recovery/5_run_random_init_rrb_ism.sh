#!/bin/bash
# Revision 02 step 5 — run Shorkie_Random_Init ISM over the RRB promoters, so the RRPE
# counter-example can be settled at the panels where RRPE is actually the headline motif.
#
# RRPE and PAC are the headline motifs of Figure 4B-C (FUN12, KRE33), which are RRB-target
# promoters. Steps 1-4 can only pair windows both models were run on, and the released ISM
# cache has 16 RRB parts for Shorkie and ZERO for Shorkie_Random_Init — so on the current
# data the answer is "unresolvable at n=48 RRPE sites in the RP/TSS windows", which is an
# honest answer, but not one that settles RRPE at those panels.
#
# This run fills that gap. It mirrors the published Shorkie invocation
# (scripts/04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/ism_run/motif_shorkie_targets.sh
# with exp_data="RRB_targets", array 0-15) exactly — same FASTA, same 500 bp windows, same
# --rc, same --stats logSED, same RNA-seq targets sheet — changing only the model.
#
# Output lands in the canonical ISM tree beside the Shorkie data, so
# 1_extract_ism_saliency.py picks it up with no arguments: it already lists
# gene_exp_motif_test_RRB_targets in SUBSETS and pairs windows by genomic coordinate.
#
# Usage:
#   scripts/common/submit.sh --profile gpu --array 0-15 5_run_random_init_rrb_ism.sh
#   SLURM_ARRAY_TASK_ID=0 bash 5_run_random_init_rrb_ism.sh --dry-run
#
# After it completes, re-run steps 1-4; RRPE and PAC then appear with paired RRB sites.
set -euo pipefail

#SBATCH --job-name=rrb_ism_random_init
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

DRY_RUN=0
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY_RUN=1; done

TASK="${SLURM_ARRAY_TASK_ID:-0}"
EXP_DATA="RRB_targets"
FOLD="f0c0"                       # the released ISM cache is fold 0 only, for both models

ISM_ROOT="${ISM_ROOT:-$(cfg results.ism_scores)}"
MODEL_DIR="$(cfg models.shorkie_random_init)"
SUP_ROOT="$(cfg datasets.supervised_root)"
GENOME_FASTA="$(cfg genome.fasta)"
WINDOW_BED="${WORK_ROOT}/data/gene_exp_ism_window/${EXP_DATA}_chunk/${EXP_DATA}_windows_$(printf '%02d' "${TASK}").bed"

# Written beside the Shorkie tree so step 1 finds it without a path override.
OUT_DIR="${ISM_ROOT}/motif_random_init_RP_TSS/gene_exp_motif_test_${EXP_DATA}/${FOLD}/part${TASK}"

if [[ ! -f "$WINDOW_BED" ]]; then
  echo "error: window BED not found: $WINDOW_BED" >&2
  exit 1
fi

CMD=(python "${BASKERVILLE_SCRIPTS}/hound_ism_bed.py"
  -f "${GENOME_FASTA}"
  -o "${OUT_DIR}"
  -p 8
  --rc
  -l 500
  --stats logSED
  -t "${SUP_ROOT}/cleaned_sheet_RNA-Seq.txt"
  "${MODEL_DIR}/params.json"
  "${MODEL_DIR}/train/${FOLD}/train/model_best.h5"
  "${WINDOW_BED}")

echo "task=${TASK} exp_data=${EXP_DATA}"
echo "windows : ${WINDOW_BED}"
echo "output  : ${OUT_DIR}"
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "${OUT_DIR}"
"${CMD[@]}" 1>"${OUT_DIR}/gene_exp_motif_${EXP_DATA}.out" \
            2>"${OUT_DIR}/gene_exp_motif_${EXP_DATA}.err"
echo "=== done. Re-run steps 1-4 to include the RRB windows in the paired comparison. ==="
