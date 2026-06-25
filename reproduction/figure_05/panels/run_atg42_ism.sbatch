#!/bin/sh
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=fig05_atg42_ism
#SBATCH --output=fig05_atg42_ism_%j.log
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=48G
#
# Figure 5 panels A/C: recompute the ATG42 (YBR139W) promoter ISM that was never
# released (the MSN2 ISM target set omits chrII:515 kb), so 5A/5C can show the REAL
# published locus instead of a representative surrogate.
#
# Faithful to scripts/04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/ism_run/
# motif_shorkie_targets.sh: SAME driver (hound_ism_bed.py), SAME fine-tuned model
# (self_supervised_unet_small_bert_drop fold f0c0), SAME flags (-l 500 --rc --stats
# logSED), SAME 3053-track RNA-seq target sheet, SAME cleaned R64 FASTA. The only
# change is a single-locus BED (panels/atg42.bed) = the 500 bp ATG42 promoter window
# chrII:515,214-515,714 (-450..+50 rel. YBR139W TSS @ 515,664, + strand). One model
# per process -> no cross-architecture restore collision.
#
# Output scores.h5 schema is byte-compatible with the released set: logSED (1,500,4,3053)
# f16, seqs (1,500,170) bool, chr/start/end/strand. Read unchanged by load_locus().
#
# Submit from repo root: sbatch reproduction/figure_05/panels/run_atg42_ism.sbatch
set -e
REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yeast_ml
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "host: $(hostname)  repo: $REPO"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible)"

cfg() { python -c "import sys; from shorkie import config; config.load(); print(config.path(sys.argv[1]))" "$1"; }
WR="$(cfg work_root)"
FASTA="$(cfg genome.fasta)"
MODEL_DIR="${WR}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f0c0/train"
SHEET="${WR}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt"
OUT="${REPO}/reproduction/figure_05/reproduced/ism_atg42"
BED="${REPO}/reproduction/figure_05/panels/atg42.bed"
DRIVER="${REPO}/external/baskerville-yeast/src/baskerville/scripts/hound_ism_bed.py"
mkdir -p "$OUT"

echo "WR=$WR"
echo "FASTA=$FASTA"
echo "MODEL_DIR=$MODEL_DIR"
echo "BED=$(cat "$BED")"

python "$DRIVER" \
    -f "$FASTA" \
    -o "$OUT" \
    -p 8 \
    --rc \
    -l 500 \
    --stats logSED \
    -t "$SHEET" \
    "$MODEL_DIR/params.json" \
    "$MODEL_DIR/model_best.h5" \
    "$BED"

echo "SBATCH_DONE exit=$?"
ls -la "$OUT"
