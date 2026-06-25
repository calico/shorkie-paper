#!/bin/sh
# Figure 4 — recompute the Shorkie_Random_Init ISM for the four 3-model panels
# (RPL26A, FUN12, KRE33, MMS2/MAD1 locus) on their EXACT published windows, using the
# learning-rate-5e-4 scratch checkpoint that the published figure's Random-Init row uses
# (pinned as Shorkie_Random_Init in scripts/04_analysis/shorkie_scratch/2_lr_search/
#  {2,3}_compare_lr_variants_*.py = supervised_unet_small_bert_drop_variants/learning_rate_0.0005).
# The released motif_random_init_RP_TSS ISM used the OLD lr=1e-4 model, so all four random
# rows are regenerated here.
#
# Submit from the repo (so env.sh resolves via the git root):
#   sbatch --chdir=<repo> reproduction/figure_04/recheck/run_fig4_random_ism.sh
# Output: <ISM_ROOT>/motif_random_init_RP_TSS/gene_exp_motif_test_fig4_lr5e4/f0c0/part0/scores.h5
#         (idx 0=RPL26A, 1=FUN12, 2=KRE33, 3=MMS2)
#SBATCH --job-name=fig4_random_lr5e4
#SBATCH --output=fig4_random_lr5e4_%j.log
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=a100
#SBATCH --account=ssalzbe1_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --export=ALL
echo "START $(date) host=$(hostname)"

# GPU TensorFlow env first (env.sh below calls `python` from shorkie) — see CLAUDE.md
source /home/kchao10/miniconda3/etc/profile.d/conda.sh
conda activate yeast_ml
echo "python: $(which python)"; python --version

source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"
echo "WORK_ROOT=$WORK_ROOT  BASKERVILLE_SCRIPTS=$BASKERVILLE_SCRIPTS"

fold_param="f0c0"
model_type="exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"
# Shorkie_Random_Init checkpoint = lr=5e-4 scratch variant (task=supervised, random init)
ckpt_dir="${WORK_ROOT}/seq_experiment/${model_type}/16bp/supervised_unet_small_bert_drop_variants/learning_rate_0.0005/train/${fold_param}/train"
out_dir="${WORK_ROOT}/experiments/SUM_data_process/motifs/motif_random_init_RP_TSS/gene_exp_motif_test_fig4_lr5e4/${fold_param}/part0"
mkdir -p "$out_dir"

echo "checkpoint: $ckpt_dir/model_best.h5"
echo "out_dir:    $out_dir"

python ${BASKERVILLE_SCRIPTS}/hound_ism_bed.py \
    -f ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
    -o "$out_dir" \
    -p 8 \
    --rc \
    -l 500 \
    --stats logSUM,logSED \
    -t ${WORK_ROOT}/seq_experiment/${model_type}/16bp/cleaned_sheet_RNA-Seq.txt \
    "$ckpt_dir/params.json" \
    "$ckpt_dir/model_best.h5" \
    ${WORK_ROOT}/data/gene_exp_ism_window/fig4_random_lr5e4/fig4_random_lr5e4.bed \
    1>"$out_dir/fig4_random.out" 2>"$out_dir/fig4_random.err"

echo "hound_ism_bed exit=$? at $(date)"
ls -la "$out_dir"
