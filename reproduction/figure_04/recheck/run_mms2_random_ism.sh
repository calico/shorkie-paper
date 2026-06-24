#!/bin/sh
# Figure 4 panel F — recompute Random-Init (scratch model) ISM on the EXACT published
# MMS2-panel window chrVII:346,669-347,169 (== Shorkie-ISM gene_exp_motif_test_TSS
# part33 idx31; BED-labeled MAD1 — MMS2/MAD1 are adjacent divergent genes sharing this
# locus). The released random-init ISM never covered this window, so panel F's third
# (Random-Init) row is recomputed here with the SAME scratch checkpoint + flags the other
# random-init entries used (motif_shorkie_random_init_TSS_windows_select.sh).
#
# Submit:  sbatch reproduction/figure_04/recheck/run_mms2_random_ism.sh
# Output:  <ISM_ROOT>/motif_random_init_RP_TSS/gene_exp_motif_test_MMS2_panel/f0c0/part0/scores.h5
#          (loaded by fig4_common.ism_saliency; then cached to recheck/mms2_random_cache.npz)
#SBATCH --job-name=mms2_random_ism
#SBATCH --output=mms2_random_ism_%j.log
#SBATCH --time=04:00:00
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

# Submit from the repo (e.g. `sbatch --chdir=<repo> reproduction/figure_04/recheck/run_mms2_random_ism.sh`)
# so the cwd is inside the git tree; env.sh is then resolved via the repo root.
source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"
echo "WORK_ROOT=$WORK_ROOT  BASKERVILLE_SCRIPTS=$BASKERVILLE_SCRIPTS"

fold_param="f0c0"
model_type="exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"
# scratch / random-init checkpoint — identical to the released random-init ISM entries
ckpt_dir="${WORK_ROOT}/seq_experiment/${model_type}/16bp/supervised_unet_small_bert_drop/train/${fold_param}/train"
out_dir="${WORK_ROOT}/experiments/SUM_data_process/motifs/motif_random_init_RP_TSS/gene_exp_motif_test_MMS2_panel/${fold_param}/part0"
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
    ${WORK_ROOT}/data/gene_exp_ism_window/MMS2_panel/MMS2_panel.bed \
    1>"$out_dir/mms2_random.out" 2>"$out_dir/mms2_random.err"

echo "hound_ism_bed exit=$? at $(date)"
ls -la "$out_dir"
