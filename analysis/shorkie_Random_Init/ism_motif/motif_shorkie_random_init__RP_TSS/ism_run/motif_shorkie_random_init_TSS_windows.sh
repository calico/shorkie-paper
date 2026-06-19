#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --array=0-5
source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"

exp_type="test"
fold_param="f0c0"  # Dynamic fold_param based on array task ID
exp_data='TSS'
output_dir="gene_exp_motif_${exp_type}_${exp_data}/${fold_param}/part${SLURM_ARRAY_TASK_ID}/"

fine_tuned_model_type="exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"
current_experiment="${WORK_ROOT}/seq_experiment/${fine_tuned_model_type}/16bp/train_prev/supervised_unet_small_bert_drop/train/${fold_param}/train"


echo "Starting job on $(hostname) at $(date)"
echo "============================"
echo "Testing eQTL: ${current_experiment} "

# Create output directory
mkdir -p "$output_dir"

echo "$current_experiment/params.json"
echo "$current_experiment/model_best.h5"

# Run the command
python ${BASKERVILLE_SCRIPTS}/hound_ism_bed.py \
    -f ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
    -o "$output_dir" \
    -p 8 \
    --rc \
    -l 500 \
    --stats SUM,logSUM,logSED \
    -t ${WORK_ROOT}/seq_experiment/${fine_tuned_model_type}/16bp/cleaned_sheet_RNA-Seq.txt \
    ${WORK_ROOT}/seq_experiment/${fine_tuned_model_type}/16bp/supervised_unet_small_bert_drop/train/${fold_param}/train/params.json \
    ${WORK_ROOT}/seq_experiment/${fine_tuned_model_type}/16bp/supervised_unet_small_bert_drop/train/${fold_param}/train/model_best.h5 \
    ${WORK_ROOT}/data/gene_exp_ism_window/TSS_chunk/TSS_windows_0${SLURM_ARRAY_TASK_ID}.bed \
    1>"$output_dir/gene_exp_motif_${exp_data}.out" \
    2>"$output_dir/gene_exp_motif_${exp_data}.err"

echo "============================"
echo "Job finished with exit code $? at: $(date)"