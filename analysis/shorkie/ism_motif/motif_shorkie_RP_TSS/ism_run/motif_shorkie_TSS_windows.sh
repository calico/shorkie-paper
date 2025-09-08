#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-40

SLURM_ARRAY_TASK_ID_IDX_offset=10
exp_type="test"
fold_param="f0c0"  # Dynamic fold_param based on array task ID
exp_data='TSS'
# Adjust SLURM_ARRAY_TASK_ID for the specific task
SLURM_ARRAY_TASK_ID_ADJ=$((SLURM_ARRAY_TASK_ID + SLURM_ARRAY_TASK_ID_IDX_offset))

output_dir="gene_exp_motif_${exp_type}_${exp_data}/${fold_param}/part${SLURM_ARRAY_TASK_ID_ADJ}/"

fine_tuned_model_type="exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"
current_experiment="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tuned_model_type}/16bp/train_prev/self_supervised_unet_small_bert_drop_v1/train/${fold_param}/train"


echo "Starting job on $(hostname) at $(date)"
echo "============================"
echo "Testing eQTL: ${current_experiment} "

# Create output directory
mkdir -p "$output_dir"

echo "$current_experiment/params.json"
echo "$current_experiment/model_best.h5"

# Run the command
python /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/baskerville-yeast/src/baskerville/scripts/hound_ism_bed.py \
    -f /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
    -o "$output_dir" \
    -p 8 \
    --rc \
    -l 500 \
    --stats logSED \
    -t /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tuned_model_type}/16bp/cleaned_sheet_RNA-Seq.txt \
    /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tuned_model_type}/16bp/self_supervised_unet_small_bert_drop_v1/train/${fold_param}/train/params.json \
    /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tuned_model_type}/16bp/self_supervised_unet_small_bert_drop_v1/train/${fold_param}/train/model_best.h5 \
    /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/TSS_chunk/TSS_windows_${SLURM_ARRAY_TASK_ID_ADJ}.bed \
    1>"$output_dir/gene_exp_motif_${exp_data}.out" \
    2>"$output_dir/gene_exp_motif_${exp_data}.err"

echo "============================"
echo "Job finished with exit code $? at: $(date)"