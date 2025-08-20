#!/bin/sh

#SBATCH --job-name=2_modisco_script
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=64G
#SBATCH --array=0-8

model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
# datasets=('strains_select' 'schizosaccharomycetales')
datasets=('orbiliales' 'saccharomycetales_select' 'ascomycota')

# Calculate indices for model and dataset based on array task ID
model_idx=$((SLURM_ARRAY_TASK_ID / ${#datasets[@]}))
dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#datasets[@]}))

model=${model_archs[$model_idx]}
dataset=${datasets[$dataset_idx]}

# Define input file paths based on the model architecture
x_true_file="${dataset}_viz_seq/${model}/x_true.npz"
x_pred_file="${dataset}_viz_seq/${model}/x_pred.npz"
output_file="${dataset}_viz_seq/${model}/modisco_results_w_16384_n_1000000.h5"
# report_dir="${dataset}_viz_seq/${model}/report_merge/"

# Run TF-MoDISco
modisco motifs -s "$x_true_file" -a "$x_pred_file" -n 1000000 -o "$output_file" -w 16384 --verbose

# modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m /home/kchao10/tools/motif_databases/YEAST/merged_meme.meme

# modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" #-m /home/kchao10/tools/motif_databases/YEAST/YEASTRACT_20130918.meme