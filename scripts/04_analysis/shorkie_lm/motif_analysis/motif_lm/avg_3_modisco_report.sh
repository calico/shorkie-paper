#!/bin/sh

#SBATCH --partition=parallel
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --job-name=tfmodisco
#SBATCH --output=job_output_%A_%a.log
#SBATCH --mail-type=end
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#SBATCH --array=0-1

# Resolve the external motif-database dir from config (config/paths.yaml: motif_db_dir)
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
MOTIF_DB_DIR="$(cfg motif_db_dir)"

# Define the model architectures
model_archs=('unet_small_bert_aux_drop' 'unet_small')
# model_archs=('unet_small')

# Get the current model architecture based on the array index
model=${model_archs[$SLURM_ARRAY_TASK_ID]}

# Define input file paths based on the model architecture
x_true_file="saccharomycetales_viz_seq/averaged_models/${model}/x_true.npz"
x_pred_file="saccharomycetales_viz_seq/averaged_models/${model}/x_pred.npz"
output_file="saccharomycetales_viz_seq/averaged_models/${model}/modisco_results.h5"
report_dir="saccharomycetales_viz_seq/averaged_models/${model}/report_merge/"

# Run TF-MoDISco
modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m "${MOTIF_DB_DIR}/merged_meme.meme"
