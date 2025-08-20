#!/bin/sh

#SBATCH --partition=parallel
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --job-name=tfmodisco
#SBATCH --output=job_output_%A_%a.log
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#SBATCH --array=0-1

# Define the species list
species_ls=('schizosaccharomycetales' 'strains_select')

# Get the current species based on the array index
if [ $SLURM_ARRAY_TASK_ID -lt ${#species_ls[@]} ]; then
    species=${species_ls[$SLURM_ARRAY_TASK_ID]}
else
    echo "Error: SLURM_ARRAY_TASK_ID out of bounds. Exiting."
    exit 1
fi

# Define the model architecture (single model in this case)
model="unet_small_bert_aux_drop"

# Define input file paths based on the species
x_true_file="${species}_viz_seq/averaged_models/${model}/x_true.npz"
x_pred_file="${species}_viz_seq/averaged_models/${model}/x_pred.npz"
output_file="${species}_viz_seq/averaged_models/${model}/modisco_results.h5"
report_dir="${species}_viz_seq/averaged_models/${model}/report_merge/"

# Run TF-MoDISco
modisco motifs -s "$x_true_file" -a "$x_pred_file" -n 20000 -o "$output_file" -w 16384 --verbose