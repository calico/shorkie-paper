#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --job-name=tfmodisco
#SBATCH --output=job_output_%A_%a.log
#SBATCH --mail-type=END
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-2

# Define arrays for species types and model architectures
# species_types=(ascomycota orbiliales)
species_types=(saccharomycetales_select)
model_archs=(unet_small_bert_drop unet_small_bert_drop_retry_1 unet_small_bert_drop_retry_2)

# Calculate indices based on SLURM_ARRAY_TASK_ID
# Total combinations = number of species_types * number of model_archs = 2*3 = 6
idx=$SLURM_ARRAY_TASK_ID
num_models=${#model_archs[@]}

# Determine species index and model index using integer division and modulo arithmetic
species_index=$(( idx / num_models ))
model_index=$(( idx % num_models ))

species=${species_types[$species_index]}
model=${model_archs[$model_index]}

echo "Processing species: $species, model: $model"

# Call the Python script with the selected parameters
python 4_viz_motif.py --species "$species" --model "$model"
