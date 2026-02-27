#!/bin/bash

#SBATCH --job-name=atten_map
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=64G
#SBATCH --array=0

# # Define the array of model types
# model_types=("fine-tune" "supervised")
# # model_types=("supervised")

# # Use the SLURM array index to pick the correct model type
# model_type=${model_types[$SLURM_ARRAY_TASK_ID]}

# # Create the output directory if it doesn't exist
# mkdir -p attention_viz/"$model_type"

# Define Defaults
ROOT_DIR=${ROOT_DIR_ENV:-"../.."}
OUT_DIR=${OUT_DIR_ENV:-"./results"}

echo "Using ROOT_DIR=$ROOT_DIR"
echo "Using OUT_DIR=$OUT_DIR"

# Run the visualization script
python 1_bin_level_freq_viz.py \
    --root_dir "$ROOT_DIR" \
    --out_dir "$OUT_DIR"
