#!/bin/bash

#SBATCH --job-name=model_eval
#SBATCH --output=job_gene_level_eval_%A_%a.log
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1-chess
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0

experiments=("gene_level_eval_rc")

experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
model_arch="unet_small_bert_drop"

# Define Defaults
ROOT_DIR=${ROOT_DIR_ENV:-"../../.."}
OUT_DIR=${OUT_DIR_ENV:-"results"}

echo "Using ROOT_DIR=$ROOT_DIR"
echo "Using OUT_DIR=$OUT_DIR"

# Run the Python script
python 3_gene_level_score_dist_viz.py \
    --model_arch "$model_arch" \
    --experiment "$experiment" \
    --root_dir "$ROOT_DIR" \
    --out_dir "$OUT_DIR"

echo "============================"