#!/bin/bash

#SBATCH --job-name=atten_map
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=a100
#SBATCH -A ssalzbe1_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=64G
#SBATCH --array=0

vmin=0.001
vmax=0.05

# Define the model architectures
model_archs=('unet_small_bert_drop')

# Get the current model architecture based on the array index
model=${model_archs[$SLURM_ARRAY_TASK_ID]}

# Create the output directory if it doesn't exist
mkdir -p attention_viz_vmin_${vmin}_vmax_${vmax}/LM_${model}

# Run the visualization script
python 1_visualize_attention.py \
    --out_dir attention_viz_vmin_${vmin}_vmax_${vmax}/LM_${model} \
    --model_type "LM_${model}" \
    --LM_exp ${model} \
    --attention_offset 74 \
    --seq_len 16384 \
    --n_reps 1 \
    --rc \
    --vmin ${vmin} \
    --vmax ${vmax} \
    > attention_viz_vmin_${vmin}_vmax_${vmax}/LM_${model}/log.out \
    2> attention_viz_vmin_${vmin}_vmax_${vmax}/LM_${model}/log.err
