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
#SBATCH --array=0-1

vmin=0.001
vmax=0.05

# Define the array of model types
# model_types=("fine-tune" "supervised")
model_types=("supervised_unet_small_bert_drop" "self_supervised_unet_small_bert_drop" )

# Use the SLURM array index to pick the correct model type
model_type=${model_types[$SLURM_ARRAY_TASK_ID]}

# Create the output directory if it doesn't exist
mkdir -p attention_viz_vmin_${vmin}_vmax_${vmax}/"$model_type"

attention_offset=0
if [[ $model_type == "supervised_unet_small" || 
      $model_type == "supervised_unet_small_drop" || 
      $model_type == "self_supervised_unet_small" || 
      $model_type == "self_supervised_unet_small_drop" ]]; then
    attention_offset=67
elif [[ $model_type == "supervised_unet_small_bert_drop" || 
        $model_type == "self_supervised_unet_small_bert_drop" ]]; then
    attention_offset=74
fi


# Run the visualization script
python 1_visualize_attention.py \
    --out_dir attention_viz_vmin_${vmin}_vmax_${vmax}/"$model_type" \
    --model_type "$model_type" \
    --attention_offset ${attention_offset} \
    --seq_len 16384 \
    --n_reps 8 \
    --rc \
    --vmin ${vmin} \
    --vmax ${vmax} \
    > attention_viz_vmin_${vmin}_vmax_${vmax}/"$model_type"/log.out \
    2> attention_viz_vmin_${vmin}_vmax_${vmax}/"$model_type"/log.err
