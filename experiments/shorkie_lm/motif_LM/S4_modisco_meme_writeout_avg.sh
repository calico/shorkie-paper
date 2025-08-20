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

# Define the model architectures
model_archs=('unet_small_bert_aux_drop' 'unet_small')
# model_archs=('unet_small')

# Get the current model architecture based on the array index
model=${model_archs[$SLURM_ARRAY_TASK_ID]}

score='CWM-PFM'
# Define input file paths based on the model architecture
input_file="saccharomycetales_viz_seq/averaged_models/${model}/modisco_results.h5"
output_file="./saccharomycetales_viz_seq/averaged_models/${model}/report_merge/${score}"

# Run TF-MoDISco
modisco meme -i ${input_file} -o "$output_file" -t ${score}