#!/bin/sh

#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0

# Define the model architectures
# model_archs=('unet_small' 'unet_small_retry_1' 'unet_small_retry_2' 'unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')

model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')

# Get the current model architecture based on the array index
model=${model_archs[$SLURM_ARRAY_TASK_ID]}

python 1_search_motif.py $model
