#!/bin/bash

#SBATCH --job-name=2_yeast_rna_seq_models
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1-chess
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-3

# Define an array of dataset types corresponding to the different tasks.
dataset_types=(RNA-seq 1000-RNA-seq Chip-MNase Chip-exo)

# The time group is common to all tasks.
time_group=T0

# Run the Python script with the dataset type based on the array index.
python 2_yeast_rna_seq_models.py --dataset_type ${dataset_types[$SLURM_ARRAY_TASK_ID]} --time_group $time_group
