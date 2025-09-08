#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-4

target_tf='MSN4'
dataset_base=../gene_exp_motif_test_${target_tf}_targets/f0c0/part

# # # Get current dataset and time
dataset=${dataset_base}${SLURM_ARRAY_TASK_ID}



# Run the Python script
python 0_timepoint_viz_scores_h5.py --exp_dir ${dataset} --target_tf ${target_tf}_
