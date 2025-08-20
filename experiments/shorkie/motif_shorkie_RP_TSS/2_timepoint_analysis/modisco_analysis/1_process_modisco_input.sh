#!/bin/sh

#SBATCH --job-name=1_process_modisco_input
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0

# target_tf='SWI4'
# target_tf='MET4'
# target_tf='MSN2'
target_tf='MSN4'

dataset=gene_exp_motif_test_${target_tf}_targets/f0c0/

# Run the Python script
python 1_process_modisco_input.py --exp_dir "${dataset}" --target_tf "${target_tf}"
