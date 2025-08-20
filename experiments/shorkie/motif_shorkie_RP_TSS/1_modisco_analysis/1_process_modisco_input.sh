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
#SBATCH --mem=32G
#SBATCH --array=0

# Get current dataset and time
# dataset=gene_exp_motif_test_RP/f0c0/
dataset=gene_exp_motif_test_TSS/f0c0/


echo "dataset : ${dataset}"
mkdir -p ${dataset} 

# Run the Python script
python 1_process_modisco_input.py --exp_dir ${dataset}