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
#SBATCH --mem=32G
#SBATCH --array=0

# Construct dataset directory based on the array task ID
# dataset="gene_exp_motif_test_SS/f0c0/part${SLURM_ARRAY_TASK_ID}/"
# dataset="gene_exp_motif_test_RP/f0c0/part${SLURM_ARRAY_TASK_ID}/"
dataset="gene_exp_motif_test_SS_snp/f0c0/part0/"
python 1_plot_dna_logo_general.py --exp_dir ${dataset}


python 1_plot_dna_logo_general.py --exp_dir gene_exp_motif_test_SS_snp/f0c0/part0/

