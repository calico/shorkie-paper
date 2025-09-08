#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=32G
#SBATCH --array=0-14

# Define datasets and times to plot
# datasets=('gene_exp_motif_test_RP/f0c0/part0/' 'gene_exp_motif_test_RP/f0c0/part1/' 'gene_exp_motif_test_RP/f0c0/part2/' 'gene_exp_motif_test_RP/f0c0/part3/' 'gene_exp_motif_test_RP/f0c0/part4/' 'gene_exp_motif_test_RP/f0c0/part5/' 'gene_exp_motif_test_RP/f0c0/part6/')
# datasets=('gene_exp_motif_test_TSS_select/f0c0/part0')
# dataset='gene_exp_motif_test_RP/f0c0/part0/'
# # datasets=('gene_exp_motif_chunk/gene_exp_motif_chunk_13' 'gene_exp_motif_chunk/gene_exp_motif_chunk_14' 'gene_exp_motif_chunk/gene_exp_motif_chunk_15' 'gene_exp_motif_chunk/gene_exp_motif_chunk_16' 'gene_exp_motif_chunk/gene_exp_motif_chunk_17' 'gene_exp_motif_chunk/gene_exp_motif_chunk_18' 'gene_exp_motif_chunk/gene_exp_motif_chunk_19' 'gene_exp_motif_chunk/gene_exp_motif_chunk_20' 'gene_exp_motif_chunk/gene_exp_motif_chunk_21' 'gene_exp_motif_chunk/gene_exp_motif_chunk_22' 'gene_exp_motif_chunk/gene_exp_motif_chunk_23' 'gene_exp_motif_chunk/gene_exp_motif_chunk_24' 'gene_exp_motif_chunk/gene_exp_motif_chunk_25' 'gene_exp_motif_chunk/gene_exp_motif_chunk_26' 'gene_exp_motif_chunk/gene_exp_motif_chunk_27' 'gene_exp_motif_chunk/gene_exp_motif_chunk_28' 'gene_exp_motif_chunk/gene_exp_motif_chunk_29')

# datasets=('gene_exp_motif_test_Proteasome/f0c0/part0' 'gene_exp_motif_test_Proteasome/f0c0/part1' 'gene_exp_motif_test_Proteasome/f0c0/part2' 'gene_exp_motif_test_Proteasome/f0c0/part3')

target_tf='SWI4'
dataset_base=gene_exp_motif_test_${target_tf}_targets/f0c0/part
# datasets=('gene_exp_motif_test_SWI4_targets/f0c0/part0' 'gene_exp_motif_test_Proteasome/f0c0/part1' 'gene_exp_motif_test_Proteasome/f0c0/part2' 'gene_exp_motif_test_Proteasome/f0c0/part3')

# # # Get current dataset and time
dataset=${dataset_base}${SLURM_ARRAY_TASK_ID}

# Run the Python script
python 3_plot_dna_logo_time_series.py --exp_dir ${dataset} --target_tf ${target_tf}
