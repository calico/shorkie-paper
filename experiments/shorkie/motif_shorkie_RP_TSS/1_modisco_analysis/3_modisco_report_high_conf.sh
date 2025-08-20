#!/bin/sh

#SBATCH --job-name=2_modisco_script
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

# Define the model architectures
scores=('logSED')
dataset="gene_exp_motif_test_RP"

# Get the current model architecture based on the array inde
score=${scores[$SLURM_ARRAY_TASK_ID]}

mkdir -p "${dataset}/f0c0/${score}/"

# Define input file paths based on the model architecture
output_file="${dataset}/f0c0/${score}/modisco_results.h5"
report_dir="${dataset}/f0c0/${score}/report"

echo "output_file : ${output_file}"

# Run TF-MoDISco
modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m /home/kchao10/tools/motif_databases/YEAST/merged_meme_high_conf.meme