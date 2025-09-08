#!/bin/sh

#SBATCH --job-name=2_modisco_script
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0

# Define the model architectures
scores=('logSED')
dataset="results/"
# dataset="gene_exp_motif_test_TSS"

# Get the current model architecture based on the array inde
score=${scores[$SLURM_ARRAY_TASK_ID]}

n=1000000
w=80
# Define input file paths based on the model architecture
output_file="${dataset}/modisco_results_${n}_${w}.h5"
report_dir="${dataset}/report"

echo "output_file : ${output_file}"

# Run TF-MoDISco
modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m /home/kchao10/tools/motif_databases/YEAST/merged_meme.meme