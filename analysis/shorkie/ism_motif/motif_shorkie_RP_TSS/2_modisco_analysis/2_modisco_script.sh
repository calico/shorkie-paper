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
# dataset="gene_exp_motif_test_RP"
dataset="gene_exp_motif_test_TSS"

# Get the current model architecture based on the array inde
score=${scores[$SLURM_ARRAY_TASK_ID]}

mkdir -p "${dataset}/f0c0/${score}/"

n=10000
w=500
# Define input file paths based on the model architecture
x_true_file="${dataset}/f0c0/${score}/ref.npz"
x_pred_file="${dataset}/f0c0/${score}/pred.npz"
output_file="${dataset}/f0c0/${score}/modisco_results_${n}_${w}.h5"

echo "x_true_file : ${x_true_file}"
echo "x_pred_file : ${x_pred_file}"
echo "output_file : ${output_file}"

# Run TF-MoDISco
modisco motifs -s "$x_true_file" -a "$x_pred_file" -n $n -o "$output_file" -w $w --verbose
