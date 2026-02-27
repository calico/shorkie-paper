#!/bin/bash

#SBATCH --job-name=2_modisco_script_all
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-15

# Define the target transcription factors.
# target_tfs=('SWI4' 'MET4')
target_tfs=('MSN2' 'MSN4')

# Calculate the correct indices for TF and time point.
tf_index=$(( SLURM_ARRAY_TASK_ID / 8 ))
time_index=$(( SLURM_ARRAY_TASK_ID % 8 ))

target_tf=${target_tfs[$tf_index]}

# Define time points based on the target TF.
if [ "$target_tf" = "SWI4" ]; then
    time_points=("T0" "T5" "T10" "T20" "T40" "T70" "T120" "T180")
# MET4 or RPN4.
elif [ "$target_tf" = "MET4" ] || [ "$target_tf" = "RPN4" ] || [ "$target_tf" = "MSN2" ] || [ "$target_tf" = "MSN4" ]; then
    time_points=("T0" "T5" "T10" "T15" "T30" "T45" "T60" "T90")
fi

time_point=${time_points[$time_index]}

# Construct the dataset directory based on the target TF.
dataset="gene_exp_motif_test_${target_tf}_targets/f0c0"

# Build the base directory for the NPZ files.
# Expected structure: results/<dataset>/<target_tf>/<time_point>/
base_dir="results/${dataset}/${target_tf}/${time_point}/"
echo "Base directory: ${base_dir}"

# Check that the directory exists.
if [ ! -d "$base_dir" ]; then
    echo "Error: Directory $base_dir does not exist."
    exit 1
fi

# Define file paths.
x_true_file="${base_dir}ref_diff.npz"
x_pred_file="${base_dir}pred_diff.npz"

# Parameters for TF-MoDISco.
n=10000
w=500
output_file="${base_dir}modisco_results_${n}_${w}_diff.h5"

echo "x_true_file : ${x_true_file}"
echo "x_pred_file : ${x_pred_file}"
echo "output_file : ${output_file}"

# Run TF-MoDISco for this combination.
modisco motifs -s "$x_true_file" -a "$x_pred_file" -n $n -o "$output_file" -w $w --verbose
