#!/bin/sh

#SBATCH --job-name=2_modisco_script_all
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-7

# Define the arrays for experiment directories, target TFs, and time points.
# exp_dirs=("gene_exp_motif_test_TSS/f0c0" "gene_exp_motif_test_RP/f0c0")
exp_dirs=('gene_exp_motif_test_Proteasome/f0c0/')
# target_tfs=("MSN2_" "MSN4_")
# target_tfs=('RPN4_' 'MET4_' )
# target_tfs=('SWI4_' )
target_tfs=('RPN4_' )
# time_points=("T0" "T5" "T10" "T20" "T40" "T70" "T120" "T180")
time_points=("T0" "T5" "T10" "T15" "T30" "T45" "T60" "T90")

# Total combinations = 2 * 2 * 8 = 32.
# Use the SLURM_ARRAY_TASK_ID to compute the indices.
i=$SLURM_ARRAY_TASK_ID
time_index=$(( i % 8 ))    # 8 time points.
i=$(( i / 8 ))
target_tf_index=$(( i % 1 ))  # 2 target TFs.
i=$(( i / 1 ))
exp_dir_index=$(( i % 1 ))    # 2 exp_dirs.

exp_dir=${exp_dirs[$exp_dir_index]}
target_tf=${target_tfs[$target_tf_index]}
time_point=${time_points[$time_index]}

echo "Selected combination:"
echo "  exp_dir   : ${exp_dir}"
echo "  target_tf : ${target_tf}"
echo "  time point: ${time_point}"

# Build the base directory for the NPZ files.
# Expected structure: results/<exp_dir>/<target_tf>/<time_point>/
base_dir="results/${exp_dir}/${target_tf}/${time_point}/"
echo "Base directory: ${base_dir}"

# Check that the directory exists.
if [ ! -d "$base_dir" ]; then
    echo "Error: Directory $base_dir does not exist."
    exit 1
fi

# Define file paths.
x_true_file="${base_dir}ref.npz"
x_pred_file="${base_dir}pred.npz"

# Parameters for TF-MoDISco.
n=10000
w=500
output_file="${base_dir}modisco_results_${n}_${w}.h5"

echo "x_true_file : ${x_true_file}"
echo "x_pred_file : ${x_pred_file}"
echo "output_file : ${output_file}"

# Run TF-MoDISco for this combination.
modisco motifs -s "$x_true_file" -a "$x_pred_file" -n $n -o "$output_file" -w $w --verbose
