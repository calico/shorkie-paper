#!/bin/sh

#SBATCH --job-name=modisco_report_all
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-7

# # Define the arrays for experiment directories, target TFs, time points, and scores.
# exp_dirs=("gene_exp_motif_test_TSS/f0c0" "gene_exp_motif_test_RP/f0c0")
# target_tfs=("RPN4_" "MET4_")
# time_points=("T0" "T5" "T10" "T15" "T30" "T45" "T60" "T90")

# #SBATCH --array=0-15

# # Define the arrays for experiment directories, target TFs, and time points.
# exp_dirs=("gene_exp_motif_test_TSS/f0c0" "gene_exp_motif_test_RP/f0c0")
# # target_tfs=("MSN2_" "MSN4_")
# # target_tfs=('RPN4_' 'MET4_' )
# target_tfs=('SWI4_' )
# time_points=("T0" "T5" "T10" "T20" "T40" "T70" "T120" "T180")



# Define the arrays for experiment directories, target TFs, and time points.
# exp_dirs=("gene_exp_motif_test_TSS/f0c0" "gene_exp_motif_test_RP/f0c0")
exp_dirs=('gene_exp_motif_test_Proteasome/f0c0/')
# target_tfs=("MSN2_" "MSN4_")
# target_tfs=('RPN4_' 'MET4_' )
# target_tfs=('SWI4_' )
target_tfs=('RPN4_' )
# time_points=("T0" "T5" "T10" "T20" "T40" "T70" "T120" "T180")
time_points=("T0" "T5" "T10" "T15" "T30" "T45" "T60" "T90")

# #SBATCH --array=0-31

# # Define the arrays for experiment directories, target TFs, time points, and scores.
# exp_dirs=("gene_exp_motif_test_TSS/f0c0" "gene_exp_motif_test_RP/f0c0")
# target_tfs=("RPN4_" "MET4_")
# time_points=("T0" "T5" "T10" "T15" "T30" "T45" "T60" "T90")
# scores=('logSED')
# score=${scores[0]}  # Only one score in this example.

# Total combinations: 2 * 2 * 8 = 32.
# Decompose the SLURM_ARRAY_TASK_ID into the corresponding indices:
i=$SLURM_ARRAY_TASK_ID
time_index=$(( i % 8 ))       # 8 time points.
i=$(( i / 8 ))
target_tf_index=$(( i % 1 ))    # 2 target TFs.
i=$(( i / 1 ))
exp_dir_index=$(( i % 1 ))      # 2 exp_dirs.

exp_dir=${exp_dirs[$exp_dir_index]}
target_tf=${target_tfs[$target_tf_index]}
time_point=${time_points[$time_index]}

echo "Processing combination:"
echo "  exp_dir   : ${exp_dir}"
echo "  target_tf : ${target_tf}"
echo "  time point: ${time_point}"
echo "  score     : ${score}"

# Parameters for modisco report.
n=10000
w=500

# Construct the base directory based on the expected structure:
# results/<exp_dir>/<target_tf>/<time_point>/
base_dir="results/${exp_dir}/${target_tf}/${time_point}/"
output_file="${base_dir}modisco_results_${n}_${w}.h5"
report_dir="${base_dir}report"

echo "Output file : ${output_file}"
echo "Report directory : ${report_dir}"

# Create the report directory if it doesn't exist.
mkdir -p "${report_dir}"

# Run modisco report.
modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m /home/kchao10/tools/motif_databases/YEAST/merged_meme.meme
