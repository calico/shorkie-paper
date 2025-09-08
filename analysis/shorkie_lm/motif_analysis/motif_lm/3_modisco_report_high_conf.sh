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
#SBATCH --mem=32G
#SBATCH --array=0-2

# Define arrays for the parameter combinations
model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
window_sizes=(16384)      # Try full-length (16k) and a cropped window (2k)
# n_values=(20000)            # Try 20000 and 5000 seqlets per metacluster
n_values=(100000)            # Try 20000 and 5000 seqlets per metacluster

# Calculate the total numbers
num_models=${#model_archs[@]}
num_windows=${#window_sizes[@]}
num_n=${#n_values[@]}

# Decode SLURM_ARRAY_TASK_ID into individual indices
# Total number of combinations = num_models * num_windows * num_n
index=$SLURM_ARRAY_TASK_ID

# The model index is determined by integer division by (num_windows * num_n)
model_idx=$(( index / (num_windows * num_n) ))
# The remainder will be split into window and n indices.
remainder=$(( index % (num_windows * num_n) ))
window_idx=$(( remainder / num_n ))
n_idx=$(( remainder % num_n ))

# Get the parameters from the arrays
model=${model_archs[$model_idx]}
window=${window_sizes[$window_idx]}
n_val=${n_values[$n_idx]}

# Include the parameter settings in the output filename for easy tracking.
output_file="saccharomycetales_viz_seq/${model}/modisco_results_w${window}_n${n_val}.h5"
report_dir="saccharomycetales_viz_seq/${model}/report_w${window}_n${n_val}_high_conf/"

# Print chosen parameters (useful for debugging)
echo "Running TF-MoDISco report with model: ${model}, window size: ${window}, n value: ${n_val}"

modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m /home/kchao10/tools/motif_databases/YEAST/merged_meme_high_conf.meme