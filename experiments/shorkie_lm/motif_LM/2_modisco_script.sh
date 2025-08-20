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
# n_values=(5000)            # Try 20000 and 5000 seqlets per metacluster
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

# Define input file paths based on the model architecture
x_true_file="saccharomycetales_viz_seq/${model}/x_true.npz"
x_pred_file="saccharomycetales_viz_seq/${model}/x_pred.npz"
# Include the parameter settings in the output filename for easy tracking.
output_file="saccharomycetales_viz_seq/${model}/modisco_results_w${window}_n${n_val}.h5"

# Print chosen parameters (useful for debugging)
echo "Running TF-MoDISco with model: ${model}, window size: ${window}, n value: ${n_val}"

# Run TF-MoDISco with these parameters
modisco motifs -s "$x_true_file" -a "$x_pred_file" -n "$n_val" -o "$output_file" -w "$window" --verbose
