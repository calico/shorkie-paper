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
#SBATCH --array=0-103

# Define datasets and times to plot
# datasets=('gene_exp_motif_chunk_0' 'gene_exp_motif_chunk_1')
# datasets=('gene_exp_motif_f0c0_test_RP')
datasets=('gene_exp_motif_chunk/gene_exp_motif_chunk_0' 'gene_exp_motif_chunk/gene_exp_motif_chunk_1' 'gene_exp_motif_chunk/gene_exp_motif_chunk_2' 'gene_exp_motif_chunk/gene_exp_motif_chunk_3' 'gene_exp_motif_chunk/gene_exp_motif_chunk_4' 'gene_exp_motif_chunk/gene_exp_motif_chunk_5' 'gene_exp_motif_chunk/gene_exp_motif_chunk_6' 'gene_exp_motif_chunk/gene_exp_motif_chunk_7' 'gene_exp_motif_chunk/gene_exp_motif_chunk_8' 'gene_exp_motif_chunk/gene_exp_motif_chunk_9' 'gene_exp_motif_chunk/gene_exp_motif_chunk_10' 'gene_exp_motif_chunk/gene_exp_motif_chunk_11' 'gene_exp_motif_chunk/gene_exp_motif_chunk_12')
times_to_plot=("T0" "T5" "T10" "T15" "T30" "T45" "T60" "T90")
# times_to_plot=("T0")

# Calculate the total number of combinations
num_datasets=${#datasets[@]}
num_times=${#times_to_plot[@]}
total_jobs=$((num_datasets * num_times))

# Check for invalid SLURM_ARRAY_TASK_ID
if [ $SLURM_ARRAY_TASK_ID -ge $total_jobs ]; then
  echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds total jobs=$total_jobs"
  exit 1
fi

# Calculate dataset and time index
dataset_index=$((SLURM_ARRAY_TASK_ID / num_times))
time_index=$((SLURM_ARRAY_TASK_ID % num_times))

# Get current dataset and time
dataset=${datasets[$dataset_index]}
time=${times_to_plot[$time_index]}

echo "num_datasets  : ${num_datasets}"
echo "num_times     : ${num_times}"
echo "total_jobs    : ${total_jobs}"
echo "dataset_index : ${dataset_index}"
echo "time_index    : ${time_index}"

# Run the Python script
python 1_save_tensor.py --exp_dir ${dataset} --time_label ${time}
