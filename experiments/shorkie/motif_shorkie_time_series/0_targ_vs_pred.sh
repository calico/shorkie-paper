#!/bin/bash

#SBATCH --job-name=model_eval
#SBATCH --output=job_gene_level_eval_%A_%a.log
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1-chess

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=64G
#SBATCH --array=0

# Define arrays for model architectures and learning types
model_archs=("unet_small_bert_drop")
# learning_types=("self_supervised" "supervised")
learning_types=("self_supervised")

idx_values=(0 1 2 3 4 5 6 7)
# idx_values=(0)
data_types=("RNA-Seq")
fine_tune_exp="exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"

# Compute sizes
num_models=${#model_archs[@]}
num_learning_types=${#learning_types[@]}
num_idx=${#idx_values[@]}
num_data_types=${#data_types[@]}

# Since --array=0-143, total is 144 = 3(models)*2(learning_types)*8(idx)*3(data_types)
# We want each task to select one combination of model_arch, learning_type, idx, and data_type

task_id=$SLURM_ARRAY_TASK_ID

# Compute indices
# data_type_idx: which data_type to pick
# model_idx:     which model_arch to pick
# learning_idx:  which learning_type to pick
# idx:           which index value to pick
data_type_idx=$((task_id / (num_models * num_learning_types * num_idx) % num_data_types))
model_idx=$((task_id / (num_learning_types * num_idx) % num_models))
learning_idx=$((task_id / num_idx % num_learning_types))
idx=${idx_values[$((task_id % num_idx))]}

# Extract specific combination
data_type=${data_types[$data_type_idx]}
model_arch=${model_archs[$model_idx]}
learning_type=${learning_types[$learning_idx]}

# Paths
root_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tune_exp}/16bp/${learning_type}_${model_arch}"

split=train
output_dir="${root_dir}/gene_target_preds_${split}"

# Create output directory
mkdir -p "${output_dir}/f${idx}c0/"

# Print task info
echo "============================"
echo "SLURM Task ID: $task_id"
echo "Data Type: $data_type"
echo "Model Architecture: $model_arch"
echo "Learning Type: $learning_type"
echo "Index: $idx"
echo "============================"

# rm -rf "${output_dir}/f${idx}c0/${data_type}/"

# --pseudo_qtl

# Run the Python script
echo python /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/baskerville-yeast/src/baskerville/scripts/hound_target_preds_gene_cmp.py \
    "${root_dir}/params.json" \
    "${root_dir}/train/f${idx}c0/train/model_best.h5" \
    "${root_dir}/train/f${idx}c0/data0/" \
    "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf" \
    --rc \
    --split ${split} \
    -t /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tune_exp}/16bp/cleaned_sheet.txt \
    --dataset_type "${data_type}" \
    --file_type gtf \
    --eval_dir "${root_dir}/train/f${idx}c0/data0/" --no_unclip \
    -o "${output_dir}/f${idx}c0/${data_type}/"

echo "============================"
        # --span \
