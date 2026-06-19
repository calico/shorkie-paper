#!/bin/sh

#SBATCH --job-name=3_modisco_report
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mem=64G
#SBATCH --array=0

# Resolve machine paths from config (config/paths.yaml)
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
MOTIF_DB_DIR="$(cfg motif_db_dir)"

# model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
model_archs=('unet_small_bert_drop')
# datasets=('strains_select' 'schizosaccharomycetales')
# datasets=('orbiliales' 'saccharomycetales_select' 'ascomycota')
datasets=('saccharomycetales_select')

# Calculate indices for model and dataset based on array task ID
model_idx=$((SLURM_ARRAY_TASK_ID / ${#datasets[@]}))
dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#datasets[@]}))

model=${model_archs[$model_idx]}
dataset=${datasets[$dataset_idx]}
n_val=1000000
w_val=16384

# Define input file paths based on the model architecture
output_file="${dataset}_viz_seq/${model}/modisco_results_w_${w_val}_n_${n_val}.h5"
# report_dir="schizosaccharomycetales_viz_seq/${model}/report_merge/"
report_dir="${dataset}_viz_seq/${model}/report_w_${w_val}_n_${n_val}_high_conf/"

# Run TF-MoDISco report
modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m ${MOTIF_DB_DIR}/merged_meme_high_conf.meme