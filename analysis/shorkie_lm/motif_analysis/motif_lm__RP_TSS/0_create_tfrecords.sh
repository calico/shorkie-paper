#!/bin/sh

#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --job-name=eval_bed
#SBATCH --output=job_output_%A_%a.log
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#SBATCH --array=0-5

# ------------------------------------------------------------------------------
# 1) Define your arrays
# ------------------------------------------------------------------------------
experiments=(
    "lm_saccharomycetales_gtf_unet_small_bert_drop"
    "lm_saccharomycetales_gtf_unet_small_bert_drop_retry_1"
    "lm_saccharomycetales_gtf_unet_small_bert_drop_retry_2"
)

# datasets=("RP" "TSS")

datasets=("RP")

# ------------------------------------------------------------------------------
# 2) Determine which experiment/dataset combo for this array index
#    - experiment_index = floor(SLURM_ARRAY_TASK_ID / 2)
#    - dataset_index    = SLURM_ARRAY_TASK_ID % 2
# ------------------------------------------------------------------------------
experiment_index=$((SLURM_ARRAY_TASK_ID / 2))
dataset_index=$((SLURM_ARRAY_TASK_ID % 2))

current_experiment="${experiments[$experiment_index]}"
current_dataset="${datasets[$dataset_index]}"

# ------------------------------------------------------------------------------
# 3) Print some status info
# ------------------------------------------------------------------------------
echo "Starting job on $(hostname) at $(date)"
echo "========================================"
echo "Job index:           ${SLURM_ARRAY_TASK_ID}"
echo "Experiment index:    ${experiment_index}"
echo "Dataset index:       ${dataset_index}"
echo "Experiment:          ${current_experiment}"
echo "Dataset:             ${current_dataset}"

# ------------------------------------------------------------------------------
# 4) Make output directory
# ------------------------------------------------------------------------------
mkdir -p "${current_experiment}/"

# ------------------------------------------------------------------------------
# 5) Choose the BED file based on the dataset
# ------------------------------------------------------------------------------
if [ "${current_dataset}" = "RP" ]; then
    bedfile="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/RP_windows.bed"
elif [ "${current_dataset}" = "TSS" ]; then
    bedfile="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/TSS_windows_ex_chrmt.bed"
else
    echo "Error: Unknown dataset: ${current_dataset}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 6) Run your Python script
# ------------------------------------------------------------------------------
python 1_write_data_with_gtf.py \
    --bed "${bedfile}" \
    --fasta /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
    --gtf /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf \
    --save_suffix "${current_experiment}" \
    --start 0 \
    --end 16384 \
    --part 0 \
    --seq_length 16384 \
    --seq_stride 4096 \
    --record_size 6500 \
    --replace_n \
    --label "${current_dataset}" \
    --out_dir ./

# ------------------------------------------------------------------------------
# 7) Final status
# ------------------------------------------------------------------------------
echo "========================================"
echo "Job finished with exit code $? at: $(date)"

