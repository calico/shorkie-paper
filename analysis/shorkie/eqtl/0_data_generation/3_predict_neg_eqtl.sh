#!/bin/sh
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=yeast_experiments
#SBATCH --output=job_output_%A_%a.log
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=16G
#SBATCH --array=0-31  # 32 tasks: 8 folds x 4 negative sets per fold

# Defaults
NEG_TSV_PREFIX="results/negative_eqtls_set"
ROOT_DIR="../.."

# Override with environment variables if provided
export NEG_TSV_PREFIX=${NEG_TSV_PREFIX_ENV:-$NEG_TSV_PREFIX}
export ROOT_DIR=${ROOT_DIR_ENV:-$ROOT_DIR}

# Calculate fold index and negative set index from SLURM_ARRAY_TASK_ID
fold_index=$(( SLURM_ARRAY_TASK_ID / 4 ))
neg_set_index=$(( SLURM_ARRAY_TASK_ID % 4 + 1 ))
fold_param="f${fold_index}c0"  # e.g., f0c0, f1c0, ..., f7c0

# Set output directory and VCF file input based on negative set index
output_dir="eqtl_neg/${fold_param}/set${neg_set_index}/"
vcf_file_input="${NEG_TSV_PREFIX}${neg_set_index}.negative.vcf"

# Set the current experiment directory based on the fold
current_experiment="${ROOT_DIR}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/${fold_param}/train/"

echo "Starting job on $(hostname) at $(date)"
echo "============================"
echo "Fold: ${fold_param}"
echo "Negative set: ${neg_set_index}"
echo "Current experiment: ${current_experiment}"
echo "Output directory: ${output_dir}"
echo "VCF input file: ${vcf_file_input}"

# Create output directory
mkdir -p "$output_dir"

# Run the command
python ${ROOT_DIR}/baskerville-yeast/src/baskerville/scripts/hound_snpgene_fix.py \
    -f ${ROOT_DIR}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask \
    -g ${ROOT_DIR}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf \
    -o "$output_dir" \
    --rc \
    --stats logSED \
    -t ${ROOT_DIR}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq_T0.txt \
    "${current_experiment}params.json" \
    "${current_experiment}model_best.h5" \
    "${vcf_file_input}" \
    1>"$output_dir/eQTL.out" \
    2>"$output_dir/eQTL.err"

echo "============================"
echo "Job finished with exit code $? at: $(date)"
