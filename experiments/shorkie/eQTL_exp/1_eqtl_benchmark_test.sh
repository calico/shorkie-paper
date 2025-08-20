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
#SBATCH --array=0  # Array to iterate over 8 folds (0 to 7)

# Map SLURM_ARRAY_TASK_ID to fold parameters
fold_param="f${SLURM_ARRAY_TASK_ID}c0"

current_experiment="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/${fold_param}/train/"
output_dir="eqtl_${fold_param}/"

echo "Starting job on $(hostname) at $(date)"
echo "============================"
echo "Testing eQTL: ${current_experiment} "

# Create output directory
mkdir -p "$output_dir"

echo "$current_experiment/params.json"
echo "$current_experiment/train/model_best.h5"
# eQTL_VCF="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/selected_eQTL/updated_intersected_data_CIS.vcf"
eQTL_VCF="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/selected_eQTL/updated_intersected_data_CIS_test.vcf"

# Run the command
python /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/baskerville-yeast/src/baskerville/scripts/hound_snpgene.py \
    -f /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask \
    -g /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf \
    -o "$output_dir" \
    --rc \
    --stats logSED \
    -t /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq_T0.txt \
    ${current_experiment}params.json \
    ${current_experiment}model_best.h5 \
    ${eQTL_VCF} \
    1>"$output_dir/eQTL.out" \
    2>"$output_dir/eQTL.err"

echo "============================"
echo "Job finished with exit code $? at: $(date)"
