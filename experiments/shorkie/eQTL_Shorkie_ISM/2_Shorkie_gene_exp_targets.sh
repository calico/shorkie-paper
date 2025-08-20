#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1-chess
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-950  # Set the array to run through 951 bed files (0 to 950)

fold_param="f0c0"  # Dynamic fold_param based on array task ID
exp_data="eqtl"

output_dir="gene_exp_motif_${exp_data}/${fold_param}/${SLURM_ARRAY_TASK_ID}/"

fine_tuned_model_type="exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"
current_experiment="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tuned_model_type}/16bp/self_supervised_unet_small_bert_drop/train/${fold_param}/train"

echo "Starting job on $(hostname) at $(date)"
echo "============================"
echo "Testing eQTL: ${current_experiment} "

# Create output directory
mkdir -p "$output_dir"

echo "$current_experiment/params.json"
echo "$current_experiment/model_best.h5"

# Define the bed file path
bed_file_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/eQTL_Shorkie_ISM/results/output_variants_80bp/"
bed_file="${bed_file_dir}output_variants_80bp_windows_$(printf "%03d" ${SLURM_ARRAY_TASK_ID}).bed"

echo "Using bed file: $bed_file"

# Run the command
python /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/baskerville-yeast/src/baskerville/scripts/hound_ism_bed.py \
    -f /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
    -o "$output_dir" \
    -p 8 \
    --rc \
    -l 80 \
    --stats logSED \
    -t /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/${fine_tuned_model_type}/16bp/cleaned_sheet_RNA-Seq.txt \
    ${current_experiment}/params.json \
    ${current_experiment}/model_best.h5 \
    "$bed_file" \
    1>"$output_dir/gene_exp_motif_${exp_data}.out" \
    2>"$output_dir/gene_exp_motif_${exp_data}.err"

echo "============================"
echo "Job finished with exit code $? at: $(date)"
