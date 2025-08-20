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
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=32G
#SBATCH --array=0

# Define the model architectures
model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')

# Get the current model architecture based on the array index
model=${model_archs[$SLURM_ARRAY_TASK_ID]}

# If you have multiple .meme files, you can do:
root_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/${model}"

current_dataset="RP"  

if [ "${current_dataset}" = "RP" ]; then
    bedfile="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/RP_windows.bed"
    eval_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/5_modisco_DNA_logo/lm_saccharomycetales_gtf_unet_small_bert_drop/eval_RP"
elif [ "${current_dataset}" = "TSS" ]; then
    bedfile="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/TSS_windows_ex_chrmt.bed"
    eval_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/5_modisco_DNA_logo/lm_saccharomycetales_gtf_unet_small_bert_drop/eval_TSS"
else
    echo "Error: Unknown dataset: ${current_dataset}"
    exit 1
fi

out_dir="./viz_motif_${current_dataset}"
mkdir -p $out_dir

echo python 0_map_modisco_pattern_to_meme_db.py --modisco_h5 /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_RP_TSS/lm_saccharomycetales_gtf_unet_small_bert_drop/eval_RP/modisco_results_w16384_n100000.h5 \
    --seq_bed ${bedfile} \
    --out_dir $out_dir/ \
    --meme_db /home/kchao10/tools/motif_databases/YEAST/merged_meme.meme \
    --trim_threshold 0.3 \
    --trim_min_length 3 \
    --tomtom_exec tomtom 
    #1>"$out_dir/0.out" 2>"$out_dir/0.err"
