#!/bin/sh

#SBATCH --job-name=3_modisco_report
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=32G
#SBATCH --array=0-2

# Define the model architectures
model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
# model_archs=('unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')

# Get the current model architecture based on the array index
model=${model_archs[$SLURM_ARRAY_TASK_ID]}

current_dataset="TSS"  

if [ "${current_dataset}" = "RP" ]; then
    bedfile="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/RP_windows.bed"
    eval_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_RP_TSS/lm_saccharomycetales_gtf_${model}/eval_RP"
elif [ "${current_dataset}" = "TSS" ]; then
    bedfile="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/TSS_windows_ex_chrmt.bed"
    eval_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_RP_TSS/lm_saccharomycetales_gtf_${model}/eval_TSS"
else
    echo "Error: Unknown dataset: ${current_dataset}"
    exit 1
fi

out_dir="./viz_motif_${current_dataset}"
mkdir -p $out_dir/motifs_with_annotation/${model}

# python 2_modisco_DNA_logo.py --no_motif_annotation --model_arch ${model} \
#     --out_dir $out_dir/ \
#     --seq_bed ${bedfile} \
#     --predictions_file ${eval_dir}/preds.npz 1>"$out_dir/preds.out" 2>"$out_dir/preds.err"

python 2_modisco_DNA_logo.py --model_arch ${model} \
    --modisco_h5 /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_RP_TSS/lm_saccharomycetales_gtf_${model}/eval_${current_dataset}/modisco_results_w16384_n100000.h5 \
    --motifs_html /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_RP_TSS/lm_saccharomycetales_gtf_${model}/eval_${current_dataset}/report_high_conf/motifs.html \
    --out_dir $out_dir/ \
    --seq_bed ${bedfile} \
    --trim_threshold 0.3 \
    --trim_min_length 3 \
    --predictions_file ${eval_dir}/preds.npz #1>"$out_dir/motifs_with_annotation/${model}/preds.out" 2>"$out_dir/motifs_with_annotation/${model}/preds.err"
