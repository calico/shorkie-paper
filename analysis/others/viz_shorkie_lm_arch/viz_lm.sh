#!/bin/sh

#SBATCH --partition=parallel
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=yeast_experiments
#SBATCH --output=job_output_%A_%a.log
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=128G
#SBATCH --array=0

output_dir="model_viz/"
# exp_name='_big'
#exp_name='_small'
#exp_name='_unet_big'
# exp_name='_unet_small'
exp_name='_shorkie'
# current_experiment="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf${exp_name}/train"

# current_experiment="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train"

current_experiment="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f0c0/train"


echo "Starting job on $(hostname) at $(date)"
echo "============================"
echo "Testing eQTL: ${current_experiment} "

# Create output directory
mkdir -p "$output_dir"

echo "$current_experiment/params.json"
echo "$current_experiment/model_best.h5"

# Run the command
echo python /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/baskerville-yeast/src/baskerville/scripts/hound_model_viz.py \
    -o model_viz/ \
    --rc \
    -n _shorkie \
    /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f0c0/train/params.json \
    /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f0c0/train/model_best.h5

echo "============================"
echo "Job finished with exit code $? at: $(date)"


echo python /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/baskerville-yeast/src/baskerville/scripts/hound_model_viz.py \
    -o "$output_dir" \
    --rc \
    -n "$exp_name" \
    ${current_experiment}/params.json \
    ${current_experiment}/model_best.h5 \
    1>"$output_dir/model_viz.out" \
    2>"$output_dir/model_viz.err"