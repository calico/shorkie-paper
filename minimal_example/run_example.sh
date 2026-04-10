#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=shorkie_minimal_example
#SBATCH --output=/home/kchao10/scr4_ssalzbe1/khchao/shorkie-paper/minimal_example/logs/run_example.out
#SBATCH --error=/home/kchao10/scr4_ssalzbe1/khchao/shorkie-paper/minimal_example/logs/run_example.err
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G

SCRIPT_DIR=/home/kchao10/scr4_ssalzbe1/khchao/shorkie-paper/minimal_example
MODEL_DIR=/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop
DATA=/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf

mkdir -p $SCRIPT_DIR/logs

/home/kchao10/miniconda3/envs/yeast_ml/bin/python3 $SCRIPT_DIR/run_shorkie_variant.py \
    --model_dir    $MODEL_DIR \
    --params_file  $SCRIPT_DIR/params.json \
    --targets_file $SCRIPT_DIR/sheet.txt \
    --fasta_file   $DATA/fasta/GCA_000146045_2.cleaned.fasta \
    --gtf_file     $DATA/gtf/GCA_000146045_2.59.gtf \
    --chrom chrI --pos 124373 --ref T --alt C --gene YAL016C-B

echo "============================"
echo "Job finished with exit code $? at: $(date)"
