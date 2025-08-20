#!/bin/bash
#SBATCH --job-name=atten_map
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=END
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=64G
#SBATCH --array=0-15

# Define an array with the 16 yeast chromosomes.
chromosomes=(chrI chrII chrIII chrIV chrV chrVI chrVII chrVIII chrIX chrX chrXI chrXII chrXIII chrXIV chrXV chrXVI)
chrom=${chromosomes[$SLURM_ARRAY_TASK_ID]}
echo "Processing chromosome: $chrom"

outdir="./embeddings_LM_sequence"
mkdir -p ${outdir}

# Call the predict script with the chromosome filter.
python 1_predict_seqs_LM.py \
    --fasta_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
    --gtf_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf \
    --out_file ${outdir}/embeddings_${chrom}.h5 \
    --model_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train/model_best.h5 \
    --params_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train/params.json \
    --seq_len 16384 \
    --chrom_filter $chrom \
    --max_count -1
