#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
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

python 1_saliency_score_genic_intergenic.py \
  --target_exp RP \
  --gtf_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf \
  --fasta_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
  --score_name logSED

  # --scores_h5 /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/motif_shorkie_RP_TSS/gene_exp_motif_test_MSN2_targets/f0c0/part0/scores.h5 \
  # --bed_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/MSN2_targets_chunk/MSN2_targets_windows_00.bed \
