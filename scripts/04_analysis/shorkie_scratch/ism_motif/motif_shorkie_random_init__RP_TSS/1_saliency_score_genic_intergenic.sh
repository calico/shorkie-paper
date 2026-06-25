#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mem=32G
#SBATCH --array=0
source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"

python 1_saliency_score_genic_intergenic.py \
  --target_exp RP \
  --gtf_file ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf \
  --fasta_file ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
  --score_name logSED