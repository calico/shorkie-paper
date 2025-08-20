#!/bin/bash

data_type=$1
# Directory containing the fasta files
FASTA_DIR=/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta
GTF_DIR=/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/gtf

busco_output_dir=/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/busco/
short_summary_files=${busco_output_dir}/*.fasta/short_summary.*.txt
busco_summary_dir=${busco_output_dir}/BUSCO_summaries/

mkdir -p $busco_summary_dir

cp $short_summary_files $busco_summary_dir

python busco/generate_plot.py -wd ${busco_summary_dir}