#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

data_type=$1
# Directory containing the fasta files
FASTA_DIR=${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta
GTF_DIR=${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/gtf

busco_output_dir=${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/busco/
short_summary_files=${busco_output_dir}/*.fasta/short_summary.*.txt
busco_summary_dir=${busco_output_dir}/BUSCO_summaries/

mkdir -p $busco_summary_dir

cp $short_summary_files $busco_summary_dir

python busco/generate_plot.py -wd ${busco_summary_dir}