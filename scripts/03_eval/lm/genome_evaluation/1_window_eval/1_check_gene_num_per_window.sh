#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

data_type=$1
# Directory containing the fasta files
# SEQ_BED="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_${data_type}/sequences.bed"
SEQ_BED_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/sequences_split"
FASTA_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
GTF_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/gtf"
OUTPUT_DIR="${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/window_eval/"

mkdir -p $OUTPUT_DIR

# Iterate over each .cleaned.fasta file in the directory
for file in "$FASTA_DIR"/*.cleaned.fasta; do
    # Extract the base name without extension
    base_name=$(basename "$file" .cleaned.fasta)

    # if [[ $base_name != "GCA_000222805_1" ]]; then
    #     continue
    # fi
    gtf_annotation_f=$GTF_DIR/${base_name}.59.gtf
    fasta_input_f=${FASTA_DIR}/${base_name}.fasta
    ovp_output_f=${OUTPUT_DIR}/${base_name}_ovp.txt
    bed_file_full=${SEQ_BED_DIR}/${base_name}.txt
    echo "Processing $base_name ..."
    echo "  bed_file_full: $bed_file_full"
    echo "  ovp_output_f: $ovp_output_f"
    echo "  gff: $gtf_annotation_f"

    python 1_check_gene_num_per_window.py --gff_file ${gtf_annotation_f} --bed_file ${bed_file_full} --output ${ovp_output_f}
    echo ""
done