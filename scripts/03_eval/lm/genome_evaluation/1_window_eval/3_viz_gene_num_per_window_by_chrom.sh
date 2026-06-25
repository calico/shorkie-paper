#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

data_type=$1
# Directory containing the fasta files
FASTA_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
OUTPUT_DIR="${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/window_eval"
mkdir -p $OUTPUT_DIR

# Iterate over each .cleaned.fasta file in the directory
for file in "$FASTA_DIR"/*.cleaned.fasta; do
    # Extract the base name without extension
    base_name=$(basename "$file" .cleaned.fasta)
    ovp_input_f=${OUTPUT_DIR}/${base_name}_ovp.txt
    ovp_output_f=${OUTPUT_DIR}/${base_name}_gene_count_per_window_by_chrom.png
    echo "Processing $base_name ..."
    echo "  ovp_input_f: $ovp_input_f"
    echo " ovp_output_f: $ovp_output_f"

    python 3_viz_gene_num_per_window_by_chrom.py --input ${ovp_input_f} --output ${ovp_output_f}
done