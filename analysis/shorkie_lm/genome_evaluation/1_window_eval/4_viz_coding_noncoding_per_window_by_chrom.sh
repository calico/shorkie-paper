#!/bin/bash

data_type=$1
# Directory containing the fasta files
FASTA_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
OUTPUT_DIR="/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/window_eval"
mkdir -p $OUTPUT_DIR

# Iterate over each .cleaned.fasta file in the directory
for file in "$FASTA_DIR"/*.cleaned.fasta; do
    # Extract the base name without extension
    base_name=$(basename "$file" .cleaned.fasta)
    ovp_input_f=${OUTPUT_DIR}/${base_name}_ovp.txt
    ovp_output_f=${OUTPUT_DIR}/${base_name}_coding_noncoding_ratio_by_chrom.png
    echo "Processing $base_name ..."
    echo "  ovp_input_f: $ovp_input_f"
    echo " ovp_output_f: $ovp_output_f"
    python 4_viz_coding_noncoding_per_window_by_chrom.py --input ${ovp_input_f} --output ${ovp_output_f}
done