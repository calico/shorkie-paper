#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"

data_type=$1
# Directory containing the fasta files
FASTA_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
GTF_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/gtf"

# Iterate over each .cleaned.fasta file in the directory
for file in "$FASTA_DIR"/*.cleaned.fasta; do
    # Extract the base name without extension
    base_name=$(basename "$file" .cleaned.fasta)
    protein_out_f=${FASTA_DIR}/${base_name}.proteins.fasta
    echo "Processing $base_name ..."
    echo "  fasta: $file"
    echo "  protein fa: $protein_out_f"

    output_dir=./
    mkdir -p $output_dir
    busco -i $protein_out_f -l ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/busco/fungi_odb10 -o ${output_dir} -m proteins -c 4 &
done

# busco -i 
