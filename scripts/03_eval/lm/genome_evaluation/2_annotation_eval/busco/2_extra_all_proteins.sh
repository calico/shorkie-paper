#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

data_type=$1
# Directory containing the fasta files
FASTA_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
GTF_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/gtf"

# Iterate over each .cleaned.fasta file in the directory
for file in "$FASTA_DIR"/*.cleaned.fasta; do
    # Extract the base name without extension
    base_name=$(basename "$file" .cleaned.fasta)
    gtf_annotation_f=$GTF_DIR/${base_name}.59.gtf
    fasta_input_f=${FASTA_DIR}/${base_name}.fasta
    protein_out_f=${FASTA_DIR}/${base_name}.proteins.fasta
    echo "Processing $base_name ..."
    echo "  fasta: $fasta_input_f"
    echo "  gff: $gtf_annotation_f"

    gffread ${gtf_annotation_f} -g ${fasta_input_f} -y ${protein_out_f}

    # output_dir="${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/busco/"
    # mkdir -p $output_dir
    # echo busco -i $file -l ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/busco/fungi_odb10 -o ${output_dir} -m proteins -c 4
done

# busco -i 
