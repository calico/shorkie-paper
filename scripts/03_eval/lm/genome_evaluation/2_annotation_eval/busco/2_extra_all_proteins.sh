#!/bin/bash

data_type=$1
# Directory containing the fasta files
FASTA_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
GTF_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/gtf"

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

    # output_dir="/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/busco/"
    # mkdir -p $output_dir
    # echo busco -i $file -l /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/busco/fungi_odb10 -o ${output_dir} -m proteins -c 4
done

# busco -i 
