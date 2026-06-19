#!/bin/bash

data_type=$1
# Directory containing the fasta files
# SEQ_BED="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/sequences.bed"
#SEQ_BED_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/sequences_train_split"

REF_FASTA="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"
REF_GTF="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf"
ref_base_name=$(basename "$REF_FASTA" .cleaned.fasta)

FASTA_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
GTF_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/gtf"
OUTPUT_DIR="/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/genome_dist/${data_type}/mash"

mkdir -p $OUTPUT_DIR

# Initialize empty arrays
gff_files=()
bed_files=()

# 1. Collect GTF and FASTA file paths
for fasta_file in "$FASTA_DIR"/*.cleaned.fasta; do
    base_name=$(basename "$fasta_file" .cleaned.fasta)

    gtf_file=$GTF_DIR/${base_name}.59.gtf
    bed_file_full=${SEQ_BED_DIR}/${base_name}.txt

    echo "Reference fasta: " $REF_FASTA
    echo "Reference gtf  : " $REF_GTF

    echo "Target fasta: " $fasta_file
    echo "Target gtf  : " $gtf_file

    mkdir -p ${OUTPUT_DIR}
    mash dist ${REF_FASTA} \
    ${fasta_file} > \
    ${OUTPUT_DIR}/${data_type}_${ref_base_name}_${base_name}.txt
    echo ""
done