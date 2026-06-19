#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

data_type=$1
# Directory containing the fasta files
#SEQ_BED_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_${data_type}/sequences_train_split"

REF_FASTA="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"
REF_GTF="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_r64_gtf/gtf/GCA_000146045_2.59.gtf"
ref_base_name=$(basename "$REF_FASTA" .cleaned.fasta)

FASTA_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_${data_type}/fasta"
GTF_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_${data_type}/gtf"
OUTPUT_DIR="${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/${data_type}/genome_dist/${data_type}/mummer"

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
    
    show-coords -lcr ${OUTPUT_DIR}/nucmer_aln_${data_type}_${ref_base_name}_${base_name}.delta > ${OUTPUT_DIR}/2_show_coords_${data_type}_${ref_base_name}_${base_name}.txt

    echo ""
done