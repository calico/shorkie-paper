#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"

DATASET=$1
for train_test_vad in train test valid; do
    mkdir -p ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/
    output_dir="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/"
    merged_fasta="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${train_test_vad}.fasta"

    if [ -d "$output_dir" ]; then
        # Create or empty the merged fasta file
        > "$merged_fasta"

        find "$output_dir" -type f -name '*.fasta' | while read FILE; do
            BASENAME=$(basename "$FILE")
            genome_id="${BASENAME%.*}"

            # Use awk for efficient processing
            awk -v genome_id="$genome_id" '
            /^>/ {print $0 "|" genome_id; next}
            {print}
            ' "$FILE" >> "$merged_fasta"

        done

        echo "Merged FASTA file created: $merged_fasta"
        echo ""
    else
        echo "Directory $output_dir does not exist."
    fi
    echo ""
done
