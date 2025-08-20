#!/bin/bash

DATASET=$1
for train_test_vad in train test valid; do
    mkdir -p /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/
    output_dir="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/"
    merged_fasta="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${train_test_vad}.fasta"

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
