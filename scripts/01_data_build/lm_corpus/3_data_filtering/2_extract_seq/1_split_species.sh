#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"

# Define the input file
DATASET=$1

for train_test_vad in train test valid;
do
    input_file="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}.bed"
    echo $input_file

    # Create an output directory
    output_dir="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/"
    mkdir -p $output_dir

    # Use awk to split the file
    awk -v outdir="$output_dir" '{
        last_column = $NF;
        print $0 >> outdir "/" last_column ".txt"
    }' "$input_file"

    echo "Files split by the last column are saved in the $output_dir directory."
done