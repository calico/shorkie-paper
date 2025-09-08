#!/bin/bash

# Define the input file
DATASET=$1

for train_test_vad in train test valid;
do
    input_file="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}.bed"
    echo $input_file

    # Create an output directory
    output_dir="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/"
    mkdir -p $output_dir

    # Use awk to split the file
    awk -v outdir="$output_dir" '{
        last_column = $NF;
        print $0 >> outdir "/" last_column ".txt"
    }' "$input_file"

    echo "Files split by the last column are saved in the $output_dir directory."
done