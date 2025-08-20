DATASET=$1
for train_test_vad in train test valid;
do
    mkdir -p /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/
    output_dir="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/"
    mkdir -p ${output_dir}
    # echo "$output_dir"
    # Check if the provided argument is a directory

    # Check if the directory exists
    if [ -d "$output_dir" ]; then
    # Iterate through files in the directory
    for FILE in "$output_dir"/*; do
        # Check if it's a file (not a directory)
        if [ -f "$FILE" ]; then
            # Perform your desired operations on each file
            BASENAME=$(basename "$FILE")
            # Remove the file extension
            genome_id="${BASENAME%.*}"
            echo "Processing file: $genome_id"
            # Add your code here to process each file without the extension
            # VERY IMPORTANT! Make sure to replace the following line with the correct path to the genome fasta file (RepeatMasker + Dust)

            if [ -f "/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/fasta/${genome_id}.cleaned.fasta.masked.dust.softmask" ]; then
                genome_fa="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/fasta/${genome_id}.cleaned.fasta.masked.dust.softmask"
            else
                genome_fa="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/fasta/${genome_id}.cleaned.fasta"
            fi

            echo "genome fasta: $genome_fa"

            echo bedtools getfasta -fi ${genome_fa} -bed ${FILE} -fo /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/${genome_id}.fasta
            bedtools getfasta -fi ${genome_fa} -bed ${FILE} -fo /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/${genome_id}.fasta
            echo ""
            fi
        done
    else
        echo "Directory $output_dir does not exist."
    fi
done
