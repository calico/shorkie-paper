cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"

DATASET=$1
for train_test_vad in train test valid;
do
    mkdir -p ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/
    output_dir="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/"
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

            if [ -f "${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/fasta/${genome_id}.cleaned.fasta.masked.dust.softmask" ]; then
                genome_fa="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/fasta/${genome_id}.cleaned.fasta.masked.dust.softmask"
            else
                genome_fa="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/fasta/${genome_id}.cleaned.fasta"
            fi

            echo "genome fasta: $genome_fa"

            echo bedtools getfasta -fi ${genome_fa} -bed ${FILE} -fo ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/${genome_id}.fasta
            bedtools getfasta -fi ${genome_fa} -bed ${FILE} -fo ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/sequences_${train_test_vad}_split/${genome_id}.fasta
            echo ""
            fi
        done
    else
        echo "Directory $output_dir does not exist."
    fi
done
