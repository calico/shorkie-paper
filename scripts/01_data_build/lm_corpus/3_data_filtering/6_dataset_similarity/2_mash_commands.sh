cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"
DATASET=$1
for fa1 in train
do
    for fa2 in test valid
    do
        mkdir -p ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mash/
        mash dist ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${fa1}.fasta \
        ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${fa2}.fasta > \
        ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mash/${DATASET}_${fa1}_${fa2}_before.txt

        mash dist ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${fa1}.cleaned.fasta \
        ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${fa2}.cleaned.fasta > \
        ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mash/${DATASET}_${fa1}_${fa2}_after.txt
    done
done