cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

DATASET=$1

combinations=(
    "train test"
    "train valid"
)

for combination in "${combinations[@]}"
do
    set -- $combination
    fa1=$1
    fa2=$2
    mkdir -p ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/minimap2/
    minimap2 -x asm20 ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${fa2}.fasta ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${fa1}.fasta > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/minimap2/overlaps_${fa2}.paf    # Nanopore read overlap
done
