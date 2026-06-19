cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

DATASET=$1
combinations=(
    # "train train"
    "train test"
    "train valid"
    # "test test"
    # "test valid"
    # "valid valid"
)

for combination in "${combinations[@]}"
do
    set -- $combination
    fa1=$1
    fa2=$2
    echo show-coords -lcr ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/1_aln_${DATASET}_${fa1}_${fa2}.delta > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/2_show_coords_${DATASET}_${fa1}_${fa2}.txt
    show-coords -lcr ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/1_aln_${DATASET}_${fa1}_${fa2}.delta > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/2_show_coords_${DATASET}_${fa1}_${fa2}.txt
done