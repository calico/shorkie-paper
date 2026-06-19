cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_RESULTS_ROOT="$(cfg corpus_build_results_root)"

DATASET=$1
# combinations=(
#     "train train"
#     "train test"
#     "train valid"
#     "test test"
#     "test valid"
#     "valid valid"
# )
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

    awk '{iden_query=$10/$2*100; seq_identity=$10/$11*100; print $1, $6, seq_identity, iden_query}' ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/minimap2/overlaps_${fa2}.paf > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/minimap2/overlaps_ratio_${fa2}.txt
done
