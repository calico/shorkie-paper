cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
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
    echo awk '{if ($10 > 80 && $16 > 10) {print }}' ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/2_show_coords_${DATASET}_${fa1}_${fa2}.txt > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt
    awk '{print $18}' ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_train.txt
    awk '{print $19}' ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_${fa2}.txt

    awk '{if ($10 > 80 && $16 > 10) {print }}' ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/2_show_coords_${DATASET}_${fa1}_${fa2}.txt > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt
    awk '{print $18}' ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_train.txt
    awk '{print $19}' ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > ${CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_${fa2}.txt
done