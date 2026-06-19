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
    echo awk '{if ($10 > 80 && $16 > 10) {print }}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/2_show_coords_${DATASET}_${fa1}_${fa2}.txt > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt
    awk '{print $18}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_train.txt
    awk '{print $19}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_${fa2}.txt

    awk '{if ($10 > 80 && $16 > 10) {print }}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/2_show_coords_${DATASET}_${fa1}_${fa2}.txt > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt
    awk '{print $18}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_train.txt
    awk '{print $19}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs.txt > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/4_filter_aln_${DATASET}_${fa1}_${fa2}_paralogs_for_${fa2}.txt
done