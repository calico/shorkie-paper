DATASET=$1
for fa1 in train
do
    for fa2 in test valid
    do
        # mkdir -p /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/dashing2/
        awk '{print $10, $16}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/2_show_coords_${DATASET}_${fa1}_${fa2}.txt | tail -n +6 > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/5_viz_target_${DATASET}_${fa1}_${fa2}.txt
    done
done