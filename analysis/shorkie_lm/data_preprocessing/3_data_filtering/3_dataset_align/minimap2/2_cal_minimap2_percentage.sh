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
    mkdir -p /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/minimap2/

    awk '{iden_query=$10/$2*100; seq_identity=$10/$11*100; print $1, $6, seq_identity, iden_query}' /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/minimap2/overlaps_${fa2}.paf > /scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/minimap2/overlaps_ratio_${fa2}.txt
done
