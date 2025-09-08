DATASET=$1
OUTPUT_FILE="/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${DATASET}/dataset_stats/dataset_similarity/mummer/summary_${DATASET}.txt"

for data_type in train test valid
do  
    {
        echo "Processing ${data_type} data"
        echo "Before cleaning"
        before_count=$(ufasta sizes -H /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${data_type}.fasta | wc -l)
        echo "    Count: ${before_count}"
        
        echo "After cleaning"
        after_count=$(ufasta sizes -H /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${DATASET}/extracted_fasta/sequences_${data_type}.cleaned.fasta | wc -l)
        echo "    Count: ${after_count}"
        
        if [ $before_count -ne 0 ]; then
            ratio=$(echo "scale=4; $after_count / $before_count" | bc)
            echo "    Ratio (after/before): ${ratio}"
        else
            echo "    Before count is zero, cannot compute ratio."
        fi
        echo ""
    } >> "$OUTPUT_FILE"
done