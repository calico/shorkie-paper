# #!/bin/bash

# data_type=$1
# # Directory containing the fasta files
# FASTA_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
# OUTPUT_DIR="/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/window_eval"
# mkdir -p $OUTPUT_DIR

# in_species_csv="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/species_${data_type}.cleaned.csv"
# out_species_csv="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/species_${data_type}.cleaned.append.csv"

# # Iterate over each .cleaned.fasta file in the directory
# for file in "$FASTA_DIR"/*.cleaned.fasta; do
#     # Extract the base name without extension
#     base_name=$(basename "$file" .cleaned.fasta)
#     ovp_input_f=${OUTPUT_DIR}/${base_name}_ovp.txt
#     ovp_output_f=${OUTPUT_DIR}/${base_name}_summary_stats.txt
#     echo "Processing $base_name ..."
#     echo "  ovp_input_f: $ovp_input_f"
#     echo " ovp_output_f: $ovp_output_f"

#     # python 5_stats.py --input ${ovp_input_f} --output ${ovp_output_f}
#     python 5_1_stats_new_cleaned_csv.py --input ${ovp_input_f} 
# done

#!/bin/bash

data_type=$1
# Directory containing the fasta files
FASTA_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_${data_type}/fasta"
OUTPUT_DIR="/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/${data_type}/window_eval"
mkdir -p $OUTPUT_DIR

in_species_csv="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/species_${data_type}.cleaned.csv"
out_species_csv="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/species_${data_type}.cleaned.append.csv"

# Read the header of the input CSV and add new columns for the report
header=$(head -n 1 "$in_species_csv")
echo "$header,Overall_Avg_Overlap,Overall_Median_Overlap,Overall_Avg_Coding_Ratio" > "$out_species_csv"


# Iterate over each .cleaned.fasta file in the directory
for file in "$FASTA_DIR"/*.cleaned.fasta; do
    # Extract the base name without extension
    base_name=$(basename "$file" .cleaned.fasta)
    ovp_input_f=${OUTPUT_DIR}/${base_name}_ovp.txt
    ovp_output_f=${OUTPUT_DIR}/${base_name}_summary_stats.txt
    echo "Processing $base_name ..."
    echo "  ovp_input_f: $ovp_input_f"
    echo " ovp_output_f: $ovp_output_f"

    report_lines=$(python 5_1_stats_new_cleaned_csv.py --input "${ovp_input_f}")

    echo $report_lines
    python 5_stats.py --input ${ovp_input_f} --output ${ovp_output_f}
done


# Process each line in the input CSV
tail -n +2 "$in_species_csv" | while IFS=, read -r line; do
    base_name=$(echo "$line" | cut -d, -f6)
    echo "$base_name"
    # replace "." with "_"
    base_name=${base_name//./_}

    ovp_input_f="${OUTPUT_DIR}/${base_name}_ovp.txt"
    
    # Capture the report lines from the Python script
    report_lines=$(python 5_1_stats_new_cleaned_csv.py --input "${ovp_input_f}")
    
    # Append the report lines to the output CSV
    echo "$line,$report_lines" >> "$out_species_csv"
done
