#!/bin/bash

data_type=$1
# Directory containing the fasta files
FASTA_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_${data_type}/fasta"

# Function to convert masked regions to lowercase
convert_to_lowercase() {
    masked_seq=$1
    unmasked_seq=$2
    output_seq=""

    for (( i=0; i<${#masked_seq}; i++ )); do
        char=${masked_seq:$i:1}
        if [ "$char" == "N" ]; then
            output_seq+=$(echo ${unmasked_seq:$i:1} | tr '[:upper:]' '[:lower:]')
        else
            output_seq+=${unmasked_seq:$i:1}
        fi
    done

    echo "$output_seq"
}




#    138955 GCA_000977445_2.cleaned.fasta.masked.dust
#         1 GCA_000977475_3.cleaned.fasta.masked.dust
#         1 GCA_000977535_2.cleaned.fasta.masked.dust
#         1 GCA_000977565_2.cleaned.fasta.masked.dust
#         1 GCA_000977595_2.cleaned.fasta.masked.dust
#         1 GCA_000977625_2.cleaned.fasta.masked.dust
#         1 GCA_000977655_2.cleaned.fasta.masked.dust
#         1 GCA_000977685_2.cleaned.fasta.masked.dust
#         1 GCA_000977745_2.cleaned.fasta.masked.dust
#         1 GCA_000977775_3.cleaned.fasta.masked.dust
#         1 GCA_000977805_2.cleaned.fasta.masked.dust
#         1 GCA_000977835_2.cleaned.fasta.masked.dust
#         1 GCA_000977865_2.cleaned.fasta.masked.dust
#         1 GCA_000977895_2.cleaned.fasta.masked.dust
#         1 GCA_000977925_2.cleaned.fasta.masked.dust
#         1 GCA_000977955_2.cleaned.fasta.masked.dust
#         1 GCA_000977985_2.cleaned.fasta.masked.dust
#         1 GCA_000978015_2.cleaned.fasta.masked.dust
#         1 GCA_000978045_2.cleaned.fasta.masked.dust
#         1 GCA_000978075_3.cleaned.fasta.masked.dust
#         1 GCA_000978105_2.cleaned.fasta.masked.dust
#         1 GCA_000978135_2.cleaned.fasta.masked.dust
#         1 GCA_000978195_2.cleaned.fasta.masked.dust
#         1 GCA_000978255_2.cleaned.fasta.masked.dust
#         1 GCA_000978315_2.cleaned.fasta.masked.dust

# base_names=(
#     "GCA_000977475_3"
#     "GCA_000977535_2"
#     "GCA_000977565_2"
#     "GCA_000977595_2"
#     "GCA_000977625_2"
#     "GCA_000977655_2"
#     "GCA_000977685_2"
#     "GCA_000977745_2"
#     "GCA_000977775_3"
#     "GCA_000977805_2"
#     "GCA_000977835_2"
#     "GCA_000977865_2"
#     "GCA_000977895_2"
#     "GCA_000977925_2"
#     "GCA_000977955_2"
#     "GCA_000977985_2"
#     "GCA_000978015_2"
#     "GCA_000978045_2"
#     "GCA_000978075_3"
#     "GCA_000978105_2"
#     "GCA_000978135_2"
#     "GCA_000978195_2"
#     "GCA_000978255_2"
#     "GCA_000978315_2"
# )

# for base_name in "${base_names[@]}"; do
    # base_name=$(basename "$fasta_file" .cleaned.fasta)

for fasta_file in "$FASTA_DIR"/*.cleaned.fasta; do
    base_name=$(basename "$fasta_file" .cleaned.fasta)
    echo "Processing: $base_name"

    fasta_file="$FASTA_DIR/$base_name.cleaned.fasta"
    echo "Running: dust ${fasta_file}.masked > ${fasta_file}.masked.dust 20"
    dust ${fasta_file}.masked > ${fasta_file}.masked.dust 20

    unmasked_fasta=${fasta_file}.masked
    masked_fasta=${fasta_file}.masked.dust
    output_fasta=${fasta_file}.masked.dust.softmask
    
    python 2__dust_softmask.py ${masked_fasta} ${unmasked_fasta} ${output_fasta}
    for file in ${unmasked_fasta} ${masked_fasta} ${output_fasta}; do  
        echo "Processing ${file}"
        n_count=$(grep -i -o N ${file} | wc -l)
        echo "Number of Ns: $n_count"
        lowercase_count=$(grep -v '^>' "${file}" | tr -cd 'acgt' | wc -c)
        echo "Number of lowercase nucleotides: $lowercase_count"
        echo ""
    done

    echo "Softmasked fasta file has been created: $output_fasta"
done
