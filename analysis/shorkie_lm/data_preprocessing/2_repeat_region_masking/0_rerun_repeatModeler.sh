#!/bin/bash

data_type=$1
# Directory containing the fasta files
FASTA_DIR="/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_${data_type}/fasta"

for fasta_file in "$FASTA_DIR"/*.cleaned.fasta; do
    base_name=$(basename "$fasta_file" .cleaned.fasta)
    # if [[ $base_name == "GCA_000146045_2" ]]; then
    #     echo "Skipping: RepeatMasker -pa 8 -xsmall $fasta_file"
    #     continue
    # fi
    #RepeatMasker -pa 8 -gff -xsmall -lib /home/khc/tools/RMRB/Libraries/RMRBSeqs.fasta $fasta_file
    #echo BuildDatabase -name yeast $fasta_file
    #echo RepeatModeler -database yeast -threads 30 -LTRStruct >& run.out &
    echo BuildDatabase -name ${base_name} $fasta_file
    echo "RepeatModeler -database ${base_name} -threads 30 -LTRStruct >& run.out &"
    echo RepeatMasker -e rmblast -pa 8 -gff -xsmall -lib ${base_name}-families.fa $fasta_file
done
