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
    echo "Running: RepeatMasker -pa 8 -species fungi -gff -xsmall $fasta_file"
    #RepeatMasker -pa 8 -species "Saccharomyces cerevisiae" -gff -xsmall $fasta_file
    #RepeatMasker -pa 8 -gff -xsmall -lib /home/khc/tools/RMRB/Libraries/RMRBSeqs.fasta $fasta_file
    RepeatMasker -pa 8 -gff -xsmall -lib /home/khc/bin/RepeatMasker/Libraries/RepeatMasker.lib -e rmblast $fasta_file &
done
