#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
RMRB_LIB="$(cfg tools.rmrb_lib)"
REPEATMASKER_LIB="$(cfg tools.repeatmasker_lib)"

data_type=$1
# Directory containing the fasta files
FASTA_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_${data_type}/fasta"

for fasta_file in "$FASTA_DIR"/*.cleaned.fasta; do
    base_name=$(basename "$fasta_file" .cleaned.fasta)
    # if [[ $base_name == "GCA_000146045_2" ]]; then
    #     echo "Skipping: RepeatMasker -pa 8 -xsmall $fasta_file"
    #     continue
    # fi
    echo "Running: RepeatMasker -pa 8 -species fungi -gff -xsmall $fasta_file"
    #RepeatMasker -pa 8 -species "Saccharomyces cerevisiae" -gff -xsmall $fasta_file
    #RepeatMasker -pa 8 -gff -xsmall -lib ${RMRB_LIB} $fasta_file
    RepeatMasker -pa 8 -gff -xsmall -lib ${REPEATMASKER_LIB} -e rmblast $fasta_file &
done
