#!/bin/bash

cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
RMRB_LIB="$(cfg tools.rmrb_lib)"

data_type=$1
# Directory containing the fasta files
FASTA_DIR="${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/data_${data_type}/fasta"

for fasta_file in "$FASTA_DIR"/*.cleaned.fasta; do
    base_name=$(basename "$fasta_file" .cleaned.fasta)
    # if [[ $base_name == "GCA_000146045_2" ]]; then
    #     echo "Skipping: RepeatMasker -pa 8 -xsmall $fasta_file"
    #     continue
    # fi
    #RepeatMasker -pa 8 -gff -xsmall -lib ${RMRB_LIB} $fasta_file
    #echo BuildDatabase -name yeast $fasta_file
    #echo RepeatModeler -database yeast -threads 30 -LTRStruct >& run.out &
    echo BuildDatabase -name ${base_name} $fasta_file
    echo "RepeatModeler -database ${base_name} -threads 30 -LTRStruct >& run.out &"
    echo RepeatMasker -e rmblast -pa 8 -gff -xsmall -lib ${base_name}-families.fa $fasta_file
done
