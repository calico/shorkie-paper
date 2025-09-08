#!/bin/bash

EXP_DIR=/group/idea/basenji_yeast/data/rossi_et_al/bigwigs/histone_marks
SAMPLE_SHEET="${EXP_DIR}/sample_sheet.csv"

source /home/mo/mo_envs/cb2.bashrc
conda activate yeast_seq

bamcov_bigwig () {
    SAMPLE_DIR=${EXP_DIR}/${SAMPLE_NAME}

    java -jar /home/mo/idea_bioinformatics/picard.jar MarkDuplicates \
    --REMOVE_DUPLICATES \
    -I "${SAMPLE_DIR}/${SAMPLE_NAME}.bam" \
    -O "${SAMPLE_DIR}/${SAMPLE_NAME}_rmdup.bam" \
    -M "${SAMPLE_DIR}/${SAMPLE_NAME}_rmdup_metrics.txt"

    python /home/mo/shared_repos/basenji/bin/bam_cov.py -c \
    "${SAMPLE_DIR}/${SAMPLE_NAME}_rmdup.bam" \
    "${SAMPLE_DIR}/${SAMPLE_NAME}_bamcov.bw"

}

{
    read
    while IFS=, read GeneName treatement Antibody replicates fastq_dir sample_name; do
            SAMPLE_NAME=$sample_name
            echo $SAMPLE_NAME
            bamcov_bigwig
    done 
} < $SAMPLE_SHEET
