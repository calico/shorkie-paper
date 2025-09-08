#!/bin/bash

while getopts i:f:r:b:s: flag
do
    case "${flag}" in
        i) id=${OPTARG};;
        f) fastq_dir=${OPTARG};;
        r) reference=${OTPARG};;
        b) bwa_index=${OPTARG};;
        s) save_dir=${OPTARG};;
    esac
done

# example use 
# bash /group/idea/basenji_yeast/bioinformatics/chipseq2bigwig.sh \
# -i SRR11466279 \
# -f /group/idea/basenji_yeast/data/rossi_et_al/meta_assemblage_fastq \
# -r /group/idea/basenji_yeast/data/references/S288C_R64-3-1.fsa \
# -b /group/idea/basenji_yeast/data/references/bwa/S288C_R64_bwaidx  
# -s ./tmp

cd "${save_dir}"

# 1) align
/home/mo/bwa-0.7.17/bwa mem -t 10 -v 0 -M "${bwa_index}" "${fastq_dir}/${id}_1.fastq" "${fastq_dir}/${id}_2.fastq" > "${id}_align.bwa"
samtools sort -o "${id}_sort.bam" "${id}_align.bwa"
# 2) remove multi-mappers
samtools view -h "${id}_sort.bam" | grep -v -e 'XA:Z:' | samtools view -b > "${id}_rmmultimap.bam"
# 3) remove pcr duplicates
java -jar /home/mo/idea_bioinformatics/picard.jar MarkDuplicates \
--ASSUME_SORT_ORDER coordinate \
--REMOVE_DUPLICATES \
-I "${id}_rmmultimap.bam" \
-O "${id}_rmdup.bam" \
-M "${id}_rmdup_metrics.txt"
# 4) mask hypervariable regions
bedtools intersect -abam "${id}_rmdup.bam" -b /group/idea/basenji_yeast/basenji_files/mask/mask_rossi.bed -v > "${id}_masked.bam"

