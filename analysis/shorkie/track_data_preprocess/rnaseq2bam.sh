#!/bin/bash

source /home/mo/.bashrc
conda activate idea_rnaseq

while getopts i:r:s:d:1:2: flag
do
    case "${flag}" in
        i) id=${OPTARG};;
        r) reference=${OTPARG};;
        s) star_reference=${OPTARG};;
        d) save_dir=${OPTARG};;
        1) read_1=${OPTARG};;
        2) read_2=${OPTARG};;
    esac
done

ncores=$(grep -c ^processor /proc/cpuinfo)
nthreads=$(( $ncores > 16 ? 16 : $ncores )) 

# # example use
# bash /group/idea/basenji_yeast/bioinformatics/rnaseq2bigwig.sh \
# -i YIL038C_0_S49 \
# -r /group/idea/basenji_yeast/data/references/S288C_R64-3-1.fsa \
# -s /group/idea/basenji_yeast/data/references/STAR \
# -r1 /group/idea/basenji_yeast/data/IDEA/idea_rnaseq/test/YIL038C_0_S49_trimmed_R1.fastq.gz \
# -r2 /group/idea/basenji_yeast/data/IDEA/idea_rnaseq/test/YIL038C_0_S49_trimmed_R2.fastq.gz

# # build index
# /home/mo/idea_bioinformatics/bin/STAR \
# --runThreadN 4 \
# --runMode genomeGenerate \
# --genomeDir /group/idea/basenji_yeast/data/references/STAR \
# --genomeFastaFiles /group/idea/basenji_yeast/data/references/S288C_R64-3-1.fsa \
# --sjdbGTFfile /group/idea/basenji_yeast/data/references/S288C_R64-3-1.gtf \
# --outFileNamePrefix S288C_R64_

## Align Reads

cd "${save_dir}"

/home/mo/idea_bioinformatics/bin/STAR \
--quantMode TranscriptomeSAM GeneCounts \
--genomeDir /group/idea/yeast_references/STAR \
--runThreadN $nthreads \
--readFilesCommand gunzip -c \
--readFilesIn "${read_1}" "${read_2}" \
--outFileNamePrefix "${id}_" \
--outSAMtype BAM SortedByCoordinate \
--outSAMunmapped Within \
--outSAMattributes Standard \
--outFilterMultimapNmax 1 \
--bamRemoveDuplicatesType UniqueIdentical \
--alignIntronMin 10 \
--alignIntronMax 2500 \
--alignMatesGapMax 2500

# percent duplicates
java -jar /home/mo/idea_bioinformatics/picard.jar MarkDuplicates \
I="${id}_Aligned.sortedByCoord.out.bam" \
O="${id}_rmdup.bam" \
M="${id}_marked_dup_metrics.txt"

## Collect QC Metrics

# # insert size
# java -jar /home/mo/idea_bioinformatics/picard.jar CollectInsertSizeMetrics \
# I="${id}_Aligned.sortedByCoord.out.bam" \
# H="${id}_insert_size_histogram.pdf" \
# HISTOGRAM_WIDTH=1000 \
# O="${id}_insert_size_metrics.txt"

# # gc content
# java -jar /home/mo/idea_bioinformatics/picard.jar CollectGcBiasMetrics \
# I="${id}_Aligned.sortedByCoord.out.bam" \
# O="${id}_gc_bias_metrics.txt" \
# CHART="${id}_gc_bias_metrics.pdf" \
# S="${id}_summary_metrics.txt" \
# R="${reference}"
      
# samtools view "${id}_Aligned.sortedByCoord.out.bam" | \
# awk '{ if ( length($10) > 0 ) print $10 }' | \
# awk '{ n = (length($1) > 0) ? length($1) : 1; print gsub(/[GCCgcs]/, "", $1)/n;}' | \
# awk '{total += $1} END {print total/NR}' > \
# "${id}_gc.txt"
