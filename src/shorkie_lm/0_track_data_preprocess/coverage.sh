#!/bin/bash

#SBATCH --job-name=rossi # job name
#SBATCH --ntasks=1 # number of nodes to allocate per job
#SBATCH --cpus-per-task=4 # cpus (threads) per task
#SBATCH --mem 32000 # request 32Gb RAM per node
#SBATCH -o rossi.%j.out
#SBATCH --mail-type=end                         # send email upon: completion
#SBATCH --mail-user=mo@calicolabs.com               # address to send status email

# example use
# sbatch --array=1-20 /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/normalize_ChIPs.sh \
# /group/idea/basenji_yeast/data/rossi_et_al/bigwigs/rossi \
# /group/idea/basenji_yeast/data/rossi_et_al/bigwigs/rossi

EXP_DIR=$1
SAVE_DIR=$2
# SAMPLE_SHEET="${EXP_DIR}/${SLURM_ARRAY_TASK_ID}_sheet.csv"
SAMPLE_SHEET="${EXP_DIR}/sample_sheet.csv"
SEQ_DEPTH=${SAVE_DIR}/n_reads.csv
touch $SEQ_DEPTH

source /home/mo/mo_envs/cb2.bashrc
conda activate basenji

normalize_exp () {
    export SAMPLE_PATH=${EXP_DIR}/${SAMPLE_NAME}/${SAMPLE_NAME}
    
    # remove reads in masked regions
    bedtools intersect -abam ${SAMPLE_PATH}.bam -b /group/idea/basenji_yeast/basenji_files/mask/mask_rossi.bed -v > ${SAMPLE_PATH}_masked.bam
    # filter reads w/ less than ~90% matched reads, select read1 for coverage
    samtools view -f 0x40 -bSq 10 ${SAMPLE_PATH}_masked.bam > ${SAMPLE_PATH}_filtered.bam
    
    # RPM normalization
    NREADS=$(samtools view -c ${SAMPLE_PATH}_filtered.bam)
    WRITE_READS="${SAMPLE_NAME},${NREADS}\n"
    printf $WRITE_READS >> $SEQ_DEPTH
    # floating pt operations for scaling factor
    # export RPM=$NREADS/10^6
    # RPM=$(bc -l <<< $RPM)
    # export SCALE=1/$RPM
    # SCALE=$(bc -l <<< $SCALE)
    
    # calculate bam coverage
    bedtools genomecov -bg -ibam ${SAMPLE_PATH}_filtered.bam -5 > ${SAMPLE_PATH}_coverage.bedgraph
    # covert to bigwig no compression
    sort -k 1,1 -k2,2n ${SAMPLE_PATH}_coverage.bedgraph > ${SAMPLE_PATH}_sort2.bedgraph
    /home/mo/bedGraphToBigWig -unc ${SAMPLE_PATH}_sort2.bedgraph /group/idea/basenji_yeast/data/references/chrom.sizes ${SAMPLE_PATH}_coverage.bw
 
}

{
    read
    while IFS=, read GeneName treatement Antibody replicates fastq_dir sample_name batch_id; do
            SAMPLE_NAME=$sample_name
            echo $SAMPLE_NAME
            normalize_exp
    done 
} < $SAMPLE_SHEET