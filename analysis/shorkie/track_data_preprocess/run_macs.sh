#!/bin/bash

#SBATCH --job-name=MACS # job name
#SBATCH --ntasks=1 # number of nodes to allocate per job
#SBATCH --cpus-per-task=4 # cpus (threads) per task
#SBATCH --mem 8000 # request 16Gb RAM per node
#SBATCH -o MACS.%j.out
#SBATCH --mail-type=end                         # send email upon: completion
#SBATCH --mail-user=mo@calicolabs.com               # address to send status email

# example use
# sbatch --array=1-20 /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/run_macs.sh \
# /group/idea/basenji_yeast/data/rossi_et_al/bigwigs/rossi 

# example pipeline
# macs3 callpeak \
# --outdir ./ \
# --treatment FHL1_filtered.bam \
# --control /group/idea/basenji_yeast/data/rossi_et_al/bigwigs/negative_control/NOTAG/NOTAG_filtered.bam \
# --format BAM \
# --gsize 12071326 \
# --keep-dup all \
# --scale-to small \
# --slocal 128 \
# --llocal 1024 \
# --max-gap 8 \
# --min-length 16 \
# --nomodel \
# --extsize 16 \
# --q 0.01 \
# --bdg \
# --name FHL1
# macs3 bdgcmp \
# --outdir ./ \
# --tfile FHL1_treat_pileup.bdg \
# --cfile FHL1_control_lambda.bdg \
# --o-prefix FHL1 \
# --pseudocount 1 \
# --method logFE
# sort -k 1,1 -k2,2n FHL1_logFE.bdg > FHL1_sort_logFE.bedgraph
# awk -v OFS="\t" ' {if($4<0)$4=0}1 ' FHL1_sort_logFE.bedgraph > FHL1_pos_logFE.bedgraph
# awk -v OFS="\t" ' {print $1, $2, $3, 1} ' FHL1_peaks.narrowPeak > FHL1_peaks.bedgraph
# bedtools merge -i FHL1_peaks.bedgraph -d 512 -c 4 -o sum > FHL1_peaks_merge.bedgraph
# # bedops --chop 512 /home/mo/mmagzoub/yeast_sequence_models/basenji_exp/TF_20220922/data_short/sort_sequences.bed > bin_sequences.bed
# # bedtools makewindows -g /group/idea/basenji_yeast/data/references/chrom.sizes -w 512 > chrom_bins.bed
# bedtools intersect -loj -c -a /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/bin_sequences.bed  -b FHL1_peaks_merge.bedgraph > FHL1_bins.bedgraph
# bedtools unionbedg -i FHL1_pos_logFE.bedgraph FHL1_peaks_merge.bedgraph -filler 0 > FHL1_join.bedgraph
# awk -v OFS="\t" ' {if($5==0)$4=0}1 ' FHL1_join.bedgraph |  awk -v OFS="\t" ' {print $1, $2, $3, $4} ' > FHL1_peaks_logFE.bedgraph
# /home/mo/bedGraphToBigWig -unc FHL1_peaks_logFE.bedgraph /group/idea/basenji_yeast/data/references/chrom.sizes FHL1_peaks_logFE.bw

EXP_DIR=$1
SAMPLE_SHEET="${EXP_DIR}/${SLURM_ARRAY_TASK_ID}_sheet.csv"

NEG_CNTRL=/group/idea/basenji_yeast/data/rossi_et_al/bigwigs/negative_control/NOTAG/NOTAG_filtered.bam

source /home/mo/mo_envs/cb2.bashrc
conda activate macs

macs_bigwig () {
    SAMPLE_DIR=${EXP_DIR}/${SAMPLE_NAME}
    
    # if test -f "${SAMPLE_DIR}/${SAMPLE_NAME}_join.bedgraph"; then
    #     echo "MACS output exists."
    # else
    # call peaks
    macs3 callpeak \
    --outdir $SAMPLE_DIR \
    --treatment "${SAMPLE_DIR}/${SAMPLE_NAME}_filtered.bam" \
    --control $NEG_CNTRL \
    --format BAM \
    --gsize 12071326 \
    --keep-dup all \
    --scale-to small \
    --slocal 128 \
    --llocal 1024 \
    --max-gap 8 \
    --min-length 16 \
    --nomodel \
    --extsize 16 \
    --q 0.05 \
    --bdg \
    --name $SAMPLE_NAME

    # calculate log fold change
    macs3 bdgcmp \
    --outdir $SAMPLE_DIR \
    --tfile "${SAMPLE_DIR}/${SAMPLE_NAME}_treat_pileup.bdg" \
    --cfile "${SAMPLE_DIR}/${SAMPLE_NAME}_control_lambda.bdg" \
    --o-prefix $SAMPLE_NAME \
    --pseudocount 1 \
    --method logFE  
    # fi
            
    # mask negative log fold change
    sort -k 1,1 -k2,2n "${SAMPLE_DIR}/${SAMPLE_NAME}_logFE.bdg" > "${SAMPLE_DIR}/${SAMPLE_NAME}_sort_logFE.bedgraph"
    awk -v OFS="\t" ' {if($4<0)$4=0}1 ' "${SAMPLE_DIR}/${SAMPLE_NAME}_sort_logFE.bedgraph" > "${SAMPLE_DIR}/${SAMPLE_NAME}_pos_logFE.bedgraph"
    
    # begraph to bigwig
    /home/mo/bedGraphToBigWig -unc \
    "${SAMPLE_DIR}/${SAMPLE_NAME}_pos_logFE.bedgraph" \
    /group/idea/basenji_yeast/data/references/chrom.sizes \
    "${SAMPLE_DIR}/${SAMPLE_NAME}_pos_logFE.bw"
    
    # if peaks detected
    if [ -s "${SAMPLE_DIR}/${SAMPLE_NAME}}_peaks.narrowPeak" ]; then
        # intersect tfrecord sequence with log fold changes peaks
        awk -v OFS="\t" ' {print $1, $2, $3, 1} ' "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks.narrowPeak" > "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks.bedgraph"
        bedtools merge -i "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks.bedgraph" -d 512 -c 4 -o sum > "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks_merge.bedgraph"
        bedtools intersect -loj -c -a /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/bin_sequences.bed  -b "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks_merge.bedgraph" > "${SAMPLE_DIR}/${SAMPLE_NAME}_bins.bedgraph"
        bedtools unionbedg -i "${SAMPLE_DIR}/${SAMPLE_NAME}_pos_logFE.bedgraph" "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks_merge.bedgraph" -filler 0 > "${SAMPLE_DIR}/${SAMPLE_NAME}_join.bedgraph"
        awk -v OFS="\t" ' {if($5==0)$4=0}1 ' "${SAMPLE_DIR}/${SAMPLE_NAME}_join.bedgraph" |  awk -v OFS="\t" ' {print $1, $2, $3, $4} ' > "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks_logFE.bedgraph"
        # begraph to bigwig
        /home/mo/bedGraphToBigWig -unc \
        "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks_logFE.bedgraph" \
        /group/idea/basenji_yeast/data/references/chrom.sizes \
        "${SAMPLE_DIR}/${SAMPLE_NAME}_peaks_logFE.bw"
    fi


        
}

{
    read
    while IFS=, read GeneName treatement Antibody replicates fastq_dir sample_name batch_id; do
            SAMPLE_NAME=$sample_name
            echo $SAMPLE_NAME
            macs_bigwig
    done 
} < $SAMPLE_SHEET
