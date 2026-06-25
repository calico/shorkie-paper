#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_gene_level_eval_%A_%a.log
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1-chess
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --array=0-5
source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"

# Define datasets and times to plot
bed_file_root='${WORK_ROOT}/data/gene_exp_ism_window/'


target_tf=MSN2
exp_type=gene_exp_motif_test_TSS/f0c0/
dataset=../gene_exp_motif_test_TSS/f0c0/part${SLURM_ARRAY_TASK_ID}

# I want preserve 2 digits for the bed file
bed_file=${bed_file_root}TSS_chunk/TSS_windows_$(printf "%02d" $SLURM_ARRAY_TASK_ID).bed

# Run the Python script
python 0_timepoint_viz_coverage_sum.py \
    --exp_dir ${dataset} \
    --target_tf ${target_tf}_ \
    --bed ${bed_file} \
    --exp_type ${exp_type} \
    --libsize_file ${WORK_ROOT}/data/library_sizes_full.csv \
    --gtf ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCF_000146045.2_R64_genomic.fix.gtf