#!/bin/sh

#SBATCH --job-name=2_plot_dna_logo
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1-chess
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0-23

# Define the arrays (4 genes x 5 experiments = 20 jobs)

genes=("COA4" "ERI1" "RSM25" "ERD1" "MRM2" "SNT2" "CSI2" "RPE1" "PKC1" "AIM11" "MAE1" "MRPL1")
# exps=("all_random_seqs" "challenging_seqs" "yeast_seqs" "high_exp_seqs" "low_exp_seqs")

exps=("all_random_seqs" "challenging_seqs")

# Compute gene and experiment indices from SLURM_ARRAY_TASK_ID.
# We assume the following ordering: for each experiment in exps, loop through all genes.
# That is, the job index i corresponds to:
#   experiment index: i / 4 (integer division)
#   gene index: i % 4
gene_index=$(( SLURM_ARRAY_TASK_ID % ${#genes[@]} ))
exp_index=$(( SLURM_ARRAY_TASK_ID / ${#genes[@]} ))

# Set the corresponding gene and experiment names
gene=${genes[$gene_index]}
exp=${exps[$exp_index]}

strand="neg"

if [ $strand == "pos" ]; then
    output_dir="MPRA/${exp}/${gene}_pos"
    promoter_seqs="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/${exp}_fix.csv"
elif [ $strand == "neg" ]; then
    output_dir="MPRA/${exp}/${gene}_neg"
    promoter_seqs="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/${exp}_fix_rev.csv"
fi

# Run the Python script with the computed values
python /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/baskerville-yeast/src/baskerville/scripts/hound_MPRA_folds.py --f_list 0,1,2,3,4,5,6,7 \
    -e yeast_ml -r --tsv ${promoter_seqs} \
    --ctx /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/genes/neg/${gene}.tsv -o ${output_dir} \
    --rc -q parallel --stats logSED \
    -t /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_all_RNA-Seq_strand.txt \
    -f /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta \
    -g /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf \
    /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/params.json \
    /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/
