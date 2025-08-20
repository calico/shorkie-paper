#!/bin/sh

#SBATCH --job-name=3_modisco_report
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=32G
#SBATCH --array=0-5

model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
datasets=('strains_select' 'schizosaccharomycetales')

# Calculate indices for model and dataset based on array task ID
model_idx=$((SLURM_ARRAY_TASK_ID / ${#datasets[@]}))
dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#datasets[@]}))

model=${model_archs[$model_idx]}
dataset=${datasets[$dataset_idx]}

# If you have multiple .meme files, you can do:
genome_fa="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask"
root_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_unseen_species/${dataset}_viz_seq/${model}"
for motif_file in ${root_dir}/motifs_meme/*.meme; do
    echo "motif_file: ${motif_file}"
    out_dir="${root_dir}/fimo_out/$(basename $motif_file .meme)"
    mkdir -p "$out_dir"
    fimo --oc "$out_dir" --verbosity 1 "$motif_file" ${genome_fa}
done
