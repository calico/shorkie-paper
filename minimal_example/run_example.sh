#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=shorkie_minimal_example
#SBATCH --output=run_example_%j.out
#SBATCH --error=run_example_%j.err
#SBATCH --mail-type=end
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#
# SLURM wrapper for the minimal logSED example. The partition/account/time above
# are JHU-cluster examples — edit them for your site. All data paths resolve from
# config/paths.yaml via scripts/common/env.sh (no hardcoded machine paths).

source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Interpreter: defaults to the active `python3`; override with a full path if the
# yeast_ml env is not activated, e.g. SHORKIE_PYTHON=/path/to/envs/yeast_ml/bin/python3
PYTHON="${SHORKIE_PYTHON:-python3}"
MODEL_DIR=${WORK_ROOT}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop
DATA=${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf

"$PYTHON" $SCRIPT_DIR/run_shorkie_variant.py \
    --model_dir    $MODEL_DIR \
    --params_file  $SCRIPT_DIR/params.json \
    --targets_file $SCRIPT_DIR/sheet.txt \
    --fasta_file   $DATA/fasta/GCA_000146045_2.cleaned.fasta \
    --gtf_file     $DATA/gtf/GCA_000146045_2.59.gtf \
    --chrom chrI --pos 124373 --ref T --alt C --gene YAL016C-B

echo "============================"
echo "Job finished with exit code $? at: $(date)"
