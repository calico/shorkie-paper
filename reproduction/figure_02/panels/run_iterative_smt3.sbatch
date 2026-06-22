#!/bin/sh
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=fig02_iter_smt3
#SBATCH --output=reproduction/figure_02/panels/fig02_iter_smt3_%j.log
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=48G
#
# Figure 2 panels 2A-iterative + 2B: iterative 15%-masked PPM reconstruction of the
# SMT3 (YDR510W) locus with the Shorkie LM (unet_small_bert_drop). Masking convention
# matches baskerville hound_eval_mlm_perplexity_region.py (the Fig-1G evaluator).
#
# Submit from repo root: sbatch reproduction/figure_02/panels/run_iterative_smt3.sbatch
set -e
REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yeast_ml
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python reproduction/figure_02/panels/run_iterative_smt3.py
echo "[done] iterative SMT3 inference"
