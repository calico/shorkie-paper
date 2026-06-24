#!/bin/sh
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --job-name=fig07_cov_jo
#SBATCH --output=fig07_cov_jo_%j.log
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=32G
#
# Figure 7 J-O coverage tracks: 8-fold Shorkie ensemble PREDICTED Ref/Alt RNA-Seq(T0)
# coverage for the six eQTL SNPs (J YER080W, K YLR036C, L YKL078W, M YKR087C/OMA1,
# N YNL239W/LAP3, O YGR046W). Saves reproduced/ism/cov_<panel>.npz.
# Submit from the repo root:  sbatch reproduction/figure_07/panels/run_cov_eqtl_jo.sbatch
set -e

REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
cd "$REPO"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yeast_ml

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   repo: $REPO"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible)"

python reproduction/figure_07/panels/run_cov_eqtl_jo.py
echo "SBATCH_DONE"
