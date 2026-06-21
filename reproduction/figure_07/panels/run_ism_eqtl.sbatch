#!/bin/sh
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=fig07_ism
#SBATCH --output=fig07_ism_%j.log
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=32G
#
# Figure 7 panels A/B + J-O: 8-fold Shorkie ensemble logSED ISM for the
# OMA1 (chrXI:604356 A>G) and LAP3 (chrXIV:200328 G>A) eQTL SNPs.
# Submit from the repo root:  sbatch reproduction/figure_07/panels/run_ism_eqtl.sbatch
set -e

REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
cd "$REPO"

# Activate the yeast_ml conda env (canonical interpreter for this repo).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yeast_ml

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   repo: $REPO"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible)"

python reproduction/figure_07/panels/run_ism_eqtl.py
echo "SBATCH_DONE"
