#!/bin/sh
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --job-name=fig03_cov
#SBATCH --output=fig03_cov_%j.log
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=48G
#
# Figure 3 panels H/I/J: held-out-fold Shorkie vs Random_Init RNA-seq coverage at
# RPL7A / RPS16B-RPL13A (fold 3) and EFM5 (fold 6) + observed bigwig.
#
# ONE model per python process (cross-architecture Keras restore collision otherwise
# flat-lines the 2nd model to the softplus floor). Five sequential invocations, then
# a model-free merge that adds observed coverage.
#
# Submit from repo root: sbatch reproduction/figure_03/panels/run_coverage.sbatch
set -e
REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yeast_ml
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "host: $(hostname)  repo: $REPO"
nvidia-smi -L 2>/dev/null || echo "(no GPU visible)"
PY=python
RUN="reproduction/figure_03/panels/run_coverage.py"
$PY "$RUN" --tree random_init --fold 3
$PY "$RUN" --tree random_init --fold 6
$PY "$RUN" --tree shorkie     --fold 3
$PY "$RUN" --tree shorkie     --fold 6
$PY "$RUN" --merge
echo "SBATCH_DONE"
