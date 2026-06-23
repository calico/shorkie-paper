#!/bin/sh
#SBATCH --job-name=4_viz_motif
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mem=32G
#SBATCH --array=0-2

# Figure 2C (in-distribution / R64): render the de-novo TF-MoDISco CWM logos from the
# masked-LM modisco results. The .py loops the 3 LM architectures and reads
#   ${WORK_ROOT}/experiments/motif_LM/saccharomycetales_viz_seq/<arch>/modisco_results_w16384_n100000.h5
# writing logos to that arch's viz_self_modisco/ (all paths resolved via shorkie.config).
source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"

python 4_viz_motif.py > 4_viz_motif.out
