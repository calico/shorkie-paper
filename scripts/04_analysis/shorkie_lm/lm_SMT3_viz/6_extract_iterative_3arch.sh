#!/usr/bin/env bash
# Extract the Figure 2A "Shorkie LM 15% iterative" row from the precomputed 3-architecture
# LM ensemble (preds_train.npz) — CPU only, no GPU. Reads ~142 MB x3 npz (~1-2 min).
#
#   bash scripts/04_analysis/shorkie_lm/lm_SMT3_viz/6_extract_iterative_3arch.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source /home/kchao10/miniconda3/etc/profile.d/conda.sh
conda activate yeast_ml
python "${HERE}/6_extract_iterative_3arch.py"
