#!/usr/bin/env bash
# Regenerate the SpeciesLM (fungi) reconstruction PWM for the SMT3 promoter (Figure 2A top row).
# Needs torch + transformers + a one-time HuggingFace download of johahi/specieslm-fungi-upstream-k1.
# In this repo those live in the `pytorch_cuda` conda env (NOT yeast_ml). CPU is sufficient.
#
#   bash scripts/04_analysis/shorkie_lm/lm_SMT3_viz/0_compute_specieslm_smt3.sh
#
set -euo pipefail

ENV_NAME="${SPECIESLM_ENV:-pytorch_cuda}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source /home/kchao10/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

# Pass work_root through if resolvable (for the optional dependencies_DNALM copy); harmless if empty.
WORK_ROOT="${SHORKIE_WORK_ROOT:-}"
EXTRA=()
[ -n "${WORK_ROOT}" ] && EXTRA+=(--work-root "${WORK_ROOT}")

python "${HERE}/0_compute_specieslm_smt3.py" --device "${SPECIESLM_DEVICE:-cpu}" "${EXTRA[@]}"
