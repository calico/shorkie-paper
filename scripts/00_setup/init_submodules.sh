#!/usr/bin/env bash
# Initialize the pinned external forks, then print the install commands.
#
# Usage:  bash scripts/00_setup/init_submodules.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git submodule update --init --recursive external/baskerville-yeast external/westminster

echo
echo "Submodules pinned at:"
git submodule status external/baskerville-yeast external/westminster
echo
echo "Next, create the environment and install editable packages:"
echo "  conda env create -f environment.yml && conda activate yeast_ml"
echo "  pip install -e external/baskerville-yeast -e external/westminster -e ."
echo
echo "Then copy and edit the config templates:"
echo "  cp config/paths.example.yaml config/paths.yaml"
echo "  cp config/slurm.example.yaml config/slurm.yaml"
