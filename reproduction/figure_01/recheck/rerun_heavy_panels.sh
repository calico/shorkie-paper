#!/bin/bash
# Figure 1 deep-recheck — Step 3: re-run the heavy external-tool panels from the
# released FASTAs to confirm determinism, and regenerate panel 1C with the
# corrected YJM195 strain representative (was YJM1078).
#
#   1D Mash   : run_mash.sh   -> reproduced/panelD_mash/{saccharomycetales,strains}_dist.tab
#   1C MUMmer : run_mummer.sh -> reproduced/panelC_mummer/*.coords  (now GCA_000975585_2 = YJM195)
#   1B tree   : build_tree.sh -> reproduced/panelB_tree/species_tree.nwk
#
# After this, recheck/diff_heavy_panels.py normalizes the symlinked-path header
# and diffs the regenerated tables vs the committed ones (byte-identity modulo
# path prefix), and records per-panel determinism.
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yeast_ml

REPO="/scratch4/ssalzbe1/khchao/shorkie-paper"
cd "$REPO"

echo "############ [1/3] Mash (panel 1D) ############"
bash reproduction/figure_01/panels/run_mash.sh

echo "############ [2/3] MUMmer (panel 1C, YJM195 fix) ############"
bash reproduction/figure_01/panels/run_mummer.sh

echo "############ [3/3] Phylogeny tree (panel 1B) ############"
bash reproduction/figure_01/panels/build_tree.sh

echo "############ DONE — heavy panels regenerated ############"
ls -la reproduction/figure_01/reproduced/panelC_mummer/*.coords
ls -la reproduction/figure_01/reproduced/panelD_mash/*.tab
