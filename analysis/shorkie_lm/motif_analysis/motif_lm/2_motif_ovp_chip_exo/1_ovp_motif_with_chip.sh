source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"
python 1_ovp_motif_with_chip.py \
    --motif_bed ${WORK_ROOT}/experiments/motif_LM/4_motif_to_tss_dist/tf_modisco_motif_hits_trimmed.bed \
    --chip_peak_bed ${WORK_ROOT}/experiments/motif_LM/2_motif_ovp_chip_exo/ChIP_peaks.bed \
    --output overlap_results.csv
