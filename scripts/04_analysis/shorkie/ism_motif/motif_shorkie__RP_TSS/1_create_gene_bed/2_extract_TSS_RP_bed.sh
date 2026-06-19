source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"
python 2_extract_TSS_bed.py -i ${WORK_ROOT}/data/gene_exp_ism_window/TSS_windows_ex_chrmt.bed -o ${WORK_ROOT}/data/gene_exp_ism_window/TSS_windows_trimmed_ex_chrmt.bed

python 2_extract_TSS_bed.py -i ${WORK_ROOT}/data/gene_exp_ism_window/RP_windows.bed -o ${WORK_ROOT}/data/gene_exp_ism_window/RP_windows_trimmed.bed
