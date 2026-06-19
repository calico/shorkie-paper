source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"
python 1_modisco_motif_on_chrom.py ${WORK_ROOT}/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n5000.h5 ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf output.png
