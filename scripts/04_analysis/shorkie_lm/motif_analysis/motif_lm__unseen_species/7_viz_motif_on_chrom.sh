# Resolve machine paths from config (config/paths.yaml)
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
WORK_ROOT="$(cfg work_root)"
python 7_viz_motif_on_chrom.py ${WORK_ROOT}/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/fimo_out/pos_patterns_pattern_27_fwd/fimo.tsv ${WORK_ROOT}/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/fimo_out/pos_patterns_pattern_29_fwd/fimo.tsv ${WORK_ROOT}/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/fimo_out/pos_patterns_pattern_32_fwd/fimo.tsv ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf output.png
