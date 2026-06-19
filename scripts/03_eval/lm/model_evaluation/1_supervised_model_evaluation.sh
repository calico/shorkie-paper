source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"
YEAST_SEQNN_EVAL_ROOT="$(cfg yeast_seqnn_eval_root)"
for idx in {0..7}; do
    echo "============================"
    echo "idx: $idx"
    mkdir -p ${YEAST_SEQNN_EVAL_ROOT}/supervised_unet_big/gene_level_eval/f${idx}c0/
    python ${BASKERVILLE_SCRIPTS}/yeast_test_genes.py ${YEAST_SEQNN_EVAL_ROOT}/supervised_unet_big/params.json ${YEAST_SEQNN_EVAL_ROOT}/supervised_unet_big/train/f${idx}c0/train/model_check.h5 ${YEAST_SEQNN_EVAL_ROOT}/supervised_unet_big/train/f${idx}c0/data0/ ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf \
    --eval_dir ${YEAST_SEQNN_EVAL_ROOT}/supervised_unet_big/train/f${idx}c0/data0/ --no_unclip \
    -o ${YEAST_SEQNN_EVAL_ROOT}/supervised_unet_big/gene_level_eval/f${idx}c0/
    echo "============================"
done