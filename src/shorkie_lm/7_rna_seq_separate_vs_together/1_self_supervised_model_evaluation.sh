# This script runs through the supervised learning model trained with only RNA-Seq data
for idx in {0..7}; do
    echo "============================"
    echo "idx: $idx"
    mkdir -p /home/khc/projects/yeast_seqNN/self_supervised/exp_105/rna_seq/16bp/self_supervised_unet_big/gene_level_eval/f${idx}c0/
    python /home/khc/projects/baskerville-yeast/src/baskerville/scripts/yeast_test_genes.py /home/khc/projects/yeast_seqNN/self_supervised/exp_105/rna_seq/16bp/self_supervised_unet_big/params.json /home/khc/projects/yeast_seqNN/self_supervised/exp_105/rna_seq/16bp/self_supervised_unet_big/train/f${idx}c0/train/model_check.h5 /home/khc/projects/yeast_seqNN/self_supervised/exp_105/rna_seq/16bp/self_supervised_unet_big/train/f${idx}c0/data0/ /scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf \
    --eval_dir /home/khc/projects/yeast_seqNN/self_supervised/exp_105/rna_seq/16bp/self_supervised_unet_big/train/f${idx}c0/data0/ --no_unclip \
    -o /home/khc/projects/yeast_seqNN/self_supervised/exp_105/rna_seq/16bp/self_supervised_unet_big/gene_level_eval/f${idx}c0/ 
    echo "============================"
done