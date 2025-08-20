# python 1_time_track_metrics.py \
#     --targets /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/gene_level_eval_rc/f0c0/RNA-Seq/gene_targets.tsv \
#     --preds   /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/gene_level_eval_rc/f0c0/RNA-Seq/gene_preds.tsv \
#     --out     gene_metrics.tsv

# TF=MSN2
# gene=YCL040W # test
# gene=YBR139W # train

# TF=MSN4
# gene=YML100W # TSL1
# gene=YIL124W # AYR1

TF=MET4
# gene=YML100W
# gene=YER177W #BMH1
gene=YJL060W #BNA3

# TF=MET4

python 1_time_track_metrics_viz.py \
    --data_dirs /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/gene_target_preds/f0c0/RNA-Seq/ \
    --tf ${TF} \
    --gene ${gene} \
    --targets_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt \
    --out eval_${TF}/eval.txt \
    --out_dir eval_${TF}




#     --data_dir /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/gene_target_preds_valid/f0c0/RNA-Seq/ /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/gene_target_preds_test/f0c0/RNA-Seq/ /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/gene_target_preds_train/f0c0/RNA-Seq/ \


    # --data_dir /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/gene_target_preds_test/f0c0/RNA-Seq/ \