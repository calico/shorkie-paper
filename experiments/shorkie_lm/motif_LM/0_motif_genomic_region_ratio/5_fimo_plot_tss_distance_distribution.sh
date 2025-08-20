model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
for model in ${model_archs[@]}; do
    mkdir -p motif_tss_distance_viz/${model}
    python 5_plot_tss_distance_distribution.py motif_tss_distance/${model} motif_tss_distance_viz/${model}
done


