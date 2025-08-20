grep -v "chrMito" tss.bed > tss_filtered.bed

model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
for model in ${model_archs[@]}; do
    mkdir -p motif_tss_distance/${model}
    for i in {1..80}; do
        motif_file="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/${model}/fimo_out/pos_patterns_pattern_${i}_fwd/motifs.bed"
        if [ ! -f ${motif_file} ]; then
            continue
        fi
        echo "motif_file: ${motif_file}"
        output_stats="motif_tss_distance/${model}/motif_${i}_tss_stats.txt"
        bedtools closest -a ${motif_file} -b tss_filtered.bed -D ref > ${output_stats}
    done
done


