grep -v "chrMito" tss.bed > tss_filtered.bed
mkdir -p motif_tss_distance


model_archs=('unet_small_bert_drop' 'unet_small_bert_drop_retry_1' 'unet_small_bert_drop_retry_2')
for model in ${model_archs[@]}; do
    mkdir -p motif_tss_distance/${model}
    for i in {1..80}; do
        motif_file="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/${model}/fimo_out/pos_patterns_pattern_${i}_fwd/motifs.bed"
        
        motif_bg_file="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/${model}/fimo_out/pos_patterns_pattern_${i}_fwd/motifs_bg.bed"

        output_stats="motif_tss_distance/${model}/motif_${i}_bg_tss_stats.txt"

        if [ ! -f ${motif_file} ]; then
            continue
        fi

        echo "Processing motif file: ${motif_file}"
        echo "Processing motif background file: ${motif_bg_file}"

        # # Generate shuffled motifs
        # bedtools shuffle \
        #     -i "${motif_file}" \
        #     -g genome.chrom.sizes \
        #     -chrom \
        # | bedtools sort \
        #     -g genome.chrom.sizes \
        # > "${motif_bg_file}"

        # echo "Shuffled and sorted background file created: ${motif_bg_file}"

        # # Calculate closest distances to TSS
        # bedtools closest \
        # -a ${motif_bg_file} \
        # -b tss_filtered.bed \
        # -D ref \
        # > ${output_stats}

        # echo "Output stats file created: ${output_stats}"
    done
done





