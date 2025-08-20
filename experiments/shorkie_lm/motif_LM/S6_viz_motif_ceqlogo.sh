# # Define the model architectures
# model_archs=('unet_small_bert_aux_drop' 'unet_small')
# # model_archs=('unet_small')

# # Get the current model architecture based on the array index
# model=${model_archs[$SLURM_ARRAY_TASK_ID]}

model='unet_small_bert_aux_drop'
# Define input file paths based on the model architecture
input_file="saccharomycetales_viz_seq/averaged_models/${model}/modisco_results.log"
output_file="saccharomycetales_viz_seq/averaged_models/${model}/report_merge/"

# Run TF-MoDISco
for i in $(seq 1 23);
do
    ceqlogo -i ${input_file} -m pattern_${i} -o logo_${i}.png -f PNG 
done
