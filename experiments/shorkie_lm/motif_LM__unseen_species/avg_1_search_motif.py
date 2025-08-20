import os
import re
import pysam
import pyranges as pr
import numpy as np
import pandas as pd
import sys
from Bio import motifs

def parse_fasta_header(header):
    match = re.match(r'>(chr[\w]+):(\d+)-(\d+)\|(\w+)', header)
    return match.groups() if match else (None, None, None, None)

def process_fasta(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    chromosome, start, end, species = parse_fasta_header(line.strip())
                    if chromosome:
                        data.append([chromosome, int(start), int(end), species])
        return pd.DataFrame(data, columns=['Chromosome', 'Start', 'End', 'species'])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame(columns=['Chromosome', 'Start', 'End', 'species'])

def process_single_model(model_arch, dataset='saccharomycetales'):
    print("Processing model architecture: ", model_arch)
    # File paths
    seqs_df_list = []
    cleaned_seqs_df_list = []
    x_true_list = []
    x_pred_list = []
    label_list = []
    weight_scale_list = []

    for target_type in ['train']:
        sequences_bed_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/sequences_{target_type}.cleaned.bed'
        cleaned_sequences_fasta_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/extracted_fasta/sequences_{target_type}.cleaned.fasta'
        if target_type == 'train':
            predictions_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_{model_arch}/test_trainset_{dataset}/preds_{target_type}.npz'

        print(f"\t>> Processing {target_type} dataset")
        print(f"\t>> predictions_file: {predictions_file}")

        # Load sequences BED file
        seqs_df = pd.read_csv(sequences_bed_file, sep='\t', names=['Chromosome', 'Start', 'End', 'label', 'species'])
        seqs_df = seqs_df[seqs_df['species'] == "GCA_000146045_2"].reset_index(drop=True)
        seqs_df = seqs_df[seqs_df['label'] == target_type].reset_index(drop=True)
        seqs_df['row_index'] = seqs_df.index
        seqs_df['Strand'] = "."
        seqs_df = seqs_df[['Chromosome', 'Start', 'End', 'species', 'row_index', 'Strand']]
        seqs_df_list.append(seqs_df)

        # Process cleaned sequences FASTA file
        cleaned_seqs_df = process_fasta(cleaned_sequences_fasta_file)
        cleaned_seqs_df_list.append(cleaned_seqs_df)

        # Load predictions and targets
        cache_bundle = np.load(predictions_file)
        x_true_list.append(cache_bundle['x_true'])
        x_pred_list.append(cache_bundle['x_pred'])
        label_list.append(cache_bundle['label'])
        weight_scale_list.append(cache_bundle['weight_scale'])

    # Concatenate across target types
    seqs_df = pd.concat(seqs_df_list, ignore_index=True)
    cleaned_seqs_df = pd.concat(cleaned_seqs_df_list, ignore_index=True)

    x_true = np.concatenate(x_true_list, axis=0)
    x_pred = np.concatenate(x_pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    weight_scale = np.concatenate(weight_scale_list, axis=0)

    return x_true, x_pred, label, weight_scale

def main():
    # dataset = 'schizosaccharomycetales'
    dataset = 'strains_select'
    if sys.argv[1] == "unet_small_bert_aux_drop":
        model_archs = ['unet_small_bert_aux_drop', 'unet_small_bert_aux_drop_retry_1', 'unet_small_bert_aux_drop_retry_2']
    elif sys.argv[1] == "unet_small":

        model_archs = ['unet_small', 'unet_small_retry_1', 'unet_small_retry_2']

    x_true_all = []
    x_pred_all = []
    label_all = []
    weight_scale_all = []

    for model_arch in model_archs:
        x_true, x_pred, label, weight_scale = process_single_model(model_arch, dataset=dataset)
        x_true_all.append(x_true)
        x_pred_all.append(x_pred)
        label_all.append(label)
        weight_scale_all.append(weight_scale)

    # Ensure that all shapes match
    x_true_all = np.stack(x_true_all, axis=0)       # (3, N, L, 4)
    x_pred_all = np.stack(x_pred_all, axis=0)       # (3, N, L, 4)
    label_all = np.stack(label_all, axis=0)         # (3, N)
    weight_scale_all = np.stack(weight_scale_all, axis=0) # (3, N, L)

    # Average across the three models
    x_true = np.mean(x_true_all, axis=0)
    x_pred = np.mean(x_pred_all, axis=0)
    label = np.mean(label_all, axis=0)
    weight_scale = np.mean(weight_scale_all, axis=0)

    print("After averaging:")
    print("x_true: ", x_true.shape)
    print("x_pred: ", x_pred.shape)
    print("label: ", label.shape)
    print("weight_scale: ", weight_scale.shape)

    # # Now apply the mask logic
    # mask = (weight_scale == 0)
    # mask_expanded = mask[:, :, np.newaxis]
    # mask_expanded = np.repeat(mask_expanded, 4, axis=2)

    # x_true[mask_expanded] = 0.25
    # x_pred[mask_expanded] = 0.25

    # Now apply the transformations after averaging
    x_pred_transformed = np.copy(x_pred)
    x_pred_transformed += 0.0001  
    x_pred_transformed_mean = np.mean(x_pred_transformed, axis=1, keepdims=True)
    x_pred_transformed = x_pred_transformed * np.log(x_pred_transformed / x_pred_transformed_mean)
    x_pred = x_pred_transformed

    # Transpose as before
    x_true = np.transpose(x_true, (0, 2, 1))
    x_pred = np.transpose(x_pred, (0, 2, 1))

    # Prepare output directory
    output_dir = f'{dataset}_viz_seq/averaged_models/{model_archs[0]}/'
    os.makedirs(output_dir, exist_ok=True)

    x_true_npz_file_path = os.path.join(output_dir, 'x_true.npz')
    np.savez(x_true_npz_file_path, arr_0=x_true)

    x_pred_npz_file_path = os.path.join(output_dir, 'x_pred.npz')
    np.savez(x_pred_npz_file_path, arr_0=x_pred)

    print("Averaged and transformed arrays saved.")

if __name__ == "__main__":
    main()
