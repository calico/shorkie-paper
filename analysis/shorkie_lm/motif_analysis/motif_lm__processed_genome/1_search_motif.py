import os
import re
import pysam
import pyranges as pr
import numpy as np
import pandas as pd
import sys
from Bio import motifs

def parse_fasta_header(header):
    """
    Parses a FASTA header to extract chromosome, start, end, and species.
    """
    match = re.match(r'>(chr[\w]+):(\d+)-(\d+)\|(\w+)', header)
    return match.groups() if match else (None, None, None, None)


def process_fasta(file_path):
    """
    Processes a FASTA file to extract sequence information into a DataFrame.
    """
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


def main():
    ################################
    # Load predicted LM scores
    ################################
    model_arch=sys.argv[1]
    dataset=sys.argv[2]
    
    # File paths
    output_dir = f'{dataset}_viz_seq/{model_arch}/'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize empty DataFrames and lists
    seqs_df_list = []
    cleaned_seqs_df_list = []
    x_true_list = []
    x_pred_list = []
    label_list = []
    weight_scale_list = []

    # for target_type in ['train', 'test', 'valid']:
    for target_type in ['train']:
        # File paths for each dataset
        sequences_bed_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/sequences_{target_type}.cleaned.bed'
        cleaned_sequences_fasta_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/extracted_fasta/sequences_{target_type}.cleaned.fasta'
        if target_type == 'train':
            predictions_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_{model_arch}/test_trainset_{dataset}/preds_{target_type}.npz'

        # Load sequences BED file
        seqs_df = pd.read_csv(sequences_bed_file, sep='\t', names=['Chromosome', 'Start', 'End', 'label', 'species'])
        seqs_df = seqs_df[seqs_df['species'] == "GCA_000146045_2"].reset_index(drop=True)

        seqs_df = seqs_df[seqs_df['label'] == target_type].reset_index(drop=True)
        seqs_df['row_index'] = seqs_df.index
        seqs_df['Strand'] = "."
        seqs_df = seqs_df[['Chromosome', 'Start', 'End', 'species', 'row_index', 'Strand']]
        seqs_df_list.append(seqs_df)
        print("\tseqs_df: ", len(seqs_df))
    
        # Process cleaned sequences FASTA file
        cleaned_seqs_df = process_fasta(cleaned_sequences_fasta_file)
        cleaned_seqs_df_list.append(cleaned_seqs_df)

        # Load predictions and targets
        cache_bundle = np.load(predictions_file)
        x_true_list.append(cache_bundle['x_true'])
        x_pred_list.append(cache_bundle['x_pred'])
        label_list.append(cache_bundle['label'])
        weight_scale_list.append(cache_bundle['weight_scale'])

    # Concatenate DataFrames and arrays
    seqs_df = pd.concat(seqs_df_list, ignore_index=True)
    cleaned_seqs_df = pd.concat(cleaned_seqs_df_list, ignore_index=True)
    x_true = np.concatenate(x_true_list, axis=0)
    x_pred = np.concatenate(x_pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    weight_scale = np.concatenate(weight_scale_list, axis=0)

    # Count the number of 0s and 1s
    count_zeros = np.sum(weight_scale == 0)
    count_ones = np.sum(weight_scale == 1)

    # Assuming x_pred has a shape of (480, 200, 4)
    x_pred_transformed = np.copy(x_pred)
    x_pred_transformed += 0.0001  # Adding a small value to avoid division by zero
    x_pred_transformed_mean = np.mean(x_pred_transformed, axis=1, keepdims=True)  # Compute mean for each position
    # Apply the normalization method from the screenshot
    x_pred_transformed = x_pred_transformed * np.log(x_pred_transformed / x_pred_transformed_mean)
    x_pred = x_pred_transformed

    x_true = np.transpose(x_true, (0, 2, 1))
    x_pred = np.transpose(x_pred, (0, 2, 1))
    
    # Save x_true and x_pred to an npz file
    x_true_npz_file_path = os.path.join(f'{output_dir}/x_true.npz')
    np.savez(x_true_npz_file_path, arr_0=x_true)

    x_pred_npz_file_path = os.path.join(f'{output_dir}/x_pred.npz')
    np.savez(x_pred_npz_file_path, arr_0=x_pred)

if __name__ == "__main__":
    main()