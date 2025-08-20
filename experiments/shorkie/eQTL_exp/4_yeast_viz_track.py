import json
import os
import re
import time

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pysam
import pyfaidx
import tensorflow as tf
import seaborn as sns
import scipy.stats as stats

import pyranges as pr

from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dataset
from yeast_helpers import *
from load_cov import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


##############################
# Helper Functions
##############################

def map_chromosome_to_roman(chromosome):
    roman_mapping = {
        'chromosome1': 'chrI',
        'chromosome2': 'chrII',
        'chromosome3': 'chrIII',
        'chromosome4': 'chrIV',
        'chromosome5': 'chrV',
        'chromosome6': 'chrVI',
        'chromosome7': 'chrVII',
        'chromosome8': 'chrVIII',
        'chromosome9': 'chrIX',
        'chromosome10': 'chrX',
        'chromosome11': 'chrXI',
        'chromosome12': 'chrXII',
        'chromosome13': 'chrXIII',
        'chromosome14': 'chrXIV',
        'chromosome15': 'chrXV',
        'chromosome16': 'chrXVI',
    }
    return roman_mapping.get(chromosome, chromosome)


def load_scores(h5_file, score):
    """
    Load scores (e.g. 'logSUM') from an HDF5 file.
    Returns a dictionary mapping "position_gene" to the average score.
    """
    scores_data = {}
    with h5py.File(h5_file, 'r') as hdf:
        score_group = hdf[score]
        for position in score_group.keys():
            position_group = score_group[position]
            for gene in position_group.keys():
                key = f"{position}_{gene}"
                scores_data[key] = np.mean(position_group[gene][:])
    return {score: scores_data}


def load_scores_multifold(score, num_folds, root_dir):
    """
    Loop through each fold directory to load scores and then average across folds.
    Assumes each fold is in a subfolder e.g., <root_dir>/experiments/eQTL_exp/eqtls/eqtl_f0c0/, etc.
    """
    all_fold_dfs = []
    folds = [f'f{i}c0' for i in range(num_folds)]
    for fold in folds:
        snp_dir = os.path.join(root_dir, 'experiments', 'SUM_data_process', 'eQTL_exp', 'eqtls', f'eqtl_{fold}')
        score_h5 = os.path.join(snp_dir, 'scores.h5')
        score_contents = load_scores(score_h5, score)
        fold_dict = score_contents[score]
        fold_score_df = pd.DataFrame(list(fold_dict.items()), columns=['Position_Gene', f'{score}_Avg'])
        fold_score_df['Fold'] = fold
        all_fold_dfs.append(fold_score_df)
    combined_df = pd.concat(all_fold_dfs)
    avg_df = combined_df.groupby('Position_Gene', as_index=False).agg({f'{score}_Avg': 'mean'})
    return avg_df


##############################
# Updated Track Visualization Function
##############################

def plot_cov_plot(search_gene, gene, gene_start, gene_end, center_pos, chrom, poses, alts, seq_len,
                  seqnn_model, models, fasta_open, condition, targets_df, logSED_avg, snpweight, bin_window=400):
    print("* center_pos:", center_pos)
    print("* chrom:", chrom)
    print("* poses:", poses)
    print("* gene_start:", gene_start)
    print("* gene_end:", gene_end)

    # Determine sequence window
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    seq_out_start = start + seqnn_model.model_strides[0] * seqnn_model.target_crops[0]
    seq_out_len = seqnn_model.model_strides[0] * seqnn_model.target_lengths[0]

    # Determine output positions for gene exons
    gene_slice = gene.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)
    sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)
    sequence_one_hot_mut = np.copy(sequence_one_hot_wt)

    # Introduce the SNP (mutation)
    for pos, alt in zip(poses, alts):
        alt_ix = -1
        if alt == 'A':
            alt_ix = 0
        elif alt == 'C':
            alt_ix = 1
        elif alt == 'G':
            alt_ix = 2
        elif alt == 'T':
            alt_ix = 3
        sequence_one_hot_mut[pos - start - 1] = 0.
        sequence_one_hot_mut[pos - start - 1, alt_ix] = 1.

    # ---- Ensemble Prediction: Average across eight models (model averaging only) ----
    def predict_tracks_ensemble(model_list, sequence):
        preds = [predict_tracks([model], sequence) for model in model_list]
        return np.mean(np.array(preds), axis=0)  # retains shape (1,1,length, num_tracks)

    y_wt = predict_tracks_ensemble(models, sequence_one_hot_wt)
    y_mut = predict_tracks_ensemble(models, sequence_one_hot_mut)
    print("y_wt.shape:", y_wt.shape)
    print("y_mut.shape:", y_mut.shape)
    print("gene_slice:", gene_slice)

    # ---- Set plotting parameters ----
    plot_window = 16384 - 2 * 64 * 16
    bin_size = 16
    pad = 64
    rescale_tracks = True
    normalize_counts = False
    anno_df = None  # e.g., exon annotation dataframe if needed
    save_figs = True
    save_suffix = f'_chr{chrom}_{center_pos}'
    save_dir = os.path.join('./viz_tracks_zoom', condition)
    os.makedirs(save_dir, exist_ok=True)

    # ---- Determine the number of predicted tracks from model output ----
    num_pred_tracks = y_wt.shape[-1]
    # Instead of pre-averaging, let the plotting function average over the full track dimension.
    track_indices = [list(range(num_pred_tracks))]
    track_names = ['Average']
    track_scales = [1]
    track_transforms = [1]
    clip_softs = [384.]

    # ---- Build reference coverage from all tracks ----
    y_cov_list = []
    for tname in targets_df['identifier'].tolist():
        row = targets_df[targets_df['identifier'] == tname]
        if row.empty:
            print("Track not found in targets file:", tname)
            continue
        bw_f = row.iloc[0]['file']
        print("Using coverage file:", bw_f)
        cov_values = read_coverage(bw_f, chrom, start + 1024, end - 1024)
        y_ref_cov = seq_norm(cov_values)
        y_cov_list.append(y_ref_cov)
    if len(y_cov_list) == 0:
        print("No coverage tracks found, exiting visualization.")
        return
    # Stack coverage to shape (length, num_tracks)
    y_ref_cov = np.array(y_cov_list).T
    # Add batch and channel dimensions: (1, 1, length, num_tracks)
    y_ref_cov = np.expand_dims(y_ref_cov, axis=0)
    y_ref_cov = np.expand_dims(y_ref_cov, axis=0)
    # For ground truth, use all track indices
    ref_track_indices = [list(range(y_ref_cov.shape[-1]))]

    # ---- Call the provided plotting routine ----
    # Note: plot_coverage_track_pair_bins_w_ref will average over the track dimension internally.
    plot_coverage_track_pair_bins_w_ref_zoomin(
        y_wt,
        y_mut,
        chrom,
        start,
        search_gene,
        center_pos,
        gene_start,
        gene_end,
        poses,
        track_indices,
        track_names,
        track_scales,
        track_transforms,
        clip_softs,
        logSED_avg,
        snpweight,
        y_ground_truth=y_ref_cov,
        ref_track_indices=ref_track_indices,
        bin_window=bin_window,            # <-- new argument here
        plot_mut=True,
        plot_window=plot_window,
        normalize_window=plot_window,
        bin_size=bin_size,
        pad=pad,
        rescale_tracks=rescale_tracks,
        normalize_counts=normalize_counts,
        save_figs=save_figs,
        save_suffix=save_suffix,
        gene_slice=gene_slice,
        anno_df=anno_df,
        save_dir=save_dir,
    )


##############################
# Main Script
##############################

def main():
    # === File paths and parameters ===
    num_folds = 1
    score = 'logSED'
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    eqtl_tsv = os.path.join(root_dir, 'data/eQTL/selected_eQTL/intersected_data_CIS.tsv')
    output_dir = './viz_tracks_zoom'
    os.makedirs(output_dir, exist_ok=True)

    # Files needed for track visualization/model
    params_file = os.path.join(root_dir, 'seq_experiment',
                               'exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp',
                               'self_supervised_unet_small_bert_drop/train', 'f0c0', 'train/params.json')
    targets_file = os.path.join(root_dir, 'seq_experiment',
                                'exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp',
                                'cleaned_sheet_RNA-Seq_T0.txt')
    gtf_file = os.path.join(root_dir, 'data/yeast/ensembl_fungi_59',
                            'test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI',
                            'data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf')
    fasta_file = os.path.join(root_dir, 'data/yeast/ensembl_fungi_59',
                              'test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI',
                              'data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta')

    # === Load eQTL Data ===
    df_eqtl = pd.read_csv(eqtl_tsv, sep='\t')
    df_eqtl['Chr'] = df_eqtl['Chr'].apply(map_chromosome_to_roman)
    df_eqtl['position'] = df_eqtl['Chr'] + ':' + df_eqtl['ChrPos'].astype(str)
    df_eqtl['Position_Gene'] = df_eqtl['position'] + '_' + df_eqtl['Pheno']

    # === Load and average scores from eight folds ===
    score_df = load_scores_multifold(score, num_folds, root_dir)
    merged_df = pd.merge(df_eqtl, score_df, on='Position_Gene')

    # === Select extreme eQTLs ===
    top_threshold = merged_df[f'{score}_Avg'].quantile(0.9)
    bottom_threshold = merged_df[f'{score}_Avg'].quantile(0.1)
    # top_threshold = merged_df[f'{score}_Avg'].quantile(0.995)
    # bottom_threshold = merged_df[f'{score}_Avg'].quantile(0.005)

    upregulated_df = merged_df[merged_df[f'{score}_Avg'] >= top_threshold]
    downregulated_df = merged_df[merged_df[f'{score}_Avg'] <= bottom_threshold]

    print(f"Found {len(upregulated_df)} upregulated and {len(downregulated_df)} downregulated eQTLs.")

    # === Load model parameters, targets, transcriptome, and fasta ===
    with open(params_file) as pf:
        params = json.load(pf)
    params_model = params['model']
    params_train = params['train']

    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
    target_index = targets_df.index
    print("T0 targets track number:", len(target_index))

    # ---- Load eight fold models for the ensemble ----
    rc = True
    num_species = 165
    params_model["num_features"] = num_species + 5
    models = []
    for fold in range(num_folds):
        fold_param = f"f{fold}c0"
        model_file = os.path.join(root_dir, 'seq_experiment',
                                  'exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp',
                                  'self_supervised_unet_small_bert_drop/train', fold_param,
                                  'train/model_best.h5')
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, trunk=False)
        seqnn_model.build_slice(target_index)
        seqnn_model.build_ensemble(rc, [0])
        models.append(seqnn_model)
    
    # For visualization, we use the first model to access model attributes (e.g. strides)
    plot_model = models[0]

    fasta_open = pysam.Fastafile(fasta_file)
    transcriptome = bgene.Transcriptome(gtf_file)

    # === Function to process a single eQTL row and plot tracks ===
    def process_eqtl_row(row, condition):
        pos = row['Position_Gene']
        alt = row['Alternate']
        logSED_avg = row[f'{score}_Avg']
        snpweight = row['SnpWeight']
        try:
            center_part = pos.split(':')[1]
            center_pos, search_gene = center_part.split('_')
            center_pos = int(center_pos)
        except Exception as e:
            print("Error parsing Position_Gene:", pos)
            return

        gene_keys = [gkey for gkey in transcriptome.genes.keys() if search_gene in gkey]
        if len(gene_keys) == 0:
            print("No matching gene found for:", search_gene)
            return
        gene = transcriptome.genes[gene_keys[0]]
        gene_exons = gene.get_exons()
        if not gene_exons:
            print("No exon info for gene:", search_gene)
            return
        gene_start = gene_exons[0][0]
        gene_end = gene_exons[-1][1]

        chrom = gene.chrom
        poses = [center_pos]
        alts = [alt]

        # Call the track visualization function with ensemble models and averaged tracks
        plot_cov_plot(search_gene, gene, gene_start, gene_end, center_pos, chrom,
                      poses, alts, 16384, plot_model, models, fasta_open,
                      condition, targets_df, logSED_avg, snpweight)
        print(f"Plotted track for gene: {search_gene} on {chrom} at position {center_pos}")

    # === Process one extreme eQTL per condition (adjust loop as needed) ===
    upregulated_df.to_csv(os.path.join(output_dir, 'upregulated_df.csv'), index=False)
    downregulated_df.to_csv(os.path.join(output_dir, 'downregulated_df.csv'), index=False)

    # counter = 3
    for idx, row in upregulated_df.iterrows():
        process_eqtl_row(row, condition='up')
        # counter -= 1
        # if counter == 0:
        #     break   
    
    # counter = 3
    for idx, row in downregulated_df.iterrows():
        process_eqtl_row(row, condition='down')
        # counter -= 1
        # if counter == 0:
        #     break

if __name__ == '__main__':
    main()
