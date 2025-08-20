import os
import re
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import scipy.stats as stats
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Genomics and custom modules
import pyranges as pr
import pysam
import pyfaidx
from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dataset
from yeast_helpers import *
from load_cov import *

# Set TensorFlow logging to error and disable GPU usage
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

############################################################
# Helper Functions
############################################################

def map_chromosome_to_roman(chromosome):
    """Map chromosome names to Roman numeral style with a 'chr' prefix."""
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
        'I': 'chrI',
        'II': 'chrII',
        'III': 'chrIII',
        'IV': 'chrIV',
        'V': 'chrV',
        'VI': 'chrVI',
        'VII': 'chrVII',
        'VIII': 'chrVIII',
        'IX': 'chrIX',
        'X': 'chrX',
        'XI': 'chrXI',
        'XII': 'chrXII',
        'XIII': 'chrXIII',
        'XIV': 'chrXIV',
        'XV': 'chrXV',
        'XVI': 'chrXVI',
    }
    return roman_mapping.get(chromosome, chromosome)

def load_scores(file_path, score):
    """
    Load scores (e.g. 'logSED') from an HDF5 file.
    Returns a dictionary: {score: { "position_gene": mean_value, ... } }
    """
    scores_data = {}
    try:
        with h5py.File(file_path, 'r') as hdf:
            if score not in hdf:
                raise KeyError(f"Score '{score}' not found in {file_path}")
            score_data = {}
            score_group = hdf[score]
            for position in score_group.keys():
                position_group = score_group[position]
                for gene in position_group.keys():
                    key = f"{position}_{gene}"
                    score_data[key] = np.mean(position_group[gene][:])
            scores_data[score] = score_data
    except Exception as e:
        print(f"Error loading scores from {file_path}: {e}")
    return scores_data

def compute_sign_checked_mean(row, fold_cols, tolerance=1):
    """
    Computes the average of fold scores if the signs of the values agree.
    Tolerates only one difference: if more than one fold score deviates
    from the majority sign, returns 0; otherwise returns the mean.
    """
    values = row[fold_cols].dropna()
    if len(values) == 0:
        return 0
    pos_count = (values > 0).sum()
    neg_count = (values < 0).sum()
    if pos_count >= neg_count:
        if neg_count > tolerance:
            return 0
    else:
        if pos_count > tolerance:
            return 0
    return values.mean()

def load_eqtl_data(eqtl_tsv):
    """
    Loads and preprocesses the positive eQTL TSV file.
    Maps chromosomes to Roman numeral style and creates a unique identifier.
    Expects columns: Chr, ChrPos, Pheno, SnpWeight, Reference, Alternate.
    """
    try:
        df_eqtl = pd.read_csv(eqtl_tsv, sep='\t')
    except Exception as e:
        print(f"Error reading TSV file {eqtl_tsv}: {e}")
        return None

    df_eqtl['Chr'] = df_eqtl['Chr'].apply(map_chromosome_to_roman)
    df_eqtl['position'] = df_eqtl['Chr'] + ':' + df_eqtl['ChrPos'].astype(str)
    df_eqtl['Position_Gene'] = df_eqtl['position'] + '_' + df_eqtl['Pheno']
    df_eqtl = df_eqtl[['Position_Gene', 'SnpWeight', 'Reference', 'Alternate']]
    return df_eqtl

def process_fold_data(fold, score, df_eqtl):
    """
    Loads fold-specific score data (from an HDF5 file) and merges it with
    the positive eQTL DataFrame. Computes the Pearson correlation.
    Returns the merged DataFrame and a tuple (correlation, p-value).
    """
    snp_dir = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/eQTL_exp/eqtls/eqtl_{fold}/'
    score_h5 = os.path.join(snp_dir, 'scores.h5')
    
    score_contents = load_scores(score_h5, score)
    if score not in score_contents:
        print(f"Score '{score}' not found for fold {fold}.")
        return None, (np.nan, np.nan)
    
    fold_dict = score_contents[score]
    fold_score_df = pd.DataFrame(list(fold_dict.items()),
                                 columns=['Position_Gene', f'{score}_Avg'])
    merged_df = pd.merge(df_eqtl, fold_score_df, on='Position_Gene', how='inner')
    
    if len(merged_df) > 1:
        corr_val, p_val = stats.pearsonr(merged_df['SnpWeight'], merged_df[f'{score}_Avg'])
    else:
        corr_val, p_val = np.nan, np.nan

    return merged_df, (corr_val, p_val)

def plot_scatter(ax, x, y, xlabel, ylabel, title):
    """Creates a scatter plot on the given axis."""
    ax.scatter(x, y, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

def calculate_tss_distance(key, tss_data):
    """
    Given a key formatted as 'chr:pos_geneID', computes the distance from pos to gene TSS.
    Returns the absolute distance if the chromosome matches, else None.
    """
    parts = key.split('_')
    if len(parts) < 2:
        return None
    gene_id = parts[-1]
    position_part = '_'.join(parts[:-1])
    try:
        chrom, pos = position_part.split(':')[0], int(position_part.split(':')[1])
        gene_info = tss_data.get(gene_id)
        if gene_info is None:
            return None
        gene_chrom = map_chromosome_to_roman(gene_info['chrom'])
        return abs(pos - gene_info['tss']) if gene_chrom == chrom else None
    except Exception as e:
        return None

def gene_within_window(key, tss_data, window=8000):
    """
    Returns True if the gene corresponding to key has length <= window.
    """
    parts = key.split('_')
    if len(parts) < 2:
        return False
    gene_id = parts[-1]
    gene_info = tss_data.get(gene_id)
    if gene_info is None:
        return False
    gene_length = gene_info['end'] - gene_info['start']
    return gene_length <= window

def get_valid_distance_bin(key, tss_data, bins, labels):
    """Assign a distance bin to the key using the computed TSS distance."""
    distance = calculate_tss_distance(key, tss_data)
    if distance is None or np.isnan(distance):
        return None
    try:
        bin_val = pd.cut([distance], bins=bins, labels=labels)[0]
        return bin_val
    except:
        return None

############################################################
# Main Script
############################################################

def main():
    # Set up paths and directories
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    eqtl_tsv = os.path.join(root_dir, 'data', 'eQTL', 'selected_eQTL', 'intersected_data_CIS.tsv')
    gtf_file = os.path.join(root_dir, 'data', 'eQTL', 'neg_eQTLS', 'GCA_000146045_2.59.gtf')
    output_dir = './viz'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'score_dist'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'roc_pr'), exist_ok=True)
    
    # Load and preprocess positive eQTL data
    df_eqtl = load_eqtl_data(eqtl_tsv)
    if df_eqtl is None:
        return

    # Parse gene information from the GTF file
    def parse_gtf_for_tss(gtf_file):
        tss = {}
        with open(gtf_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                chrom = fields[0]
                feature = fields[2].lower()
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                attr = fields[8]
                m = re.search(r'gene_id\s+"([^"]+)"', attr)
                if m and feature == 'gene':
                    gene_id = m.group(1)
                    tss[gene_id] = {'chrom': chrom, 'tss': start if strand=='+' else end, 'start': start, 'end': end}
        return tss

    tss_data = parse_gtf_for_tss(gtf_file)

    # Define eight folds and four negative sets per fold
    folds = [f'f{i}c0' for i in range(8)]
    # neg_sets = [1, 2, 3, 4]
    neg_sets = [1]
    score = 'logSED'

    ##########################################################
    # Process positive eQTLs per fold (preserving per-fold resolution)
    ##########################################################
    pos_master = df_eqtl[['Position_Gene', 'SnpWeight']].copy()
    # Merge each fold's score into pos_master (wide format)
    for fold in folds:
        merged_df, _ = process_fold_data(fold, score, df_eqtl)
        if merged_df is None:
            print(f"Skipping fold {fold} due to errors.")
            continue
        col_name = f'{score}_{fold}'
        temp_df = merged_df[['Position_Gene', f'{score}_Avg']].copy()
        temp_df.rename(columns={f'{score}_Avg': col_name}, inplace=True)
        pos_master = pd.merge(pos_master, temp_df, on='Position_Gene', how='left')
    # Convert wide-format to long-format so that each positive has a per-fold row
    fold_cols = [f'{score}_{fold}' for fold in folds]
    pos_long = pd.melt(pos_master, 
                       id_vars=['Position_Gene','SnpWeight'], 
                       value_vars=fold_cols, 
                       var_name='fold', 
                       value_name='score')
    pos_long['fold'] = pos_long['fold'].str.replace(f'{score}_', '')
    # Compute distance and assign bins for positives
    pos_long['distance'] = pos_long['Position_Gene'].apply(lambda k: calculate_tss_distance(k, tss_data))
    pos_long = pos_long.dropna(subset=['distance'])
    pos_long['distance_bin'] = pd.cut(pos_long['distance'], 
                                      bins=[0,600,1000,2000,3000,5000,8000,np.inf],
                                      labels=['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb','>8kb'])
    pos_long = pos_long[pos_long['distance_bin'] != '>8kb']
    pos_long['label'] = 1
    pos_long['label_type'] = 'Positive'

    ##########################################################
    # Plot positive data per fold (using long format)
    ##########################################################
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    axes = axes.flatten()
    unique_pos_folds = sorted(pos_long['fold'].unique())
    for i, fold in enumerate(unique_pos_folds):
        df_fold = pos_long[pos_long['fold'] == fold]
        title = f'Positive Fold {fold}'
        plot_scatter(axes[i],
                     df_fold['SnpWeight'],
                     df_fold['score'],
                     'SnpWeight',
                     score,
                     title)
    # Also plot average across folds (using the average of per-fold scores from pos_master)
    avg_df = pos_master.copy()
    avg_df['score'] = avg_df[fold_cols].mean(axis=1)
    avg_df['distance'] = avg_df['Position_Gene'].apply(lambda k: calculate_tss_distance(k, tss_data))
    avg_df = avg_df.dropna(subset=['distance'])
    avg_df['distance_bin'] = pd.cut(avg_df['distance'], 
                                    bins=[0,600,1000,2000,3000,5000,8000,np.inf],
                                    labels=['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb','>8kb'])
    avg_df = avg_df[avg_df['distance_bin'] != '>8kb']
    plot_scatter(axes[8],
                 avg_df['SnpWeight'],
                 avg_df['score'],
                 'SnpWeight',
                 f'Avg {score}',
                 'Average across folds')
    plt.tight_layout()
    pos_fig_path = os.path.join(output_dir, f'positive_by_fold_{score}.png')
    plt.savefig(pos_fig_path, dpi=300)
    plt.close()
    print(f"Positive fold plots saved to {pos_fig_path}")
    print("\nPositive eQTL processing complete.")
    print("shape of pos_long:", pos_long.shape)
    print("head of pos_long:")
    print(pos_long.head())

    ##########################################################
    # Load ALL negatives from all folds and all negative sets (no sampling)
    ##########################################################
    neg_data = []
    for fold in folds:
        for ns in neg_sets:
            neg_h5 = f'./eqtl_neg/{fold}/set{ns}/scores.h5'
            neg_scores = load_scores(neg_h5, score)[score]
            for k, v in neg_scores.items():
                neg_data.append((k, v, fold, ns))
    neg_df = pd.DataFrame([{
        'Position_Gene': k,
        'score': v,
        'fold': fold,
        'neg_set': ns,
        'label': 0
    } for k, v, fold, ns in neg_data]).dropna()
    neg_df['distance'] = neg_df['Position_Gene'].apply(lambda k: calculate_tss_distance(k, tss_data))
    neg_df = neg_df.dropna(subset=['distance'])
    bins = [0,600,1000,2000,3000,5000,8000,np.inf]
    labels = ['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb','>8kb']
    neg_df['distance_bin'] = pd.cut(neg_df['distance'], bins=bins, labels=labels)
    neg_df = neg_df[neg_df['distance_bin'] != '>8kb']
    neg_df['label_type'] = 'Negative'
    # (Negatives retain their 'fold' and 'neg_set')

    print("\nNegative eQTL processing complete.")
    print("head of neg_df:")
    print(neg_df.head())

    ##########################################################
    # Combine positives and negatives for overall comparative analysis
    ##########################################################
    # Use pos_long (long-format positives) to preserve per-fold resolution.
    pos_df = pos_long.copy()
    pos_df['label'] = 1
    pos_df['label_type'] = 'Positive'
    combined_df = pd.concat([pos_df[['Position_Gene','SnpWeight','score','distance','fold','distance_bin','label','label_type']],
                             neg_df[['Position_Gene','score','distance','fold','neg_set','label','distance_bin','label_type']]
                            ], ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    print("head of combined_df:")
    print(combined_df.head())

    ##########################################################
    # Plotting – Histograms per fold (overall)
    ##########################################################
    for fold in sorted(combined_df['fold'].unique()):
        plt.figure(figsize=(5, 3))
        sns.histplot(combined_df[(combined_df['fold'] == fold) & (combined_df['label_type']=='Positive')]['distance'],
                     bins=50, kde=True, color='blue')
        plt.title(f'Positive Distance Distribution - {score} ({fold})')
        plt.savefig(os.path.join(output_dir, f'{score}_pos_distance_{fold}.png'), dpi=300)
        plt.tight_layout()
        plt.close()

        plt.figure(figsize=(5, 3))
        sns.histplot(combined_df[(combined_df['fold'] == fold) & (combined_df['label_type']=='Negative')]['distance'],
                     bins=50, kde=True, color='red')
        plt.title(f'Negative Distance Distribution - {score} ({fold})')
        plt.savefig(os.path.join(output_dir, f'{score}_neg_distance_{fold}.png'), dpi=300)
        plt.tight_layout()
        plt.close()

    ##########################################################
    # Violin Plots per fold (overall)
    ##########################################################
    plt.figure(figsize=(20, 16))
    g = sns.catplot(
        data=combined_df,
        x='distance_bin',
        y='score',
        hue='label_type',
        col='fold',
        kind='violin',
        split=True,
        col_wrap=4,
        height=4,
        aspect=1.5,
        palette={'Positive': 'blue', 'Negative': 'red'}
    )
    g.set_xticklabels(rotation=45)
    plt.suptitle(f'Score Distribution by TSS Distance Bin and Fold - {score}', y=1.02)
    plt.savefig(os.path.join(output_dir, f'{score}_violin_by_fold.png'), dpi=300)
    plt.tight_layout()
    plt.close()

    ##########################################################
    # ROC/PR Analysis by fold (overall)
    ##########################################################
    roc_pr_dir = os.path.join(output_dir, 'roc_pr')
    os.makedirs(roc_pr_dir, exist_ok=True)
    unique_folds = sorted(combined_df['fold'].unique())
    colors = plt.cm.Pastel1(np.linspace(0,1,len(unique_folds)))

    plt.figure(figsize=(8,8))
    roc_aucs = []
    for i, fold in enumerate(unique_folds):
        fold_df = combined_df[combined_df['fold'] == fold].dropna(subset=['score','label'])
        print(f"\tFold {fold} shape: {fold_df.shape}")
        # Print all with label 1  
        print(f"\tFold {fold} label 1 shape: {fold_df[fold_df['label']==1].shape}")
        # Print all with label 0
        print(f"\tFold {fold} label 0 shape: {fold_df[fold_df['label']==0].shape}")

        fpr, tpr, _ = roc_curve(fold_df['label'], np.abs(fold_df['score']))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.7, color=colors[i],
                 label=f'Fold {fold} (AUC = {roc_auc:.2f})')
        roc_aucs.append(roc_auc)
    agg_df = combined_df.groupby('Position_Gene').agg({'label':'first','score':'mean'}).reset_index().dropna(subset=['score','label'])

    print(f"Aggregated data shape: {agg_df.shape}")
    # print all with label 1    
    print(f"Aggregated data label 1 shape: {agg_df[agg_df['label']==1].shape}")
    # print all with label 0
    print(f"Aggregated data label 0 shape: {agg_df[agg_df['label']==0].shape}")

    agg_fpr, agg_tpr, _ = roc_curve(agg_df['label'], np.abs(agg_df['score']))
    agg_roc_auc = auc(agg_fpr, agg_tpr)
    plt.plot(agg_fpr, agg_tpr, 'k--', lw=2.5, label=f'Mean ROC (AUC = {agg_roc_auc:.2f})')
    plt.plot([0,1],[0,1],'r--',lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves by Fold - {score}')
    plt.axis('square')
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(roc_pr_dir, f'{score}_roc_curves.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8,8))
    for i, fold in enumerate(unique_folds):
        fold_df = combined_df[combined_df['fold'] == fold].dropna(subset=['score','label'])
        precision, recall, _ = precision_recall_curve(fold_df['label'], np.abs(fold_df['score']))
        pr_auc = average_precision_score(fold_df['label'], np.abs(fold_df['score']))
        plt.plot(recall, precision, lw=1, alpha=0.7, color=colors[i],
                 label=f'Fold {fold} (AP = {pr_auc:.2f})')
    agg_precision, agg_recall, _ = precision_recall_curve(agg_df['label'], np.abs(agg_df['score']))
    agg_pr_auc = average_precision_score(agg_df['label'], np.abs(agg_df['score']))
    plt.plot(agg_recall, agg_precision, 'k--', lw=2.5, label=f'Mean PR (AP = {agg_pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves by Fold - {score}')
    plt.axis('square')
    plt.ylim(-0.05, 1.1)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(roc_pr_dir, f'{score}_pr_curves.png'), dpi=300)
    plt.close()

    ##########################################################
    # Additional Analysis by Negative Set
    # For each neg_set, combine positives with negatives from that set,
    # then compute per-fold curves and also compute an aggregated curve by
    # grouping by Position_Gene (i.e. averaging the scores across folds)
    ##########################################################
    base_neg_dir = os.path.join(output_dir, "negset_analysis")
    os.makedirs(base_neg_dir, exist_ok=True)
    for ns in neg_sets:
        negset_dir = os.path.join(base_neg_dir, f"negset_{ns}")
        os.makedirs(negset_dir, exist_ok=True)
        hist_dir = os.path.join(negset_dir, "histograms")
        violin_dir = os.path.join(negset_dir, "violin")
        roc_dir = os.path.join(negset_dir, "roc")
        pr_dir = os.path.join(negset_dir, "pr")
        os.makedirs(hist_dir, exist_ok=True)
        os.makedirs(violin_dir, exist_ok=True)
        os.makedirs(roc_dir, exist_ok=True)
        os.makedirs(pr_dir, exist_ok=True)
        
        # Combine positives (from pos_long) with negatives for this neg_set.
        pos_sub = pos_long.copy()
        pos_sub['neg_set'] = 'all'  # positives are common
        neg_sub = neg_df[neg_df['neg_set'] == ns].copy()
        combined_sub = pd.concat([pos_sub, neg_sub], ignore_index=True)
        
        # Plot histograms by fold for this negative set
        for fold in sorted(combined_sub['fold'].unique()):
            plt.figure(figsize=(5, 3))
            sns.histplot(combined_sub[(combined_sub['fold'] == fold) & (combined_sub['label_type']=='Positive')]['distance'],
                         bins=50, kde=True, color='blue')
            plt.title(f'Positive Distance Distribution (NegSet {ns}) - Fold {fold}')
            plt.savefig(os.path.join(hist_dir, f'pos_distance_{fold}_negset{ns}.png'), dpi=300)
            plt.tight_layout()
            plt.close()

            plt.figure(figsize=(5, 3))
            sns.histplot(combined_sub[(combined_sub['fold'] == fold) & (combined_sub['label_type']=='Negative')]['distance'],
                         bins=50, kde=True, color='red')
            plt.title(f'Negative Distance Distribution (NegSet {ns}) - Fold {fold}')
            plt.savefig(os.path.join(hist_dir, f'neg_distance_{fold}_negset{ns}.png'), dpi=300)
            plt.tight_layout()
            plt.close()

        # Violin plots by fold for this negative set
        plt.figure(figsize=(20, 16))
        g = sns.catplot(
            data=combined_sub,
            x='distance_bin',
            y='score',
            hue='label_type',
            col='fold',
            kind='violin',
            split=True,
            col_wrap=4,
            height=4,
            aspect=1.5,
            palette={'Positive': 'blue', 'Negative': 'red'}
        )
        g.set_xticklabels(rotation=45)
        plt.suptitle(f'Score Distribution by TSS Distance Bin and Fold (NegSet {ns}) - {score}', y=1.02)
        plt.savefig(os.path.join(violin_dir, f'{score}_violin_by_fold_negset{ns}.png'), dpi=300)
        plt.tight_layout()
        plt.close()

        # ROC/PR curves by fold for this negative set
        plt.figure(figsize=(8,8))
        negset_folds = sorted(combined_sub['fold'].unique())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(negset_folds)))
        roc_aucs = []
        for i, fold in enumerate(negset_folds):
            fold_df = combined_sub[combined_sub['fold'] == fold].dropna(subset=['score','label'])
            if fold_df.empty:
                continue
            fpr, tpr, _ = roc_curve(fold_df['label'], np.abs(fold_df['score']))
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, alpha=0.7, color=colors[i],
                     label=f'Fold {fold} (AUC = {roc_auc:.2f})')
            roc_aucs.append(roc_auc)
        # Compute aggregated results: average the score across folds by grouping by Position_Gene
        agg_df = combined_sub.groupby('Position_Gene').agg({'label':'first','score':'mean'}).reset_index().dropna(subset=['score','label'])

        print(f"Aggregated data shape (NegSet {ns}): {agg_df.shape}")
        # print all with label 1
        print(f"Aggregated data label 1 shape (NegSet {ns}): {agg_df[agg_df['label']==1].shape}")
        # print all with label 0
        print(f"Aggregated data label 0 shape (NegSet {ns}): {agg_df[agg_df['label']==0].shape}")

        agg_fpr, agg_tpr, _ = roc_curve(agg_df['label'], np.abs(agg_df['score']))
        agg_roc_auc = auc(agg_fpr, agg_tpr)
        plt.plot(agg_fpr, agg_tpr, 'k--', lw=2.5, label=f'Aggregated ROC (AUC = {agg_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'r--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves (NegSet {ns}) - {score}')
        plt.axis('square')
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(roc_dir, f'{score}_roc_curves_negset{ns}.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(8,8))
        for i, fold in enumerate(negset_folds):
            fold_df = combined_sub[combined_sub['fold'] == fold].dropna(subset=['score','label'])
            if fold_df.empty:
                continue
            precision, recall, _ = precision_recall_curve(fold_df['label'], np.abs(fold_df['score']))
            pr_auc = average_precision_score(fold_df['label'], np.abs(fold_df['score']))
            plt.plot(recall, precision, lw=1, alpha=0.7, color=colors[i],
                     label=f'Fold {fold} (AP = {pr_auc:.2f})')
        agg_precision, agg_recall, _ = precision_recall_curve(agg_df['label'], np.abs(agg_df['score']))
        agg_pr_auc = average_precision_score(agg_df['label'], np.abs(agg_df['score']))
        plt.plot(agg_recall, agg_precision, 'k--', lw=2.5, label=f'Aggregated PR (AP = {agg_pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves (NegSet {ns}) - {score}')
        plt.axis('square')
        plt.ylim(-0.05, 1.1)
        plt.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(pr_dir, f'{score}_pr_curves_negset{ns}.png'), dpi=300)
        plt.close()

    ##########################################################
    # End of neg_set loop
    ##########################################################
    print("\nDone! All plots saved.")
    
if __name__ == '__main__':
    main()
