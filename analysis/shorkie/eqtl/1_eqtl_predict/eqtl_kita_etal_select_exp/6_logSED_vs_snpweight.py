import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import scipy.stats as stats
import tensorflow as tf

# Genomics and custom modules
import pyranges as pr
import pysam
import pyfaidx
from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dataset
from yeast_helpers import *
from load_cov import *

# Set TensorFlow logging to error and disable GPU usage (only once)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


############################################################
# 1. Helper Functions
############################################################

def map_chromosome_to_roman(chromosome):
    """Map chromosome names to Roman numeral style."""
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
    It tolerates only one difference: if more than one fold score deviates
    from the majority sign, returns 0. Otherwise, returns the mean of the scores.
    """
    values = row[fold_cols].dropna()
    if len(values) == 0:
        return 0
    pos_count = (values > 0).sum()
    neg_count = (values < 0).sum()
    # Determine majority sign and check tolerance
    if pos_count >= neg_count:
        # Majority positive; tolerate only 'tolerance' negatives
        if neg_count > tolerance:
            return 0
    else:
        # Majority negative; tolerate only 'tolerance' positives
        if pos_count > tolerance:
            return 0
    return values.mean()


def load_eqtl_data(eqtl_tsv):
    """
    Loads and preprocesses the eQTL TSV file.
    Maps chromosomes to Roman numeral style and creates unique identifiers.
    """
    try:
        df_eqtl = pd.read_csv(eqtl_tsv, sep='\t')
    except Exception as e:
        print(f"Error reading TSV file {eqtl_tsv}: {e}")
        return None

    df_eqtl['Chr'] = df_eqtl['Chr'].apply(map_chromosome_to_roman)
    df_eqtl['position'] = df_eqtl['Chr'] + ':' + df_eqtl['ChrPos'].astype(str)
    df_eqtl['Position_Gene'] = df_eqtl['position'] + '_' + df_eqtl['#Gene']
    df_eqtl = df_eqtl[['Position_Gene', 'effectSize', 'Reference', 'Alternate']]
    return df_eqtl


def process_fold_data(fold, score, df_eqtl):
    """
    Loads the fold-specific score data, merges it with the eQTL DataFrame,
    and computes the Pearson correlation.
    Returns the merged DataFrame and a tuple (correlation, p-value).
    """
    snp_dir = f'./eqtls/eqtl_{fold}/'
    score_h5 = os.path.join(snp_dir, 'scores.h5')
    
    score_contents = load_scores(score_h5, score)
    if score not in score_contents:
        print(f"Score '{score}' not found for fold {fold}.")
        return None, (np.nan, np.nan)
    
    fold_dict = score_contents[score]
    fold_score_df = pd.DataFrame(list(fold_dict.items()),
                                 columns=['Position_Gene', f'{score}_Avg'])
    print("fold_score_df: ", fold_score_df)
    merged_df = pd.merge(df_eqtl, fold_score_df, on='Position_Gene', how='inner')
    print("merged_df: ", merged_df)
    
    # Compute Pearson correlation if possible
    if len(merged_df) > 1:
        corr_val, p_val = stats.pearsonr(merged_df['effectSize'], merged_df[f'{score}_Avg'])
    else:
        corr_val, p_val = np.nan, np.nan

    return merged_df, (corr_val, p_val)


def plot_scatter(ax, x, y, xlabel, ylabel, title):
    """
    Creates a scatter plot on the given axis.
    """
    ax.scatter(x, y, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)


############################################################
# 2. Main Script
############################################################

def main():
    # Setup score types, directories, and file paths
    scores = ['logSED']
    score = scores[0]
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    eqtl_tsv = os.path.join(root_dir, 'data', 'eQTL_kita_etal', 'fix', 'selected_eQTL', 'intersected_CIS.tsv')
    
    # Load and preprocess eQTL data
    df_eqtl = load_eqtl_data(eqtl_tsv)
    if df_eqtl is None:
        return

    # Initialize a master DataFrame for fold scores
    all_data = df_eqtl[['Position_Gene', 'effectSize']].copy()

    print("all_data: ", all_data)

    # Create a 3x3 plot (8 folds + 1 average)
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    axes = axes.flatten()
    fold_correlations = []
    folds = [f'f{i}c0' for i in range(8)]
    output_dir = './viz'
    os.makedirs(output_dir, exist_ok=True)

    # Process each fold
    for i, fold in enumerate(folds):
        merged_df, (corr_val, p_val) = process_fold_data(fold, score, df_eqtl)
        if merged_df is None:
            print(f"Skipping fold {fold} due to errors.")
            continue

        fold_correlations.append((corr_val, p_val))
        title = f'Fold {fold}\nr={corr_val:.3f}, p={p_val:.1e}'
        plot_scatter(axes[i],
                     merged_df['effectSize'],
                     merged_df[f'{score}_Avg'],
                     'effectSize',
                     score,
                     title)

        # Merge fold-specific score into the master DataFrame
        col_name = f'{score}_{fold}'
        temp_df = merged_df[['Position_Gene', f'{score}_Avg']].copy()
        temp_df.rename(columns={f'{score}_Avg': col_name}, inplace=True)
        all_data = pd.merge(all_data, temp_df, on='Position_Gene', how='left')
        print("all_data merged: ", all_data)

    # Compute the average score across folds
    fold_cols = [f'{score}_{fold}' for fold in folds]
    
    # Option to perform sign checking before averaging folds.
    # Set apply_sign_check to True to only average when almost all fold signs agree (tolerate only one difference).
    apply_sign_check = True

    if apply_sign_check:
        all_data[f'{score}_mean'] = all_data.apply(lambda row: compute_sign_checked_mean(row, fold_cols), axis=1)
    else:
        all_data[f'{score}_mean'] = all_data[fold_cols].mean(axis=1)
    
    plot_data = all_data.dropna(subset=['effectSize', f'{score}_mean']).copy()
    
    if len(plot_data) > 1:
        avg_corr_val, avg_p_val = stats.pearsonr(plot_data['effectSize'], plot_data[f'{score}_mean'])
    else:
        avg_corr_val, avg_p_val = np.nan, np.nan

    avg_title = f'Average across folds\nr={avg_corr_val:.3f}, p={avg_p_val:.1e}'
    plot_scatter(axes[8],
                 plot_data['effectSize'],
                 plot_data[f'{score}_mean'],
                 'effectSize',
                 f'Avg {score}',
                 avg_title)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'all_folds_and_average_{score}.png')
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"\nDone! Plots saved to '{fig_path}'")


if __name__ == '__main__':
    main()