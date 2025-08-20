#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import scipy.stats as stats
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# -------------------------------
# Shared Helper Functions
# (unchanged)
# -------------------------------
def map_chromosome_to_roman(chromosome):
    """Map chromosome names to Roman numeral style with a 'chr' prefix."""
    roman_mapping = {
        'chromosome1': 'chrI',  'chromosome2': 'chrII',  'chromosome3': 'chrIII',
        'chromosome4': 'chrIV',   'chromosome5': 'chrV',   'chromosome6': 'chrVI',
        'chromosome7': 'chrVII',  'chromosome8': 'chrVIII', 'chromosome9': 'chrIX',
        'chromosome10': 'chrX',   'chromosome11': 'chrXI', 'chromosome12': 'chrXII',
        'chromosome13': 'chrXIII','chromosome14': 'chrXIV','chromosome15': 'chrXV',
        'chromosome16': 'chrXVI', 'I': 'chrI', 'II': 'chrII', 'III': 'chrIII',
        'IV': 'chrIV', 'V': 'chrV', 'VI': 'chrVI', 'VII': 'chrVII',
        'VIII': 'chrVIII', 'IX': 'chrIX', 'X': 'chrX', 'XI': 'chrXI',
        'XII': 'chrXII', 'XIII': 'chrXIII', 'XIV': 'chrXIV', 'XV': 'chrXV', 'XVI': 'chrXVI'
    }
    return roman_mapping.get(chromosome, chromosome)

def calculate_tss_distance(key, tss_data):
    """
    Given a key in the format "chr:pos_geneID", compute the absolute distance 
    from the genomic position to the gene TSS (if chromosomes match).
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
    except Exception:
        return None

def process_mpra_data(pos_file, neg_file):
    """
    Read MPRA positive and negative prediction TSV files,
    bin TSS distances, and return a combined DataFrame.
    """
    pos_df = pd.read_csv(pos_file, sep="\t")
    neg_df = pd.read_csv(neg_file, sep="\t")
    pos_df['label'] = 1
    neg_df['label'] = 0
    bins = [0, 600, 1000, 2000, 3000, 5000, 8000]
    bin_labels = ['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb']
    pos_df['distance_bin'] = pd.cut(pos_df['tss_dist'], bins=bins, labels=bin_labels)
    neg_df['distance_bin'] = pd.cut(neg_df['tss_dist'], bins=bins, labels=bin_labels)
    pos_df['label_type'] = 'Positive'
    neg_df['label_type'] = 'Negative'
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined.rename(columns={'logSED': 'score'}, inplace=True)
    return combined

# -------------------------------
# Yeast Model Processing
# -------------------------------
def load_eqtl_data(eqtl_tsv):
    """
    Load positive eQTL data from TSV.
    Expects columns: Chr, ChrPos, Pheno, SnpWeight, etc.
    """
    try:
        df_eqtl = pd.read_csv(eqtl_tsv, sep='\t')
    except Exception as e:
        print(f"Error reading TSV file {eqtl_tsv}: {e}")
        return None

    df_eqtl['Chr'] = df_eqtl['Chr'].apply(map_chromosome_to_roman)
    df_eqtl['position'] = df_eqtl['Chr'] + ':' + df_eqtl['ChrPos'].astype(str)
    df_eqtl['Position_Gene'] = df_eqtl['position'] + '_' + df_eqtl['Pheno']
    df_eqtl = df_eqtl[['Position_Gene', 'SnpWeight']]
    return df_eqtl

def parse_gtf_for_tss(gtf_file):
    """Parse a GTF file to extract TSS information for each gene."""
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
                tss[gene_id] = {'chrom': chrom, 'tss': start if strand=='+' else end,
                                'start': start, 'end': end}
    return tss

def load_scores(file_path, score):
    """
    Load scores (e.g. 'logSED') from an HDF5 file.
    Returns a dictionary: { score: { "position_gene": mean_value, ... } }
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

def process_fold_data(fold, score, df_eqtl):
    """
    For a given fold, load fold-specific score data from an HDF5 file and merge with the eQTL positives.
    Returns a DataFrame and (correlation, p-value).
    """
    snp_dir = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/eQTL_exp/eqtls/eqtl_{fold}/'
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

def process_yeast_combined_data(eqtl_tsv, gtf_file, neg_base_dir, score='logSED'):
    """
    Process yeast data: load positives from eQTL TSV and fold scores, load negatives from HDF5 files,
    compute TSS distances and bin them, and return a combined DataFrame.
    """
    df_eqtl = load_eqtl_data(eqtl_tsv)
    if df_eqtl is None:
        return None
    tss_data = parse_gtf_for_tss(gtf_file)
    
    folds = [f'f{i}c0' for i in range(8)]
    neg_sets = [1]  # using one negative set for simplicity
    pos_master = df_eqtl.copy()
    for fold in folds:
        merged_df, _ = process_fold_data(fold, score, df_eqtl)
        if merged_df is None:
            print(f"Skipping fold {fold} due to errors.")
            continue
        col_name = f'{score}_{fold}'
        temp_df = merged_df[['Position_Gene', f'{score}_Avg']].copy()
        temp_df.rename(columns={f'{score}_Avg': col_name}, inplace=True)
        pos_master = pd.merge(pos_master, temp_df, on='Position_Gene', how='left')
    
    fold_cols = [f'{score}_{fold}' for fold in folds]
    pos_long = pd.melt(pos_master, id_vars=['Position_Gene','SnpWeight'], 
                       value_vars=fold_cols, var_name='fold', value_name='score')
    pos_long['fold'] = pos_long['fold'].str.replace(f'{score}_', '')
    pos_long['distance'] = pos_long['Position_Gene'].apply(lambda k: calculate_tss_distance(k, tss_data))
    pos_long = pos_long.dropna(subset=['distance'])
    pos_long['distance_bin'] = pd.cut(pos_long['distance'],
                                      bins=[0,600,1000,2000,3000,5000,8000],
                                      labels=['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb'])
    pos_long = pos_long[pos_long['distance_bin'] != '>8kb']
    pos_long['label'] = 1
    pos_long['label_type'] = 'Positive'
    
    neg_data = []
    for fold in folds:
        for ns in neg_sets:
            neg_h5 = os.path.join(neg_base_dir, f'{fold}', f'set{ns}', 'scores.h5')
            try:
                neg_scores = load_scores(neg_h5, score)[score]
            except Exception as e:
                print(f"Error loading negatives for fold {fold}, set {ns}: {e}")
                continue
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
    bin_labels = ['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb','>8kb']
    neg_df['distance_bin'] = pd.cut(neg_df['distance'], bins=bins, labels=bin_labels)
    neg_df = neg_df[neg_df['distance_bin'] != '>8kb']
    neg_df['label_type'] = 'Negative'
    
    pos_df = pos_long.copy()
    pos_df['label'] = 1
    pos_df['label_type'] = 'Positive'
    yeast_df = pd.concat([pos_df[['Position_Gene','score','distance','distance_bin','label','label_type']],
                          neg_df[['Position_Gene','score','distance','distance_bin','label','label_type']]],
                         ignore_index=True)
    return yeast_df

# -------------------------------
# Main Visualization
# -------------------------------
def main():
    # --- File paths & parameters ---
    model_archs = ["DREAM_CNN", "DREAM_RNN", "DREAM_Atten"]
    model_archs_name_map = {
        "DREAM_CNN": "DREAM-CNN",
        "DREAM_RNN": "DREAM-RNN",
        "DREAM_Atten": "DREAM-Atten"
    }

    root_dir   = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    eqtl_tsv   = os.path.join(root_dir, 'data', 'eQTL', 'selected_eQTL', 'intersected_data_CIS.tsv')
    gtf_file   = os.path.join(root_dir, 'data', 'eQTL', 'neg_eQTLS', 'GCA_000146045_2.59.gtf')
    neg_base   = os.path.join(root_dir, 'experiments', 'SUM_data_process', 'eQTL_pseudo_negatives', 'eqtl_neg')
    out_dir    = 'results/viz'
    os.makedirs(out_dir, exist_ok=True)

    # Pre‑compute yeast once
    yeast_df = process_yeast_combined_data(eqtl_tsv, gtf_file, neg_base, score='logSED')
    if yeast_df is None:
        print("Error processing yeast data."); return

    # Enforce ordering
    cat_order = ['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb']
    yeast_df.dropna(subset=['distance_bin','score'], inplace=True)
    yeast_df['distance_bin'] = pd.Categorical(yeast_df['distance_bin'], categories=cat_order, ordered=True)

    violin_kwargs = dict(
        x='distance_bin', y='score', hue='label_type',
        palette={'Positive': '#1f77b4', 'Negative': '#ff7f0e'},
        split=False, dodge=True, scale='width',
        bw=0.3, cut=0, inner='quartile'
    )

    # 4 rows, 1 column
    fig, axes = plt.subplots(nrows=len(model_archs)+1, ncols=1,
                             figsize=(10, 3.4*(len(model_archs)+1)), sharey=True)

    # Row 0–2: each MPRA model
    for i, modelarch in enumerate(model_archs):
        pos_f = os.path.join(root_dir, 'experiments', 'SUM_data_process', 'eQTL_MPRA_models_eval',
                             'results', modelarch, 'final_pos_predictions.tsv')
        neg_f = os.path.join(root_dir, 'experiments', 'SUM_data_process', 'eQTL_MPRA_models_eval',
                             'results', modelarch, 'final_neg_predictions_1.tsv')
        mpra_df = process_mpra_data(pos_f, neg_f)
        if mpra_df is None:
            print(f"Error processing {modelarch}"); continue

        mpra_df.dropna(subset=['distance_bin','score'], inplace=True)
        mpra_df['distance_bin'] = pd.Categorical(mpra_df['distance_bin'], categories=cat_order, ordered=True)

        ax = axes[i]
        sns.violinplot(data=mpra_df, ax=ax, **violin_kwargs)
        ax.set_title(f"{model_archs_name_map[modelarch]}: Score by TSS Distance", fontsize=16)
        ax.set_xlabel('TSS Distance Bin')
        ax.set_ylabel('Score')
        ax.legend(title='Label', loc='lower right')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Row 3: yeast
    ax = axes[-1]
    sns.violinplot(data=yeast_df, ax=ax, **violin_kwargs)
    ax.set_title("Shorkie: Score by TSS Distance", fontsize=16)
    ax.set_xlabel('TSS Distance Bin')
    ax.set_ylabel('Score')
    ax.legend(title='Label', loc='lower right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'violin_plots_4rows.png'))
    plt.close()

if __name__ == '__main__':
    main()
