#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import h5py
import scipy.stats as stats

# -------------------------------
# Constants
# -------------------------------
NUM_FOLDS = 8   # number of eQTL folds (f0c0...f7c0)
SCORE_NAME = 'logSED'
NEG_SUBDIRS = ['set1', 'set2', 'set3', 'set4']

# -------------------------------
# Shared Helper Functions
# -------------------------------
def map_chromosome_to_roman(chromosome):
    roman_mapping = {
        'chromosome1':'chrI','chromosome2':'chrII','chromosome3':'chrIII',
        'chromosome4':'chrIV','chromosome5':'chrV','chromosome6':'chrVI',
        'chromosome7':'chrVII','chromosome8':'chrVIII','chromosome9':'chrIX',
        'chromosome10':'chrX','chromosome11':'chrXI','chromosome12':'chrXII',
        'chromosome13':'chrXIII','chromosome14':'chrXIV','chromosome15':'chrXV','chromosome16':'chrXVI',
        'I':'chrI','II':'chrII','III':'chrIII','IV':'chrIV','V':'chrV','VI':'chrVI','VII':'chrVII',
        'VIII':'chrVIII','IX':'chrIX','X':'chrX','XI':'chrXI','XII':'chrXII','XIII':'chrXIII','XIV':'chrXIV','XV':'chrXV','XVI':'chrXVI'
    }
    return roman_mapping.get(chromosome, chromosome)


def calculate_tss_distance(key, tss_data):
    parts = key.split('_')
    gene_id = parts[-1] if len(parts) >= 2 else None
    pos_part = '_'.join(parts[:-1])
    try:
        chrom, pos = pos_part.split(':')
        pos = int(pos)
        gene_info = tss_data.get(gene_id)
        if not gene_info:
            return None
        gene_chrom = map_chromosome_to_roman(gene_info['chrom'])
        return abs(pos - gene_info['tss']) if gene_chrom == chrom else None
    except Exception:
        return None

# -------------------------------
# Data Loaders
# -------------------------------
def load_eqtl_data(eqtl_tsv):
    df = pd.read_csv(eqtl_tsv, sep='\t')
    df['Chr'] = df['Chr'].apply(map_chromosome_to_roman)
    df['Position_Gene'] = df['Chr'] + ':' + df['ChrPos'].astype(str) + '_' + df['Pheno']
    return df[['Position_Gene', 'SnpWeight']]


def parse_gtf_for_tss(gtf_file):
    tss = {}
    with open(gtf_file) as f:
        for line in f:
            if line.startswith('#'): continue
            cols = line.strip().split('\t')
            if len(cols) < 9 or cols[2].lower() != 'gene': continue
            chrom = cols[0]
            start, end = int(cols[3]), int(cols[4])
            strand = cols[6]
            attr = cols[8]
            m = re.search(r'gene_id\s+"([^"]+)"', attr)
            if not m: continue
            gid = m.group(1)
            tss[gid] = {'chrom': chrom, 'tss': start if strand == '+' else end}
    return tss


def load_scores(h5_path, score):
    scores = {}
    with h5py.File(h5_path, 'r') as hdf:
        grp = hdf[score]
        for pos in grp:
            for gene in grp[pos]:
                key = f"{pos}_{gene}"
                scores[key] = np.mean(grp[pos][gene][:])
    return scores

# -------------------------------
# Fold Processing
# -------------------------------
def process_fold(eqtl_df, tss_data, base_dir, fold, score_name, is_positive, neg_subdir=None):
    if is_positive:
        h5_file = os.path.join(base_dir, f"eqtl_{fold}", 'scores.h5')
    else:
        if neg_subdir is None:
            raise ValueError("neg_subdir must be specified for negative examples")
        h5_file = os.path.join(base_dir, fold, neg_subdir, 'scores.h5')

    if not os.path.exists(h5_file):
        print(f"Missing scores for fold {fold} ({'positive' if is_positive else neg_subdir}): {h5_file}")
        return pd.DataFrame()

    data_dict = load_scores(h5_file, score_name)
    df = pd.DataFrame(list(data_dict.items()), columns=['Position_Gene', 'score'])

    if is_positive:
        df = df.merge(eqtl_df, on='Position_Gene', how='inner')
    df['distance'] = df['Position_Gene'].map(lambda k: calculate_tss_distance(k, tss_data))
    df = df.dropna(subset=['distance'])
    df['fold'] = fold
    df['label'] = 1 if is_positive else 0
    df['label_type'] = 'Positive' if is_positive else f"Negative_{neg_subdir}"
    if not is_positive:
        df['SnpWeight'] = np.nan
    return df

# -------------------------------
# Main
# -------------------------------
def main():
    root = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    eqtl_tsv = os.path.join(root, 'data', 'eQTL', 'selected_eQTL', 'intersected_data_CIS.tsv')
    gtf = os.path.join(root, 'data', 'eQTL', 'neg_eQTLS', 'GCA_000146045_2.59.gtf')
    pos_base = os.path.join(root, 'experiments', 'SUM_data_process', 'eQTL_exp', 'eqtls')
    neg_base = os.path.join(root, 'experiments', 'SUM_data_process', 'eQTL_pseudo_negatives', 'eqtl_neg')

    eqtl_df = load_eqtl_data(eqtl_tsv)
    tss_data = parse_gtf_for_tss(gtf)

    out_dir = 'results'

    for i in range(NUM_FOLDS):
        fold = f'f{i}c0'
        print(f"Processing positives for fold {fold}...")
        pos_df = process_fold(eqtl_df, tss_data, pos_base, fold, SCORE_NAME, is_positive=True)

        for neg_subdir in NEG_SUBDIRS:
            os.makedirs(f'{out_dir}/{neg_subdir}', exist_ok=True)
            print(f"Processing negatives ({neg_subdir}) for fold {fold}...")
            neg_df = process_fold(eqtl_df, tss_data, neg_base, fold, SCORE_NAME, is_positive=False, neg_subdir=neg_subdir)

            df_all = pd.concat([pos_df, neg_df], ignore_index=True)
            tmp = df_all['Position_Gene'].str.split('_').str[0].str.split(':', expand=True)
            df_all['Chr'] = tmp[0]
            df_all['ChrPos'] = tmp[1].astype(int)

            out_file = os.path.join(out_dir, neg_subdir, f'yeast_eqtl_{fold}.tsv')
            df_all.to_csv(out_file, sep='\t', index=False)

            print("df_all: ", df_all.head())
            print("Positive: ", len(pos_df), "Negative: ", len(neg_df))
            print(f"Written fold {fold}, negative set {neg_subdir}: {len(df_all)} rows -> {out_file}")

if __name__ == '__main__':
    main()
