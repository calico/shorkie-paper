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
    df['Position_Gene'] = df['Chr'] + ':' + df['ChrPos'].astype(str) + '_' + df['#Gene']
    df['effectSize'] = df['effectSize'].astype(float)
    return df[['Position_Gene', 'effectSize', 'Chr', 'ChrPos', '#Gene', 'locationType']]


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
    print(f"Loaded {len(df)} scores for fold {fold} ({'positive' if is_positive else neg_subdir})")

    if is_positive:
        df = df.merge(eqtl_df, on='Position_Gene', how='inner')
    else:
        df["#Gene"] = df['Position_Gene'].str.split('_').str[-1]
    df['distance'] = df['Position_Gene'].map(lambda k: calculate_tss_distance(k, tss_data))
    df = df.dropna(subset=['distance'])
    df['fold'] = fold
    df['label'] = 1 if is_positive else 0
    df['label_type'] = 'Positive' if is_positive else f"Negative_{neg_subdir}"
    if not is_positive:
        df['effectSize'] = np.nan
    return df

# -------------------------------
# Main
# -------------------------------
# def main():
#     root = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
#     eqtl_tsv = os.path.join(root, 'data', 'eQTL_kita_etal', 'fix', 'selected_eQTL', 'intersected_CIS.tsv')
#     gtf = os.path.join(root, 'data', 'eQTL', 'neg_eQTLS', 'GCA_000146045_2.59.gtf')
#     pos_base = os.path.join(root, 'experiments', 'SUM_data_process', 'eQTL_kita_etal_select_exp', 'eqtls')
#     neg_base = os.path.join(root, 'experiments', 'SUM_data_process', 'eQTL_kita_etal_select_exp_pseudo_negatives', 'eqtl_neg')

#     eqtl_df = load_eqtl_data(eqtl_tsv)
#     print("Loaded eQTL data: ", eqtl_df.head())
#     print("eqtl_df: ", eqtl_df['locationType'].value_counts())
    
#     tss_data = parse_gtf_for_tss(gtf)

#     out_dir = 'results'

#     # for i in range(NUM_FOLDS):
#     #     fold = f'f{i}c0'
#     #     print(f"Processing positives for fold {fold}...")
#     #     pos_df = process_fold(eqtl_df, tss_data, pos_base, fold, SCORE_NAME, is_positive=True)
#     #     print("pos_df: ", pos_df.head())

#     #     for neg_subdir in NEG_SUBDIRS:
#     #         os.makedirs(f'{out_dir}/{neg_subdir}', exist_ok=True)
#     #         print(f"Processing negatives ({neg_subdir}) for fold {fold}...")
#     #         neg_df = process_fold(eqtl_df, tss_data, neg_base, fold, SCORE_NAME, is_positive=False, neg_subdir=neg_subdir)

#     #         df_all = pd.concat([pos_df, neg_df], ignore_index=True)
#     #         tmp = df_all['Position_Gene'].str.split('_').str[0].str.split(':', expand=True)
#     #         df_all['Chr'] = tmp[0]
#     #         df_all['ChrPos'] = tmp[1].astype(int)

#     #         out_file = os.path.join(out_dir, neg_subdir, f'yeast_eqtl_{fold}.tsv')
#     #         df_all.to_csv(out_file, sep='\t', index=False)
#     #         print("df_all: ", df_all.head())
#     #         print("Positive: ", len(pos_df), "Negative: ", len(neg_df))
#     #         print(f"Written fold {fold}, negative set {neg_subdir}: {len(df_all)} rows -> {out_file}")

#     # process each fold
#     for i in range(NUM_FOLDS):
#         fold = f'f{i}c0'
#         print(f"\n=== Processing fold {fold} positives ===")
#         pos_df = process_fold(eqtl_df, tss_data, pos_base, fold, SCORE_NAME, is_positive=True)
#         print(pos_df.head())

#         for neg_subdir in NEG_SUBDIRS:
#             os.makedirs(f'{out_dir}/{neg_subdir}', exist_ok=True)
#             print(f"--- Processing fold {fold} negatives ({neg_subdir}) ---")
#             neg_df = process_fold(eqtl_df, tss_data, neg_base, fold, SCORE_NAME, is_positive=False, neg_subdir=neg_subdir)

#             # combine pos & neg for this fold
#             df_all = pd.concat([pos_df, neg_df], ignore_index=True)

#             # ensure Chr & ChrPos columns
#             tmp = df_all['Position_Gene'].str.split('_').str[0].str.split(':', expand=True)
#             df_all['Chr']    = tmp[0]
#             df_all['ChrPos'] = tmp[1].astype(int)

#             # write per‐fold TSV
#             out_file = os.path.join(out_dir, neg_subdir, f'yeast_eqtl_{fold}.tsv')
#             df_all.to_csv(out_file, sep='\t', index=False)
#             print(f"Written fold {fold}, negative set {neg_subdir}: {len(df_all)} rows -> {out_file}")

#             # accumulate for averaging later
#             all_results[neg_subdir].append(df_all)

#     # after all folds: compute and write average scores
#     for neg_subdir, dfs in all_results.items():
#         print(f"\n=== Averaging scores across folds for negative set {neg_subdir} ===")
#         df_concat = pd.concat(dfs, ignore_index=True)

#         # group by all identifying columns except 'score' & 'fold'
#         group_cols = ['Position_Gene', 'Chr', 'ChrPos', '#Gene', 'label', 'label_type', 'effectSize', 'distance']
#         df_avg = (
#             df_concat
#             .groupby(group_cols, as_index=False)['score']
#             .mean()
#             .rename(columns={'score': 'average_score'})
#         )

#         summary_file = os.path.join(out_dir, neg_subdir, 'yeast_eqtl_average.tsv')
#         df_avg.to_csv(summary_file, sep='\t', index=False)
#         print(f"Wrote average-score summary for {neg_subdir}: {summary_file}")


def main():
    root      = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    eqtl_tsv  = os.path.join(root, 'data', 'eQTL_kita_etal', 'fix', 'selected_eQTL', 'intersected_CIS.tsv')
    gtf       = os.path.join(root, 'data', 'eQTL', 'neg_eQTLS', 'GCA_000146045_2.59.gtf')
    pos_base  = os.path.join(root, 'experiments', 'SUM_data_process', 'eQTL_kita_etal_select_exp',      'eqtls')
    neg_base  = os.path.join(root, 'experiments', 'SUM_data_process', 'eQTL_kita_etal_select_exp_pseudo_negatives', 'eqtl_neg')
    out_dir   = 'results'

    # load shared data
    eqtl_df = load_eqtl_data(eqtl_tsv)
    print("Loaded eQTL data:\n", eqtl_df.head(), "\n", eqtl_df['locationType'].value_counts())
    tss_data = parse_gtf_for_tss(gtf)

    # prepare accumulator for cross‐fold averaging
    all_results = {neg: [] for neg in NEG_SUBDIRS}

    # process each fold
    for i in range(NUM_FOLDS):
        fold = f'f{i}c0'
        print(f"\n=== Processing fold {fold} positives ===")
        pos_df = process_fold(eqtl_df, tss_data, pos_base, fold, SCORE_NAME, is_positive=True)
        print(pos_df.head())

        for neg_subdir in NEG_SUBDIRS:
            os.makedirs(f'{out_dir}/{neg_subdir}', exist_ok=True)
            print(f"--- Processing fold {fold} negatives ({neg_subdir}) ---")
            neg_df = process_fold(eqtl_df, tss_data, neg_base, fold, SCORE_NAME, is_positive=False, neg_subdir=neg_subdir)

            # combine pos & neg for this fold
            df_all = pd.concat([pos_df, neg_df], ignore_index=True)

            # ensure Chr & ChrPos columns
            tmp = df_all['Position_Gene'].str.split('_').str[0].str.split(':', expand=True)
            df_all['Chr']    = tmp[0]
            df_all['ChrPos'] = tmp[1].astype(int)

            # write per‐fold TSV
            out_file = os.path.join(out_dir, neg_subdir, f'yeast_eqtl_{fold}.tsv')
            df_all.to_csv(out_file, sep='\t', index=False)
            print(f"Written fold {fold}, negative set {neg_subdir}: {len(df_all)} rows -> {out_file}")

            # accumulate for averaging later
            all_results[neg_subdir].append(df_all)

    # after all folds: compute and write per‐fold scores and average
    for neg_subdir, dfs in all_results.items():
        print(f"\n=== Aggregating scores across folds for negative set {neg_subdir} ===")
        # 1) concatenate all the per‐fold DataFrames
        df_concat = pd.concat(dfs, ignore_index=True)

        # 2) define the columns that uniquely identify each test point
        group_cols = ['Position_Gene', 'Chr', 'ChrPos', '#Gene', 'label', 'label_type', 'effectSize', 'distance']

        # 3) pivot so that each fold becomes its own column
        wide = (
            df_concat
            .pivot_table(
                index=group_cols,
                columns='fold',
                values='score',
                aggfunc='first'    # each (Position_Gene, fold) is unique
            )
            .reset_index()
        )
        # flatten the MultiIndex columns
        wide.columns.name = None

        # 4) compute the average across all fold‐columns
        #    we know folds are named f0c0…f7c0
        fold_cols = [f'f{i}c0' for i in range(NUM_FOLDS)]
        # it's possible some folds are missing for a given neg_subdir; only average existing columns
        existing_fold_cols = [c for c in fold_cols if c in wide.columns]
        wide['average_score'] = wide[existing_fold_cols].mean(axis=1)

        # 5) reorder columns: ids, per‐fold scores, then average_score
        out_cols = group_cols + existing_fold_cols + ['average_score']
        final_df = wide[out_cols]

        # 6) write to TSV
        summary_file = os.path.join(out_dir, neg_subdir, 'yeast_eqtl_scores_by_fold.tsv')
        final_df.to_csv(summary_file, sep='\t', index=False)
        print(f"Wrote per‐fold + average scores for {neg_subdir}: {summary_file}")

if __name__ == '__main__':
    main()
