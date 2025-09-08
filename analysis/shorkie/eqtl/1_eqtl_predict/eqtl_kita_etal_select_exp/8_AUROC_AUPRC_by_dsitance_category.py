#!/usr/bin/env python3
import os
import re
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# ─── Configuration ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BINS        = [0, 1000, 2000, 3000, 6000, 8000]
BIN_LABELS  = ['0-1kb','1kb-2kb','2kb-3kb','3kb-6kb','6kb-8kb']
BIN_MAPPING = dict(zip(BIN_LABELS, BINS[1:]))

FOLDS = [f"f{i}c0" for i in range(8)]

MPRA_MODELS   = ["DREAM_Atten", "DREAM_RNN", "DREAM_CNN"]
MPRA_NAME_MAP = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

ROOT       = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
YEAST_DIR  = os.path.join(ROOT, 'experiments','SUM_data_process','eQTL_kita_etal_select_exp','results')
GTF_FILE   = os.path.join(ROOT, 'data','eQTL','neg_eQTLS','GCA_000146045_2.59.gtf')
OUTPUT_DIR = 'results/viz'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def map_chromosome_to_roman(chrom):
    roman = {
        'chromosome1':'chrI','chromosome2':'chrII','chromosome3':'chrIII',
        'chromosome4':'chrIV','chromosome5':'chrV','chromosome6':'chrVI',
        'chromosome7':'chrVII','chromosome8':'chrVIII','chromosome9':'chrIX',
        'chromosome10':'chrX','chromosome11':'chrXI','chromosome12':'chrXII',
        'chromosome13':'chrXIII','chromosome14':'chrXIV','chromosome15':'chrXV',
        'chromosome16':'chrXVI'
    }
    return roman.get(chrom, chrom)


def parse_gtf_for_tss(gtf_path):
    tss = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, _, feat, start, end, _, strand, _, attrs = line.strip().split('\t')
            if feat.lower() != 'gene':
                continue
            m = re.search(r'gene_id "([^"]+)"', attrs)
            if not m:
                continue
            gid = m.group(1)
            s, e = int(start), int(end)
            tss[gid] = {
                'chrom': f"chr{chrom}",
                'tss': s if strand == '+' else e
            }
    return tss


def calculate_tss_distance(key, tss_data):
    parts = key.split('_')
    if len(parts) < 2:
        return np.nan
    pos_str, gid = '_'.join(parts[:-1]), parts[-1]
    try:
        chrom, pos = pos_str.split(':')
        pos = int(pos)
    except ValueError:
        return np.nan
    info = tss_data.get(gid)
    if not info or map_chromosome_to_roman(info['chrom']) != chrom:
        return np.nan
    return abs(pos - info['tss'])


def process_mpra_data(pos_file, neg_file):
    pos = pd.read_csv(pos_file, sep='\t'); pos['label'] = 1
    neg = pd.read_csv(neg_file, sep='\t'); neg['label'] = 0
    pos.rename(columns={'logSED': 'score'}, inplace=True)
    neg.rename(columns={'logSED': 'score'}, inplace=True)
    df = pd.concat([pos, neg], ignore_index=True).dropna(subset=['score'])
    df['score'] = df['score'].abs()
    p, n = df[df.label == 1], df[df.label == 0]
    if len(p) > len(n):
        p = p.sample(len(n), random_state=SEED)
    else:
        n = n.sample(len(p), random_state=SEED)
    df = pd.concat([p, n], ignore_index=True)
    df['locationType'] = 'All'   # no categories for MPRA
    return df


def compute_metrics_by_bin_and_category(df, model_name):
    """
    For each category in df['locationType'] plus 'All', compute ROC AUC and AUPRC
    in each distance bin.
    """
    metrics = []
    cats = ['All'] + sorted(df['locationType'].dropna().unique())
    for cat in cats:
        sub_cat = df if cat == 'All' else df[df['locationType'] == cat]
        for bin_label in BIN_LABELS:
            sub = sub_cat[sub_cat.distance_bin == bin_label]
            if len(sub) < 10:
                continue
            y, s = sub['label'], sub['score']
            fpr, tpr, _ = roc_curve(y, s)
            roc_val     = auc(fpr, tpr)
            prec, rec, _ = precision_recall_curve(y, s)
            pr_val      = average_precision_score(y, s)
            metrics.append({
                'model':        model_name,
                'category':     cat,
                'distance_bin': bin_label,
                'distance':     BIN_MAPPING[bin_label],
                'roc_auc':      roc_val,
                'pr_auc':       pr_val,
                'n_pos':        int((sub.label==1).sum()),
                'n_neg':        int((sub.label==0).sum())
            })
    return metrics


def main():
    tss_data = parse_gtf_for_tss(GTF_FILE)

    # ─── MPRA metrics (no locationType split) ───────────────────────────────
    mpra_metrics = []
    for mdl in MPRA_MODELS:
        base = os.path.join(ROOT, 'experiments','SUM_data_process',
                            'eQTL_MPRA_models_eval_kita_etal_select','results', mdl)
        pos_f = os.path.join(base, 'final_pos_predictions.tsv')
        for ns in [1,2,3,4]:
            neg_f = os.path.join(base, f'final_neg_predictions_{ns}.tsv')
            df    = process_mpra_data(pos_f, neg_f)
            label = f"{MPRA_NAME_MAP[mdl]} (neg_set {ns})"
            df['model'] = label
            df['distance']     = df['tss_dist']
            df['distance_bin'] = pd.cut(df['distance'], bins=BINS, labels=BIN_LABELS)
            mpra_metrics += compute_metrics_by_bin_and_category(df, label)

    mpra_df = pd.DataFrame(mpra_metrics)

    # ─── Yeast (Shorkie) metrics, split by locationType ─────────────────────
    yeast_metrics = []
    for neg_set in [1,2,3,4]:
        yeast_folds = []
        for fold in FOLDS:
            path = os.path.join(YEAST_DIR, f'set{neg_set}', f'yeast_eqtl_{fold}.tsv')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing TSV for fold {fold}: {path}")
            df = pd.read_csv(path, sep='\t',
                             usecols=['Position_Gene','score','label','locationType']) \
                   .drop_duplicates(['Position_Gene']) \
                   .dropna(subset=['score','label'])
            df['score'] = df['score'].abs()
            p, n = df[df.label==1], df[df.label==0]
            if len(p) > len(n):
                p = p.sample(len(n), random_state=SEED)
            else:
                n = n.sample(len(p), random_state=SEED)
            df_bal = pd.concat([p,n], ignore_index=True)
            df_bal['distance'] = df_bal['Position_Gene'] \
                                   .apply(lambda k: calculate_tss_distance(k, tss_data))
            df_bal = df_bal.dropna(subset=['distance'])
            df_bal['distance_bin'] = pd.cut(df_bal['distance'], bins=BINS, labels=BIN_LABELS)
            df_bal['model'] = f"Shorkie (neg_set {neg_set})"
            yeast_metrics += compute_metrics_by_bin_and_category(df_bal, df_bal['model'].iloc[0])

    yeast_df = pd.DataFrame(yeast_metrics)

    # ─── Save metrics to CSV ────────────────────────────────────────────────
    mpra_df.to_csv(os.path.join(OUTPUT_DIR, 'mpra_distance_metrics.csv'), index=False)
    yeast_df.to_csv(os.path.join(OUTPUT_DIR, 'yeast_distance_metrics_by_category.csv'), index=False)

    # ─── Combine for plotting ───────────────────────────────────────────────
    combined_roc = pd.concat([
        mpra_df[['model','category','distance','roc_auc']],
        yeast_df[['model','category','distance','roc_auc']]
    ], ignore_index=True)
    combined_pr  = pd.concat([
        mpra_df[['model','category','distance','pr_auc']],
        yeast_df[['model','category','distance','pr_auc']]
    ], ignore_index=True)

    # extract base model name for consistent coloring
    combined_roc['base_model'] = combined_roc['model'].str.replace(r' \(neg_set [0-9]\)', '', regex=True)
    combined_pr ['base_model'] = combined_pr ['model'].str.replace(r' \(neg_set [0-9]\)', '', regex=True)

    model_colors = {
        'DREAM-Atten': 'tab:blue',
        'DREAM-CNN':   'tab:orange',
        'DREAM-RNN':   'tab:green',
        'Shorkie':     'tab:red',
    }

    # ─── ROC plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11,5))
    sns.scatterplot(data=combined_roc, x='distance', y='roc_auc',
                    hue='base_model', style='category',
                    palette=model_colors, alpha=0.7, edgecolor='w',
                    ax=ax, legend=False)

    for base_model, color in model_colors.items():
        subset = combined_roc[combined_roc.base_model == base_model]
        sns.lineplot(data=subset, x='distance', y='roc_auc',
                     hue='category', palette='dark', estimator=None,
                     lw=2.5, ax=ax, legend=False)

    ax.set_xticks(list(BIN_MAPPING.values()))
    ax.set_xticklabels(list(BIN_MAPPING.keys()), rotation=45)
    ax.set_xlabel('TSS Distance Bin', fontsize=14)
    ax.set_ylabel('ROC AUC', fontsize=14)
    ax.set_title('ROC AUC by Distance Bin & locationType', fontsize=16, pad=50)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.17), fontsize=12)
    fig.subplots_adjust(top=0.80)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,'distance_bin_roc_by_category.png'), dpi=300)
    plt.close(fig)

    # ─── PR plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11,5))
    sns.scatterplot(data=combined_pr, x='distance', y='pr_auc',
                    hue='base_model', style='category',
                    palette=model_colors, alpha=0.7, edgecolor='w',
                    ax=ax, legend=False)

    for base_model, color in model_colors.items():
        subset = combined_pr[combined_pr.base_model == base_model]
        sns.lineplot(data=subset, x='distance', y='pr_auc',
                     hue='category', palette='dark', estimator=None,
                     lw=2.5, ax=ax, legend=False)

    ax.set_xticks(list(BIN_MAPPING.values()))
    ax.set_xticklabels(list(BIN_MAPPING.keys()), rotation=45)
    ax.set_xlabel('TSS Distance Bin', fontsize=14)
    ax.set_ylabel('AUPRC', fontsize=14)
    ax.set_title('AUPRC by Distance Bin & locationType', fontsize=16, pad=50)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.17), fontsize=12)
    fig.subplots_adjust(top=0.80)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,'distance_bin_pr_by_category.png'), dpi=300)
    plt.close(fig)

    print("Done: metrics saved and plots rendered in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
