#!/usr/bin/env python3
import os, re, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ─── Configuration ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BINS       = [0, 600, 1000, 2000, 3000, 6000, 8000]
BIN_LABELS = ['0-600b','600b-1kb','1kb-2kb','2kb-3kb','3kb-6kb','6kb-8kb']
BIN_MAPPING= dict(zip(BIN_LABELS, BINS[1:]))

FOLDS        = [f"f{i}c0" for i in range(8)]
MPRA_MODELS  = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP= {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

ROOT       = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
YEAST_DIR  = os.path.join(ROOT, 'experiments','SUM_data_process','eQTL_exp','results')
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
            chrom,_,feat,start,end,_,strand,_,attrs = line.strip().split('\t')
            if feat.lower() != 'gene': 
                continue
            m = re.search(r'gene_id "([^"]+)"', attrs)
            if not m: 
                continue
            gid = m.group(1)
            # convert e.g. "1" -> "chrI"
            chr_key = map_chromosome_to_roman(f"chromosome{chrom}")
            pos = int(start) if strand=='+' else int(end)
            tss[gid] = {'chrom': chr_key, 'tss': pos}
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
    if not info or info['chrom'] != chrom:
        return np.nan
    return abs(pos - info['tss'])


def process_mpra_data(pos_file, neg_file):
    pos = pd.read_csv(pos_file, sep='\t'); pos['label'] = 1
    neg = pd.read_csv(neg_file, sep='\t'); neg['label'] = 0
    pos.rename(columns={'logSED':'score'}, inplace=True)
    neg.rename(columns={'logSED':'score'}, inplace=True)
    df = pd.concat([pos, neg], ignore_index=True).dropna(subset=['score'])
    df['score'] = df['score'].abs()
    p, n = df[df.label==1], df[df.label==0]
    if len(p) > len(n):
        p = p.sample(len(n), random_state=SEED)
    else:
        n = n.sample(len(p), random_state=SEED)
    return pd.concat([p, n], ignore_index=True)


def compute_metrics_by_bin(df, model_name):
    metrics = []
    for bin_label in BIN_LABELS:
        sub = df[df.distance_bin == bin_label]
        if len(sub) < 10:
            continue
        y, s = sub['label'], sub['score']
        fpr, tpr, _ = roc_curve(y, s)
        roc_val     = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y, s)
        pr_val      = average_precision_score(y, s)
        metrics.append({
            'model':       model_name,
            'distance_bin':bin_label,
            'distance':    BIN_MAPPING[bin_label],
            'roc_auc':     roc_val,
            'pr_auc':      pr_val
        })
    return metrics


def main():
    tss_data = parse_gtf_for_tss(GTF_FILE)

    # ── MPRA metrics ───────────────────────────────────────────────────────
    mpra_metrics = []
    for mdl in MPRA_MODELS:
        base = os.path.join(ROOT, 'experiments','SUM_data_process',
                            'eQTL_MPRA_models_eval','results', mdl)
        pos_f = os.path.join(base, 'final_pos_predictions.tsv')
        if not os.path.exists(pos_f):
            raise FileNotFoundError(f"Missing file: {pos_f}")
        for ns in [1,2,3,4]:
            neg_f = os.path.join(base, f'final_neg_predictions_{ns}.tsv')
            if not os.path.exists(neg_f):
                continue
            df = process_mpra_data(pos_f, neg_f)
            df['tss_dist']     = df['Position_Gene'].apply(lambda k: calculate_tss_distance(k, tss_data))
            df = df.dropna(subset=['tss_dist'])
            df['model']        = f"{MPRA_NAME_MAP[mdl]} (neg_set {ns})"
            df['distance']     = df['tss_dist']
            df['distance_bin'] = pd.cut(df['distance'], bins=BINS, labels=BIN_LABELS)
            mpra_metrics += compute_metrics_by_bin(df, df['model'].iloc[0])
    mpra_df = pd.DataFrame(mpra_metrics)

    # ── Yeast (Shorkie) metrics ───────────────────────────────────────────
    yeast_metrics = []
    for neg_set in [1,2,3,4]:
        yeast_folds = []
        for fold in FOLDS:
            path = os.path.join(YEAST_DIR, f'set{neg_set}', f'yeast_eqtl_{fold}.tsv')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing TSV for fold {fold}: {path}")
            df = (pd.read_csv(path, sep='\t', usecols=['Position_Gene','score','label'])
                    .drop_duplicates(['Position_Gene'])
                    .dropna(subset=['score','label']))
            df['score'] = df['score'].abs()

            p, n = df[df.label==1], df[df.label==0]
            if len(p) > len(n):
                p = p.sample(len(n), random_state=SEED)
            else:
                n = n.sample(len(p), random_state=SEED)
            df_bal = pd.concat([p, n], ignore_index=True)
            df_bal['fold'] = fold
            df_bal['model']= 'Shorkie'
            yeast_folds.append(df_bal)

        yeast_all = pd.concat(yeast_folds, ignore_index=True)
        yeast_all['distance']     = yeast_all['Position_Gene'].apply(lambda k: calculate_tss_distance(k, tss_data))
        yeast_all = yeast_all.dropna(subset=['distance'])
        yeast_all['distance_bin']= pd.cut(yeast_all['distance'], bins=BINS, labels=BIN_LABELS)
        yeast_metrics += compute_metrics_by_bin(yeast_all, f'Shorkie (neg_set {neg_set})')

    yeast_df = pd.DataFrame(yeast_metrics)

    # ── Combine & plot ──────────────────────────────────────────────────────
    combined_roc = pd.concat([
        mpra_df[['model','distance','roc_auc']],
        yeast_df[['model','distance','roc_auc']]
    ], ignore_index=True)
    combined_pr  = pd.concat([
        mpra_df[['model','distance','pr_auc']],
        yeast_df[['model','distance','pr_auc']]
    ], ignore_index=True)

    # point palette
    cmap_defs = {
        'DREAM-Atten': cm.get_cmap('Reds',    5),
        'DREAM-CNN':   cm.get_cmap('Greens',  5),
        'DREAM-RNN':   cm.get_cmap('Purples', 5),
        'Shorkie':     cm.get_cmap('Blues',   5),
    }
    palette = {}
    for base, cmap in cmap_defs.items():
        for i in range(4):
            label = f"{base} (neg_set {i+1})"
            palette[label] = mcolors.to_hex(cmap(0.3 + 0.7 * i / 3))

    # trend palette (colorblind-friendly)
    trend_colors = dict(zip(
        cmap_defs.keys(),
        sns.color_palette("colorblind", len(cmap_defs))
    ))

    def plot_metric(df, ycol, ylabel, fname):
        plt.figure(figsize=(10,5))
        sns.scatterplot(
            data=df, x='distance', y=ycol,
            hue='model', palette=palette,
            alpha=0.7, linewidth=0, marker='o'
        )
        for mdl, color in trend_colors.items():
            sub = df[df.model.str.startswith(mdl)]
            if sub.empty: continue
            sns.regplot(
                data=sub, x='distance', y=ycol,
                scatter=False, color=color, label=f'{mdl} trend'
            )
        plt.xticks(list(BIN_MAPPING.values()), list(BIN_MAPPING.keys()), rotation=45)
        plt.xlabel('TSS Distance Bin')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.close()

    plot_metric(combined_roc, 'roc_auc', 'ROC AUC', 'distance_bin_roc.png')
    plot_metric(combined_pr,  'pr_auc',  'PR AUC',  'distance_bin_pr.png')

    print("Done: plots saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
