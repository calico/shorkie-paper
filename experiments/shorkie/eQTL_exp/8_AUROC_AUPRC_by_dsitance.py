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

# Directories & files
ROOT           = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
YEAST_DIR      = os.path.join(ROOT, 'experiments','SUM_data_process','eQTL_exp','results')
GTF_FILE       = os.path.join(ROOT, 'data','eQTL','neg_eQTLS','GCA_000146045_2.59.gtf')
OUTPUT_DIR     = 'results/viz'
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
            m = re.search(r'gene_id "([^\"]+)"', attrs)
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
    return pd.concat([p, n], ignore_index=True)


def compute_metrics_by_bin(df, model_name):
    metrics = []
    for bin_label in BIN_LABELS:
        sub = df[df.distance_bin == bin_label]
        if len(sub) < 10:
            continue
        y, s = sub['label'], sub['score']
        fpr, tpr, _ = roc_curve(y, s)
        roc_val = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y, s)
        pr_val = average_precision_score(y, s)
        metrics.append({
            'model': model_name,
            'distance_bin': bin_label,
            'distance': BIN_MAPPING[bin_label],
            'roc_auc': roc_val,
            'pr_auc': pr_val
        })
    return metrics


def main():
    tss_data = parse_gtf_for_tss(GTF_FILE)

    # MPRA metrics
    mpra_metrics = []
    for mdl in MPRA_MODELS:
        base = os.path.join(ROOT, 'experiments','SUM_data_process',
                            'eQTL_MPRA_models_eval','results', mdl)
        pos_f = os.path.join(base, 'final_pos_predictions.tsv')
        for ns in [1, 2, 3, 4]:
            neg_f = os.path.join(base, f'final_neg_predictions_{ns}.tsv')
            df    = process_mpra_data(pos_f, neg_f)
            label = MPRA_NAME_MAP[mdl]
            df['model'] = f"{label} (neg_set {ns})"
            df['distance']     = df['tss_dist']
            df['distance_bin'] = pd.cut(df['distance'], bins=BINS, labels=BIN_LABELS)
            mpra_metrics += compute_metrics_by_bin(df, f"{label} (neg_set {ns})")
    mpra_df = pd.DataFrame(mpra_metrics)

    # Yeast (Shorkie) metrics
    yeast_metrics = []
    for neg_set in [1, 2, 3, 4]:
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
            df_bal['model'] = 'Shorkie'
            yeast_folds.append(df_bal)

        yeast_all = pd.concat(yeast_folds, ignore_index=True)
        yeast_all['distance'] = yeast_all['Position_Gene'] \
            .apply(lambda k: calculate_tss_distance(k, tss_data))
        yeast_all = yeast_all.dropna(subset=['distance'])
        yeast_all['distance_bin'] = pd.cut(yeast_all['distance'], bins=BINS, labels=BIN_LABELS)
        yeast_metrics += compute_metrics_by_bin(yeast_all, f'Shorkie (neg_set {neg_set})')

    yeast_df = pd.DataFrame(yeast_metrics)

    # Combine metrics
    combined_roc = pd.concat([
        mpra_df[['model','distance','roc_auc']],
        yeast_df[['model','distance','roc_auc']]
    ], ignore_index=True)
    combined_pr  = pd.concat([
        mpra_df[['model','distance','pr_auc']],
        yeast_df[['model','distance','pr_auc']]
    ], ignore_index=True)

    # Group replicates
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
                    hue='base_model', palette=model_colors,
                    alpha=0.7, edgecolor='w', legend=False, ax=ax)

    for base_model, color in model_colors.items():
        subset = combined_roc[combined_roc.base_model == base_model]
        sns.regplot(data=subset, x='distance', y='roc_auc',
                    scatter=False, color=color, ax=ax,
                    label=base_model, line_kws={'linewidth': 2.5})

    ax.set_xticks(list(BIN_MAPPING.values()))
    ax.set_xticklabels(list(BIN_MAPPING.keys()), rotation=45)
    ax.set_xlabel('TSS Distance Bin', fontsize=14)
    ax.set_ylabel('ROC AUC', fontsize=14)
    ax.set_title('ROC AUC by Distance Bin: Shorkie vs DREAM models', fontsize=16, pad=50)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.17), fontsize=14)
    fig.subplots_adjust(top=0.80)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,'distance_bin_roc.png'), dpi=300)
    plt.close(fig)

    # ─── PR plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11,5))
    sns.scatterplot(data=combined_pr, x='distance', y='pr_auc',
                    hue='base_model', palette=model_colors,
                    alpha=0.7, edgecolor='w', legend=False, ax=ax)

    for base_model, color in model_colors.items():
        subset = combined_pr[combined_pr.base_model == base_model]
        sns.regplot(data=subset, x='distance', y='pr_auc',
                    scatter=False, color=color, ax=ax,
                    label=base_model, line_kws={'linewidth': 2.5})

    ax.set_xticks(list(BIN_MAPPING.values()))
    ax.set_xticklabels(list(BIN_MAPPING.keys()), rotation=45)
    ax.set_xlabel('TSS Distance Bin', fontsize=14)
    ax.set_ylabel('AUPRC', fontsize=14)
    ax.set_title('AUPRC by Distance Bin: Shorkie vs DREAM models', fontsize=16, pad=50)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.17), fontsize=14)
    fig.subplots_adjust(top=0.80)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,'distance_bin_pr.png'), dpi=300)
    plt.close(fig)

    print("Done: plots saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()


####################################
# Old code for visualization
####################################
# #!/usr/bin/env python3
# import os
# import re
# import random

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm

# from sklearn.metrics import (
#     roc_curve, auc,
#     precision_recall_curve, average_precision_score,
# )

# # ─── Configuration ────────────────────────────────────────────────────────
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)

# # BINS        = [0, 200, 800, 1400, 2000, 3000, 6000, 8000]
# # BIN_LABELS  = ['0-200b','200b-1400b','800b-1kb','1kb-2kb','2kb-3kb','3kb-6kb','6kb-8kb']

# BINS        = [0, 1000, 2000, 3000, 6000, 8000]
# BIN_LABELS  = ['0-1kb','1kb-2kb','2kb-3kb','3kb-6kb','6kb-8kb']

# BIN_MAPPING = dict(zip(BIN_LABELS, BINS[1:]))

# FOLDS = [f"f{i}c0" for i in range(8)]

# MPRA_MODELS   = ["DREAM_Atten", "DREAM_RNN", "DREAM_CNN"]
# MPRA_NAME_MAP = {
#     "DREAM_Atten": "DREAM-Atten",
#     "DREAM_CNN":   "DREAM-CNN",
#     "DREAM_RNN":   "DREAM-RNN",
# }

# # Directories & files
# ROOT           = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
# YEAST_DIR      = os.path.join(ROOT, 'experiments','SUM_data_process','eQTL_exp','results')
# GTF_FILE       = os.path.join(ROOT, 'data','eQTL','neg_eQTLS','GCA_000146045_2.59.gtf')
# OUTPUT_DIR     = 'results/viz'
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def lighten_color(color, amount=0.5):
#     """
#     Lightens the given color by `amount` toward white.
#     amount=0  -> original color
#     amount=1  -> white
#     """
#     c = mcolors.to_rgb(color)
#     white = (1.0, 1.0, 1.0)
#     return mcolors.to_hex([ci + (wi - ci) * amount for ci, wi in zip(c, white)])


# def map_chromosome_to_roman(chrom):
#     roman = {
#         'chromosome1':'chrI','chromosome2':'chrII','chromosome3':'chrIII',
#         'chromosome4':'chrIV','chromosome5':'chrV','chromosome6':'chrVI',
#         'chromosome7':'chrVII','chromosome8':'chrVIII','chromosome9':'chrIX',
#         'chromosome10':'chrX','chromosome11':'chrXI','chromosome12':'chrXII',
#         'chromosome13':'chrXIII','chromosome14':'chrXIV','chromosome15':'chrXV',
#         'chromosome16':'chrXVI'
#     }
#     return roman.get(chrom, chrom)


# def parse_gtf_for_tss(gtf_path):
#     tss = {}
#     with open(gtf_path) as f:
#         for line in f:
#             if line.startswith('#'):
#                 continue
#             chrom, _, feat, start, end, _, strand, _, attrs = line.strip().split('\t')
#             if feat.lower() != 'gene':
#                 continue
#             m = re.search(r'gene_id "([^"]+)"', attrs)
#             if not m:
#                 continue
#             gid = m.group(1)
#             s, e = int(start), int(end)
#             tss[gid] = {
#                 'chrom': f"chr{chrom}",
#                 'tss': s if strand == '+' else e
#             }
#     return tss


# def calculate_tss_distance(key, tss_data):
#     parts = key.split('_')
#     if len(parts) < 2:
#         return np.nan
#     pos_str, gid = '_'.join(parts[:-1]), parts[-1]
#     try:
#         chrom, pos = pos_str.split(':')
#         pos = int(pos)
#     except ValueError:
#         return np.nan
#     info = tss_data.get(gid)
#     if not info or map_chromosome_to_roman(info['chrom']) != chrom:
#         return np.nan
#     return abs(pos - info['tss'])


# def process_mpra_data(pos_file, neg_file):
#     pos = pd.read_csv(pos_file, sep='\t'); pos['label'] = 1
#     neg = pd.read_csv(neg_file, sep='\t'); neg['label'] = 0
#     pos.rename(columns={'logSED': 'score'}, inplace=True)
#     neg.rename(columns={'logSED': 'score'}, inplace=True)
#     df = pd.concat([pos, neg], ignore_index=True).dropna(subset=['score'])
#     df['score'] = df['score'].abs()
#     p, n = df[df.label == 1], df[df.label == 0]
#     if len(p) > len(n):
#         p = p.sample(len(n), random_state=SEED)
#     else:
#         n = n.sample(len(p), random_state=SEED)
#     return pd.concat([p, n], ignore_index=True)


# def compute_metrics_by_bin(df, model_name):
#     metrics = []
#     for bin_label in BIN_LABELS:
#         sub = df[df.distance_bin == bin_label]
#         if len(sub) < 10:
#             continue
#         y, s = sub['label'], sub['score']
#         fpr, tpr, _ = roc_curve(y, s)
#         roc_val = auc(fpr, tpr)
#         prec, rec, _ = precision_recall_curve(y, s)
#         pr_val = average_precision_score(y, s)
#         metrics.append({
#             'model': model_name,
#             'distance_bin': bin_label,
#             'distance': BIN_MAPPING[bin_label],
#             'roc_auc': roc_val,
#             'pr_auc': pr_val
#         })
#     return metrics


# def main():
#     tss_data = parse_gtf_for_tss(GTF_FILE)

#     # MPRA metrics
#     mpra_metrics = []
#     for mdl in MPRA_MODELS:
#         base = os.path.join(ROOT, 'experiments','SUM_data_process',
#                             'eQTL_MPRA_models_eval','results', mdl)
#         pos_f = os.path.join(base, 'final_pos_predictions.tsv')
#         for ns in [1, 2, 3, 4]:
#             neg_f = os.path.join(base, f'final_neg_predictions_{ns}.tsv')
#             df    = process_mpra_data(pos_f, neg_f)
#             label = MPRA_NAME_MAP[mdl]
#             df['model'] = f"{label} (neg_set {ns})"
#             df['distance']     = df['tss_dist']
#             df['distance_bin'] = pd.cut(df['distance'], bins=BINS, labels=BIN_LABELS)
#             mpra_metrics += compute_metrics_by_bin(df, f"{label} (neg_set {ns})")
#     mpra_df = pd.DataFrame(mpra_metrics)

#     # Yeast (Shorkie) metrics
#     yeast_metrics = []
#     for neg_set in [1, 2, 3, 4]:
#         yeast_folds = []
#         for fold in FOLDS:
#             path = os.path.join(YEAST_DIR, f'set{neg_set}', f'yeast_eqtl_{fold}.tsv')
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"Missing TSV for fold {fold}: {path}")
#             df = (pd.read_csv(path, sep='\t', usecols=['Position_Gene','score','label'])
#                     .drop_duplicates(['Position_Gene'])
#                     .dropna(subset=['score','label']))
#             df['score'] = df['score'].abs()
#             p, n = df[df.label==1], df[df.label==0]
#             if len(p) > len(n):
#                 p = p.sample(len(n), random_state=SEED)
#             else:
#                 n = n.sample(len(p), random_state=SEED)
#             df_bal = pd.concat([p, n], ignore_index=True)
#             df_bal['fold'] = fold
#             df_bal['model'] = 'Shorkie'
#             yeast_folds.append(df_bal)

#         yeast_all = pd.concat(yeast_folds, ignore_index=True)
#         yeast_all['distance'] = yeast_all['Position_Gene'] \
#             .apply(lambda k: calculate_tss_distance(k, tss_data))
#         yeast_all = yeast_all.dropna(subset=['distance'])
#         yeast_all['distance_bin'] = pd.cut(yeast_all['distance'], bins=BINS, labels=BIN_LABELS)
#         yeast_metrics += compute_metrics_by_bin(yeast_all, f'Shorkie (neg_set {neg_set})')

#     yeast_df = pd.DataFrame(yeast_metrics)

#     # Combine metrics
#     combined_roc = pd.concat([
#         mpra_df[['model','distance','roc_auc']],
#         yeast_df[['model','distance','roc_auc']]
#     ], ignore_index=True)
#     combined_pr  = pd.concat([
#         mpra_df[['model','distance','pr_auc']],
#         yeast_df[['model','distance','pr_auc']]
#     ], ignore_index=True)

#     model_colors = {
#         'DREAM-Atten': 'tab:blue',
#         'DREAM-CNN':   'tab:orange',
#         'DREAM-RNN':   'tab:green',
#         'Shorkie':     'tab:red',
#     }

#     # Build a flat palette so every “model (neg_set i)” gets its model’s color
#     palette = {}
#     for model, color in model_colors.items():
#         for i in range(1, 5):
#             # i=1 → amount=0.0 (original)
#             # i=2 → amount=0.2
#             # i=3 → amount=0.4
#             # i=4 → amount=0.6
#             amount = 0.2 * (i - 1)
#             label = f"{model} (neg_set {i})"
#             palette[label] = lighten_color(color, amount)

#     # ─── ROC plot ────────────────────────────────────────────────────────────
#     plt.figure(figsize=(11,5))
#     sns.scatterplot(data=combined_roc, x='distance', y='roc_auc',
#                     hue='model', palette=palette, alpha=0.7, edgecolor='w')

#     for model, color in model_colors.items():
#         subset = combined_roc[combined_roc.model.str.startswith(model)]
#         sns.regplot(data=subset, x='distance', y='roc_auc',
#                     scatter=False, color=color,
#                     label=f'{model} Trend', line_kws={'linewidth': 2.5})
    
#     plt.xticks(list(BIN_MAPPING.values()), list(BIN_MAPPING.keys()), rotation=45)
#     plt.xlabel('TSS Distance Bin', fontsize=14)
#     plt.ylabel('ROC AUC', fontsize=14)
#     plt.title('ROC AUC by Distance Bin', fontsize=16)
#     plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR,'distance_bin_roc.png'), dpi=300)
#     plt.close()

#     # ─── PR plot ─────────────────────────────────────────────────────────────
#     plt.figure(figsize=(11,5))
#     sns.scatterplot(data=combined_pr, x='distance', y='pr_auc',
#                     hue='model', palette=palette, alpha=0.7, edgecolor='w')

#     for model, color in model_colors.items():
#         subset = combined_pr[combined_pr.model.str.startswith(model)]
#         sns.regplot(data=subset, x='distance', y='pr_auc',
#                     scatter=False, color=color,
#                     label=f'{model} Trend', line_kws={'linewidth': 2.5})
    
#     plt.xticks(list(BIN_MAPPING.values()), list(BIN_MAPPING.keys()), rotation=45)
#     plt.xlabel('TSS Distance Bin', fontsize=14)
#     plt.ylabel('PR AUC', fontsize=14)
#     plt.title('PR AUC by Distance Bin', fontsize=16)
#     plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR,'distance_bin_pr.png'), dpi=300)
#     plt.close()

#     print("Done: plots saved to", OUTPUT_DIR)

# if __name__ == "__main__":
#     main()
