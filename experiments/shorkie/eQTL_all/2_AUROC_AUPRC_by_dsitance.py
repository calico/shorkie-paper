#!/usr/bin/env python3
import os
import re
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# ─── Configuration ───────────────────────────────────────────────────────────
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

# Which MPRA models to compare
MPRA_MODELS    = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP  = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}
MPRA_BASES = list(MPRA_NAME_MAP.values())

# Which experiments / their display names
EXPS           = ["Caudal_etal", "Kita_etal"]
EXPS_NAME_MAP  = {
    "Caudal_etal": "Caudal et al.",
    "Kita_etal":   "Kita et al.",
}

# Negative‐set indices
NEG_SETS    = [1, 2, 3, 4]

BINS       = []
BIN_LABELS = []
BIN_MAP    = {}

# GTF for TSS lookup
ROOT        = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
GTF_FILE    = os.path.join(ROOT, 'data','eQTL','neg_eQTLS','GCA_000146045_2.59.gtf')

# Where to save the figures
OUTPUT_DIR  = os.path.join(ROOT, 'experiments','SUM_data_process','eQTL_all','results','viz')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def map_chromosome_to_roman(chrom: str) -> str:
    m = {
      'chromosome1':'chrI','chromosome2':'chrII','chromosome3':'chrIII',
      'chromosome4':'chrIV','chromosome5':'chrV','chromosome6':'chrVI',
      'chromosome7':'chrVII','chromosome8':'chrVIII','chromosome9':'chrIX',
      'chromosome10':'chrX','chromosome11':'chrXI','chromosome12':'chrXII',
      'chromosome13':'chrXIII','chromosome14':'chrXIV','chromosome15':'chrXV',
      'chromosome16':'chrXVI'
    }
    return m.get(chrom, chrom)


def parse_gtf_for_tss(gtf_path):
    tss = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'): continue
            chrom,_,feat,start,end,_,strand,_,attrs = line.strip().split('\t')
            if feat.lower() != 'gene': continue
            m = re.search(r'gene_id "([^"]+)"', attrs)
            if not m: continue
            gid = m.group(1)
            pos = int(start) if strand=='+' else int(end)
            tss[gid] = {'chrom': f"chr{chrom}", 'tss': pos}
    return tss


def calculate_tss_distance(pos_gene: str, tss_data: dict) -> float:
    try:
        coord, gid = pos_gene.rsplit('_', 1)
        chrom, pos = coord.split(':')
        pos = int(pos)
    except ValueError:
        return np.nan
    info = tss_data.get(gid)
    if not info or map_chromosome_to_roman(info['chrom']) != chrom:
        return np.nan
    return abs(pos - info['tss'])


def process_shorkie(exp: str, neg_set: int, tss_data: dict) -> pd.DataFrame:
    fname = f"{exp.lower()}_scores.tsv"
    path = os.path.join(
        ROOT, 'experiments','SUM_data_process','eQTL_all','results',
        f'negset_{neg_set}', fname
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No Shorkie file for {exp} neg{neg_set}: {path}")
    df = pd.read_csv(path, sep='\t', usecols=[
        'Position_Gene','Chr','ChrPos','logSED_agg','logSED_avg','label'
    ]).drop_duplicates(['Position_Gene'])
    df['agg']      = df['logSED_agg'].abs()
    df['avg']      = df['logSED_avg'].abs()
    df['label']    = df['label'].astype(int)
    df['distance'] = df['Position_Gene'].map(
        lambda x: calculate_tss_distance(x, tss_data)
    )
    return df[['Position_Gene','Chr','ChrPos','agg','avg','label','distance']]


def process_mpra(model: str, neg_set: int, mpra_base: str) -> pd.DataFrame:
    model_dir = os.path.join(mpra_base, model)
    pos_f = os.path.join(model_dir, 'final_pos_predictions.tsv')
    neg_f = os.path.join(model_dir, f'final_neg_predictions_{neg_set}.tsv')
    if not (os.path.exists(pos_f) and os.path.exists(neg_f)):
        raise FileNotFoundError(f"MPRA missing for {model} neg{neg_set}")
    pos = pd.read_csv(pos_f, sep='\t', usecols=['Position_Gene','logSED']); pos['label']=1
    neg = pd.read_csv(neg_f, sep='\t', usecols=['Position_Gene','logSED']); neg['label']=0
    for df in (pos, neg):
        df.rename(columns={'logSED':'score'}, inplace=True)
        df['score'] = df['score'].abs()
    return pd.concat([pos, neg], ignore_index=True)[['Position_Gene','score','label']]


def compute_metrics_by_bin(df: pd.DataFrame, model_name: str):
    out = []
    df = df.dropna(subset=['score'])
    sub = df.copy()
    sub['distance_bin'] = pd.cut(sub.distance, bins=BINS, labels=BIN_LABELS)
    for b in BIN_LABELS:
        seg = sub[sub.distance_bin == b]
        if len(seg) < 10:
            continue
        y = seg.label
        s = seg.score
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        out.append({
            'model':        model_name,
            'distance_bin': b,
            'distance':     BIN_MAP[b],
            'n_pos':        int((y==1).sum()),
            'n_neg':        int((y==0).sum()),
            'roc_auc':      auc(fpr, tpr),
            'pr_auc':       average_precision_score(y, s),
        })
        print(f"\t  {model_name} {b}: {len(seg)} rows, "
              f"{out[-1]['n_pos']} pos, {out[-1]['n_neg']} neg, "
              f"ROC AUC: {out[-1]['roc_auc']:.3f}, "
              f"PR AUC: {out[-1]['pr_auc']:.3f}")
    return out


def main():
    global BINS, BIN_LABELS, BIN_MAP
    tss_data = parse_gtf_for_tss(GTF_FILE)

    for exp in EXPS:
        # per-experiment binning
        if exp == "Caudal_etal":
            BINS       = [0, 1000, 2000, 3000, 4000, 5000]
            BIN_LABELS = ['0-1kb','1kb-2kb','2kb-3kb','3k-4kb','4k-5kb']
        else:
            BINS       = [0, 500, 1200, 2000, 3000]
            BIN_LABELS = ['0-0.5kb','0.5-1.2kb','1.2kb-2kb','2kb-3kb']

        BIN_MAP = dict(zip(BIN_LABELS, BINS[1:]))

        # choose the correct MPRA base path
        if exp == "Caudal_etal":
            MPRA_EVAL_BASE = os.path.join(
                ROOT, 'experiments','SUM_data_process','eQTL_MPRA_models_eval','results'
            )
        else:
            MPRA_EVAL_BASE = os.path.join(
                ROOT, 'experiments','SUM_data_process',
                'eQTL_MPRA_models_eval_kita_etal_select','results'
            )

        exp_metrics = []
        for ns in NEG_SETS:
            print(f"Processing {exp} neg{ns}…")
            eqtl_df = process_shorkie(exp, ns, tss_data)

            for model in MPRA_MODELS:
                display = MPRA_NAME_MAP[model]
                mpra_df = process_mpra(model, ns, MPRA_EVAL_BASE)

                common = eqtl_df[['Position_Gene','label','distance','agg','avg']].merge(
                    mpra_df[['Position_Gene','score']], on='Position_Gene'
                )

                # Shorkie_agg
                agg_df = common[['label','distance','agg']].rename(columns={'agg':'score'})
                exp_metrics += compute_metrics_by_bin(agg_df, f"Shorkie_agg (neg{ns})")

                # Shorkie_avg
                avg_df = common[['label','distance','avg']].rename(columns={'avg':'score'})
                exp_metrics += compute_metrics_by_bin(avg_df, f"Shorkie_avg (neg{ns})")

                # MPRA
                mpra_common = common[['label','distance','score']]
                exp_metrics += compute_metrics_by_bin(mpra_common, f"{display} (neg{ns})")

        # assemble DataFrame
        df = pd.DataFrame(exp_metrics)
        df['base'] = df['model'].str.replace(r"\s*\(neg[0-9]\)", '', regex=True)
        counts = df.groupby('distance')[['n_pos','n_neg']].first()
        xticks = [
            f"{lbl}\nPos: {counts.loc[BIN_MAP[lbl],'n_pos']}\nNeg: {counts.loc[BIN_MAP[lbl],'n_neg']}"
            for lbl in BIN_LABELS
        ]

        # ─── make separate plots for agg vs avg ─────────────────────────────────
        for metric in ['agg','avg']:
            key   = f"Shorkie_{metric}"
            subdf = df[(df.base==key) | (df.base.isin(MPRA_BASES))]

            # ROC plot
            plt.figure(figsize=(9.5,6))
            sns.scatterplot(
                data=subdf, x='distance', y='roc_auc',
                hue='base', style='base', marker='o', s=60, edgecolor='w',
                legend=False                  # ← hide dot labels
            )
            # for base in subdf.base.unique():
            #     sns.regplot(
            #         data=subdf[subdf.base==base],
            #         x='distance', y='roc_auc',
            #         scatter=False, label=base
            #     )
            for base in subdf.base.unique():
                plot_label = 'Shorkie' if base.startswith('Shorkie') else base
                sns.regplot(
                    data=subdf[subdf.base==base],
                    x='distance', y='roc_auc',
                    scatter=False, label=plot_label
                )

            plt.xticks(list(BIN_MAP.values()), xticks, fontsize=9, rotation=0, ha='center')
            plt.xlabel('TSS Distance Bin', fontsize=14)
            plt.ylabel('AUROC', fontsize=14)
            plt.title(f"{EXPS_NAME_MAP[exp]}: AUROC by Distance Bin", fontsize=18, pad=40)
            # plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.15))
            plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.12), fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{exp}_{metric}_distance_bin_roc.png'), dpi=300)
            plt.close()

            # PR plot
            plt.figure(figsize=(9.5,6))
            sns.scatterplot(
                data=subdf, x='distance', y='pr_auc',
                hue='base', style='base', marker='o', s=60, edgecolor='w',
                legend=False                  # ← hide dot labels
            )
            # for base in subdf.base.unique():
            #     sns.regplot(
            #         data=subdf[subdf.base==base],
            #         x='distance', y='pr_auc',
            #         scatter=False, label=base
            #     )
            for base in subdf.base.unique():
                plot_label = 'Shorkie' if base.startswith('Shorkie') else base
                sns.regplot(
                    data=subdf[subdf.base==base],
                    x='distance', y='pr_auc',
                    scatter=False, label=plot_label
                )
            plt.xticks(list(BIN_MAP.values()), xticks, fontsize=9, rotation=0, ha='center')
            plt.xlabel('TSS Distance Bin', fontsize=14)
            plt.ylabel('AUPRC', fontsize=14)
            plt.title(f"{EXPS_NAME_MAP[exp]}: AUPRC by Distance Bin", fontsize=18, pad=40)
            # plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.15))
            plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.12), fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{exp}_{metric}_distance_bin_pr.png'), dpi=300)
            plt.close()

        print(f"Done – plots for {exp} written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
