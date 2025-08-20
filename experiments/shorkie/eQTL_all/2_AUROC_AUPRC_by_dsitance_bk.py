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

# Which experiments / their display names
# EXPS           = ["Caudal_etal", "Kita_etal"]
EXPS           = ["Caudal_etal"]
EXPS_NAME_MAP  = {
    "Caudal_etal": "Caudal et al.",  
    "Kita_etal":   "Kita et al.",
}

# Negative‐set indices
NEG_SETS    = [1, 2, 3, 4]
# NEG_SETS    = [1]

# Default distance bins (overridden per EXP)
BINS        = [0, 1000, 2000, 3000, 5000, 8000]
BIN_LABELS  = ['0-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb']
BIN_MAP     = dict(zip(BIN_LABELS, BINS[1:]))

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
    """Parse GTF → dict gene_id → {chrom, tss}."""
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
    """|variant_pos – gene_TSS| or NaN."""
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

def balance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all positives and all negatives (no down‐sampling here)."""
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    if len(pos) > len(neg):
        pos = pos.sample(len(neg), random_state=SEED)
    else:
        neg = neg.sample(len(pos), random_state=SEED)
    return pd.concat([pos, neg], ignore_index=True)


def process_shorkie(exp: str, neg_set: int, tss_data: dict) -> pd.DataFrame:
    """
    Load the single TSV for this dataset+neg_set:
      ROOT/.../eQTL_all/results/negset_{neg_set}/{exp.lower()}_scores.tsv
    and compute distance to TSS.
    """
    fname = f"{exp.lower()}_scores.tsv"
    path = os.path.join(
        ROOT, 'experiments','SUM_data_process','eQTL_all','results',
        f'negset_{neg_set}', fname
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No Shorkie file for {exp} neg{neg_set}: {path}")
    df = pd.read_csv(path, sep='\t', usecols=[
        'Position_Gene','Chr','ChrPos','score','label'
    ]).drop_duplicates(['Position_Gene'])
    df['ensemble'] = df['score'].abs()
    df['label']    = df['label'].astype(int)
    df['distance'] = df['Position_Gene'].map(
        lambda x: calculate_tss_distance(x, tss_data)
    )
    return df[['Position_Gene','Chr','ChrPos','ensemble','label','distance']]


def process_mpra(model: str, neg_set: int, mpra_base: str) -> pd.DataFrame:
    """Exactly as before."""
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
    sub = df.copy()
    sub['distance_bin'] = pd.cut(sub.distance, bins=BINS, labels=BIN_LABELS)
    for b in BIN_LABELS:
        seg = sub[sub.distance_bin==b]
        if len(seg) < 10: continue
        y, s = seg.label, seg.score
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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global BINS, BIN_LABELS, BIN_MAP
    tss_data = parse_gtf_for_tss(GTF_FILE)

    for exp in EXPS:
        # per‐experiment binning
        if exp == "Caudal_etal":
            # BINS       = [0, 1000, 2000, 3000, 5000, 8000]
            # BIN_LABELS = ['0-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb']
            BINS       = [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000]#, 8000]
            BIN_LABELS = ['0-1kb','1kb-2kb','2kb-3kb','3k-4kb','4k-5kb','5k-6kb','6k-8kb']#,'7kb-8kb']
        else:
            # BINS       = [0, 1000, 1500, 2000, 3000]
            # BIN_LABELS = ['0-1kb','1kb-1.5kb','1.5kb-2kb','2kb-3kb']
            # BINS       = [0, 300, 500, 1000, 1500, 2000, 3000]
            # BIN_LABELS = ['0-0.3kb','0.3-0.5kb','0.5-1kb','1kb-1.5kb','1.5kb-2kb','2kb-3kb']
            BINS       = [0, 300, 500, 1000, 2000, 3000]
            BIN_LABELS = ['0-0.3kb','0.3-0.5kb','0.5-1kb','1kb-2kb','2kb-3kb']
        BIN_MAP = dict(zip(BIN_LABELS, BINS[1:]))

        # MPRA eval paths
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
            print(f"  Shorkie neg{ns}: {eqtl_df.label.sum()} pos, {(eqtl_df.label==0).sum()} neg")
            sh_df = eqtl_df.rename(columns={'ensemble':'score'})
            exp_metrics += compute_metrics_by_bin(sh_df, f"Shorkie (neg{ns})")

            for model in MPRA_MODELS:
                display = MPRA_NAME_MAP[model]
                mpra_df = process_mpra(model, ns, MPRA_EVAL_BASE)
                common = eqtl_df[['Position_Gene','ensemble','label','distance']].merge(
                    mpra_df[['Position_Gene','score']], on='Position_Gene'
                )
                print(f"  {display} neg{ns}: {common.label.sum()} pos, {(common.label==0).sum()} neg")
                print("Head of common DataFrame:")
                print(common.head())
                
                exp_metrics += compute_metrics_by_bin(
                    common[['label','distance','score']], f"{display} (neg{ns})"
                )

        # # Build DataFrame + xticks with counts
        # df = pd.DataFrame(exp_metrics)
        # df['base'] = df['model'].str.replace(r"\s*\(neg[0-9]\)", '', regex=True)

        # print(f"Total metrics for {exp}: {len(df)} rows")

        
        
        # counts = df.groupby('distance')[['n_pos','n_neg']].first()
        # xticks = [
        #     f"{lbl}\nPos: {counts.loc[BIN_MAP[lbl],'n_pos']}\nNeg: {counts.loc[BIN_MAP[lbl],'n_neg']}"
        #     for lbl in BIN_LABELS
        # ]

        # # ROC by distance
        # plt.figure(figsize=(9.5,6))
        # sns.scatterplot(data=df[df.base=='Shorkie'], x='distance', y='roc_auc',
        #                 color='black', alpha=0.7, edgecolor='w', s=50, legend=False)
        # sns.regplot(data=df[df.base=='Shorkie'], x='distance', y='roc_auc',
        #             scatter=False, color='black', label='Shorkie')
        # sns.scatterplot(data=df[df.base!='Shorkie'], x='distance', y='roc_auc',
        #                 hue='base', alpha=0.7, edgecolor='w', s=50, legend=False)
        # for base in df.base.unique():
        #     if base=='Shorkie': continue
        #     sns.regplot(data=df[df.base==base], x='distance', y='roc_auc',
        #                 scatter=False, label=base)

        # plt.xticks(list(BIN_MAP.values()), xticks, fontsize=9, rotation=0, ha='center')
        # plt.yticks(fontsize=12)
        # plt.gcf().subplots_adjust(bottom=0.25)
        # plt.xlabel('TSS Distance Bin', fontsize=14, labelpad=20)
        # plt.ylabel('AUROC', fontsize=14)
        # plt.title(f'{EXPS_NAME_MAP[exp]}: AUROC by Distance Bin', fontsize=16, pad=50)
        # plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.15), fontsize=12)
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT_DIR, f'{exp}_distance_bin_roc.png'), dpi=300)
        # plt.close()

        # # PR by distance
        # plt.figure(figsize=(9.5,6))
        # sns.scatterplot(data=df[df.base=='Shorkie'], x='distance', y='pr_auc',
        #                 color='black', alpha=0.7, edgecolor='w', s=50,legend=False)
        # sns.regplot(data=df[df.base=='Shorkie'], x='distance', y='pr_auc',
        #             scatter=False, color='black', label='Shorkie')
        # sns.scatterplot(data=df[df.base!='Shorkie'], x='distance', y='pr_auc',
        #                 hue='base', alpha=0.7, edgecolor='w', s=50, legend=False)
        # for base in df.base.unique():
        #     if base=='Shorkie': continue
        #     sns.regplot(data=df[df.base==base], x='distance', y='pr_auc',
        #                 scatter=False, label=base)

        # plt.xticks(list(BIN_MAP.values()), xticks, fontsize=9, rotation=0, ha='center')
        # plt.yticks(fontsize=12)
        # plt.gcf().subplots_adjust(bottom=0.25)
        # plt.xlabel('TSS Distance Bin', fontsize=14, labelpad=20)
        # plt.ylabel('AUPRC', fontsize=14)
        # plt.title(f'{EXPS_NAME_MAP[exp]}: AUPRC by Distance Bin', fontsize=16, pad=50)
        # plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.15), fontsize=12)
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT_DIR, f'{exp}_distance_bin_pr.png'), dpi=300)
        # plt.close()

        # print(f"Done – plots for {exp} written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
