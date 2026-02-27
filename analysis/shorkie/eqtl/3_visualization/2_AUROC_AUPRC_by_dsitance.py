#!/usr/bin/env python3
"""
AUROC / AUPRC stratified by variant-to-TSS distance bins.

Changes from previous version:
  - No class balancing
  - Per-set metrics aggregated as mean ± SEM with error bars
  - Cleaner plotting with errorbar + trend line
"""
import os
import re
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# ─── Configuration ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MPRA_MODELS = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

EXPS = ["caudal_etal", "kita_etal", "Renganaath_etal"]
EXPS_NAME_MAP = {
    "caudal_etal":     "Caudal et al.",
    "kita_etal":       "Kita et al.",
    "Renganaath_etal": "Renganaath et al.",
}

NEG_SETS = [1, 2, 3, 4]

ROOT = args.root_dir
OUTPUT_DIR = os.path.join(ROOT, "revision_experiments/eQTL/viz_new/results/viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
MODEL_COLORS = {
    'Shorkie':              '#2196F3',
    'Shorkie_LM':           '#E91E63',
    'Shorkie_Random_Init':  '#FF9800',
    'DREAM-Atten':          '#4CAF50',
    'DREAM-CNN':            '#9C27B0',
    'DREAM-RNN':            '#795548',
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def process_shorkie(exp: str, neg_set: int) -> pd.DataFrame:
    base_dir = os.path.join(
        ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results',
        f'negset_{neg_set}'
    )
    # Base Shorkie
    df_orig = pd.read_csv(
        os.path.join(base_dir, f"{exp}_Shorkie_scores.tsv"),
        sep='\t', usecols=['Position_Gene', 'logSED_agg', 'label', 'distance']
    )
    df_orig['Position_Gene'] = df_orig['Position_Gene'].astype(str).str.strip()
    df_orig = df_orig.drop_duplicates(['Position_Gene'])
    df_orig['Shorkie'] = df_orig['logSED_agg'].abs()
    df_orig = df_orig.drop(columns=['logSED_agg'])

    # LM
    df_lm = pd.read_csv(
        os.path.join(base_dir, f"{exp}_Shorkie_LM_scores.tsv"),
        sep='\t', usecols=['Position_Gene', 'LLR', 'label']
    )
    df_lm['Position_Gene'] = df_lm['Position_Gene'].astype(str).str.strip()
    df_lm = df_lm.drop_duplicates(['Position_Gene'])
    df_lm['Shorkie_LM'] = df_lm['LLR'].abs()
    df_lm = df_lm.drop(columns=['LLR', 'label'])

    # Random Init
    df_ri = pd.read_csv(
        os.path.join(base_dir, f"{exp}_Shorkie_Random_Init_scores.tsv"),
        sep='\t', usecols=['Position_Gene', 'logSED_agg']
    )
    df_ri['Position_Gene'] = df_ri['Position_Gene'].astype(str).str.strip()
    df_ri = df_ri.drop_duplicates(['Position_Gene'])
    df_ri['Shorkie_Random_Init'] = df_ri['logSED_agg'].abs()
    df_ri = df_ri.drop(columns=['logSED_agg'])

    merged = pd.merge(df_orig, df_lm, on='Position_Gene', how='inner')
    merged = pd.merge(merged, df_ri, on='Position_Gene', how='inner')
    merged['label'] = merged['label'].astype(int)
    return merged[['Position_Gene', 'Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init', 'label', 'distance']]


def process_mpra(model: str, neg_set: int, mpra_base: str) -> pd.DataFrame:
    model_dir = os.path.join(mpra_base, model)
    pos = pd.read_csv(os.path.join(model_dir, 'final_pos_predictions.tsv'),
                       sep='\t', usecols=['Position_Gene', 'logSED'])
    neg = pd.read_csv(os.path.join(model_dir, f'final_neg_predictions_{neg_set}.tsv'),
                       sep='\t', usecols=['Position_Gene', 'logSED'])
    pos['label'], neg['label'] = 1, 0
    for df in (pos, neg):
        df.rename(columns={'logSED': 'score'}, inplace=True)
        df['score'] = df['score'].abs()
    return pd.concat([pos, neg], ignore_index=True)[['Position_Gene', 'score', 'label']]


def get_mpra_base(exp):
    if exp == "caudal_etal":
        return os.path.join(ROOT, 'experiments', 'SUM_data_process', 'eQTL',
                            'eqtl_MPRA_modeals_eval', 'eQTL_MPRA_models_eval_caudal_etal', 'results')
    elif exp == "kita_etal":
        return os.path.join(ROOT, 'experiments', 'SUM_data_process', 'eQTL',
                            'eqtl_MPRA_modeals_eval', 'eQTL_MPRA_models_eval_kita_etal_select', 'results')
    elif exp == "Renganaath_etal":
        return os.path.join(ROOT, 'revision_experiments', 'eQTL',
                            'eQTL_MPRA_models_eval_Renganaath_etal', 'results')
    return None


def compute_metrics_by_bin(df, model_name, bins, bin_labels, bin_map):
    """Compute AUROC / AUPRC per distance bin for ONE negative set.
    Returns a list of row dicts for a flat DataFrame."""
    out = []
    df = df.dropna(subset=['score', 'distance'])
    df = df.copy()
    df['distance_bin'] = pd.cut(df['distance'], bins=bins, labels=bin_labels)
    for b in bin_labels:
        seg = df[(df.distance_bin == b) & (df.score > 0)]
        if len(seg) < 10:
            continue
        y, s = seg['label'], seg['score']
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        out.append({
            'model':        model_name,
            'distance_bin': b,
            'distance':     bin_map[b],
            'n_pos':        int((y == 1).sum()),
            'n_neg':        int((y == 0).sum()),
            'roc_auc':      auc(fpr, tpr),
            'pr_auc':       average_precision_score(y, s),
        })
        print(f"\t  {model_name} {b}: {len(seg)} rows, "
              f"{out[-1]['n_pos']} pos, {out[-1]['n_neg']} neg, "
              f"ROC AUC: {out[-1]['roc_auc']:.3f}, "
              f"PR AUC: {out[-1]['pr_auc']:.3f}")
    return out


def main():
    for exp in EXPS:
        print(f"\n=== {EXPS_NAME_MAP[exp]} ===")

        # Per-experiment binning
        if exp == "caudal_etal":
            bins = [0, 1000, 2000, 3000, 4000, 5000]
            bin_labels = ['0-1kb', '1kb-2kb', '2kb-3kb', '3k-4kb', '4k-5kb']
        else:
            bins = [0, 500, 1200, 2000, 3000]
            bin_labels = ['0-0.5kb', '0.5-1.2kb', '1.2kb-2kb', '2kb-3kb']

        bin_map = dict(zip(bin_labels, bins[1:]))

        mpra_base = get_mpra_base(exp)
        shorkie_cols = ['Shorkie']
        mpra_cols = [MPRA_NAME_MAP[m] for m in MPRA_MODELS] if mpra_base else []
        model_cols = shorkie_cols + mpra_cols

        # Collect per-(model, neg_set, bin) metrics into a flat list
        exp_metrics = []

        for ns in NEG_SETS:
            print(f"  neg{ns}...")
            try:
                combined_df = process_shorkie(exp, ns)
            except FileNotFoundError as e:
                print(f"    SKIP Shorkie: {e}")
                continue

            if mpra_base:
                for model in MPRA_MODELS:
                    try:
                        mpra_df = process_mpra(model, ns, mpra_base)
                        col_name = MPRA_NAME_MAP[model]
                        mpra_df = mpra_df.rename(columns={'score': col_name})
                        combined_df = combined_df.merge(
                            mpra_df[['Position_Gene', col_name]],
                            on='Position_Gene', how='inner'
                        )
                    except FileNotFoundError:
                        pass

            combined_df = combined_df.dropna(subset=model_cols)

            for model_col in model_cols:
                temp_df = combined_df[['Position_Gene', 'label', 'distance']].copy()
                temp_df['score'] = combined_df[model_col]
                exp_metrics += compute_metrics_by_bin(
                    temp_df, f"{model_col} (neg{ns})", bins, bin_labels, bin_map
                )

        # Assemble flat DataFrame
        df = pd.DataFrame(exp_metrics)
        if df.empty:
            print(f"  No metrics for {exp}, skipping plots.")
            continue

        # Strip "(negN)" suffix to get base model name
        df['base'] = df['model'].str.replace(r"\s*\(neg[0-9]\)", '', regex=True)

        # Build x-tick labels with sample counts
        counts = df.groupby('distance')[['n_pos', 'n_neg']].first()
        xtick_labels = []
        for lbl in bin_labels:
            d_val = bin_map[lbl]
            if d_val in counts.index:
                n_p = counts.loc[d_val, 'n_pos']
                n_n = counts.loc[d_val, 'n_neg']
            else:
                n_p, n_n = 0, 0
            xtick_labels.append(f"{lbl}\nPos: {n_p}\nNeg: {n_n}")

        # Filter for relevant models
        target_bases = ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init'] + list(MPRA_NAME_MAP.values())
        subdf = df[df.base.isin(target_bases)]

        # Custom palette for seaborn
        palette = {m: MODEL_COLORS.get(m, 'gray') for m in target_bases}

        # ─── ROC by distance plot (scatter + regression with CI) ─────
        plt.figure(figsize=(9, 7.5))
        sns.scatterplot(
            data=subdf, x='distance', y='roc_auc',
            hue='base', style='base', marker='o', s=60, edgecolor='w',
            palette=palette, legend=False
        )
        for base in subdf.base.unique():
            sns.regplot(
                data=subdf[subdf.base == base],
                x='distance', y='roc_auc',
                scatter=False, label=base,
                color=MODEL_COLORS.get(base, 'gray')
            )

        plt.xticks(list(bin_map.values()), xtick_labels, fontsize=11, rotation=0, ha='center')
        plt.xlabel('TSS Distance Bin', fontsize=14)
        plt.ylabel('AUROC', fontsize=14)
        plt.title(f"{EXPS_NAME_MAP[exp]}: AUROC by Distance Bin", fontsize=17.5)
        plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=17.5, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{exp}_distance_bin_roc.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ─── PR by distance plot (scatter + regression with CI) ─────
        plt.figure(figsize=(9, 7.5))
        sns.scatterplot(
            data=subdf, x='distance', y='pr_auc',
            hue='base', style='base', marker='o', s=60, edgecolor='w',
            palette=palette, legend=False
        )
        for base in subdf.base.unique():
            sns.regplot(
                data=subdf[subdf.base == base],
                x='distance', y='pr_auc',
                scatter=False, label=base,
                color=MODEL_COLORS.get(base, 'gray')
            )

        plt.xticks(list(bin_map.values()), xtick_labels, fontsize=11, rotation=0, ha='center')
        plt.xlabel('TSS Distance Bin', fontsize=14)
        plt.ylabel('AUPRC', fontsize=14)
        plt.title(f"{EXPS_NAME_MAP[exp]}: AUPRC by Distance Bin", fontsize=17.5)
        plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=17.5, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{exp}_distance_bin_pr.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Done – plots for {exp} written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
