#!/usr/bin/env python3
"""
Model interpretability comparison: boxplots, ECDF, Mann-Whitney U.

Changes from previous version:
  - No class balancing
  - Ensemble across negative sets: per-set quantile medians → mean ± SEM
  - Ensemble boxplot with individual neg-set medians overlaid as points
"""
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from scipy.stats import mannwhitneyu

# ─── Config ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = args.root_dir
NEG_NUM = 4
EXPS = ["caudal_etal", "kita_etal", "Renganaath_etal"]
MPRA_MODELS = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

MODEL_COLORS = {
    'Shorkie':              '#2196F3',
    'Shorkie_LM':           '#E91E63',
    'Shorkie_Random_Init':  '#FF9800',
    'DREAM-Atten':          '#4CAF50',
    'DREAM-CNN':            '#9C27B0',
    'DREAM-RNN':            '#795548',
}

# ─── Helpers ──────────────────────────────────────────────────────────────

def load_shorkie_data(exp: str, neg_set: int, variant: str) -> pd.DataFrame:
    base_dir = os.path.join(ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results',
                            f'negset_{neg_set}')
    if variant == 'Shorkie_LM':
        fname, col = f"{exp}_Shorkie_LM_scores.tsv", 'LLR'
    elif variant == 'Shorkie_Random_Init':
        fname, col = f"{exp}_Shorkie_Random_Init_scores.tsv", 'logSED_agg'
    else:
        fname, col = f"{exp}_Shorkie_scores.tsv", 'logSED_agg'

    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, sep='\t', usecols=['Position_Gene', col, 'label'])
    df = df.drop_duplicates('Position_Gene')
    df['score'] = df[col].abs()
    return df[['Position_Gene', 'score', 'label']]


def load_shorkie_all(neg_set: int, variant: str) -> pd.DataFrame:
    dfs = []
    for exp in EXPS:
        try:
            dfs.append(load_shorkie_data(exp, neg_set, variant))
        except FileNotFoundError:
            pass
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    return (all_df.groupby('Position_Gene', as_index=False)
            .agg(score=('score', 'mean'), label=('label', 'first')))


def load_dream_exp(model: str, exp: str, neg_set: int) -> pd.DataFrame:
    base = None
    if exp == "caudal_etal":
        base = os.path.join(ROOT, 'experiments', 'SUM_data_process', 'eQTL',
                            'eqtl_MPRA_modeals_eval', 'eQTL_MPRA_models_eval_caudal_etal', 'results')
    elif exp == "kita_etal":
        base = os.path.join(ROOT, 'experiments', 'SUM_data_process', 'eQTL',
                            'eqtl_MPRA_modeals_eval', 'eQTL_MPRA_models_eval_kita_etal_select', 'results')
    elif exp == "Renganaath_etal":
        base = os.path.join(ROOT, 'revision_experiments', 'eQTL',
                            'eQTL_MPRA_models_eval_Renganaath_etal', 'results')
    if not base:
        return pd.DataFrame()

    model_dir = os.path.join(base, model)
    pos_f = os.path.join(model_dir, 'final_pos_predictions.tsv')
    neg_f = os.path.join(model_dir, f'final_neg_predictions_{neg_set}.tsv')
    if not (os.path.exists(pos_f) and os.path.exists(neg_f)):
        raise FileNotFoundError(f"Missing MPRA: {model}, neg{neg_set}, {exp}")

    pos = pd.read_csv(pos_f, sep='\t', usecols=['Position_Gene', 'logSED'])
    neg = pd.read_csv(neg_f, sep='\t', usecols=['Position_Gene', 'logSED'])
    pos['label'], neg['label'] = 1, 0
    for df in (pos, neg):
        df.rename(columns={'logSED': 'score'}, inplace=True)
        df['score'] = df['score'].abs()
    return pd.concat([pos[['Position_Gene', 'score', 'label']],
                      neg[['Position_Gene', 'score', 'label']]], ignore_index=True)


def load_dream_all(model: str, neg_set: int) -> pd.DataFrame:
    dfs = []
    for exp in EXPS:
        try:
            dfs.append(load_dream_exp(model, exp, neg_set))
        except FileNotFoundError:
            pass
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    return (all_df.groupby('Position_Gene', as_index=False)
            .agg(score=('score', 'mean'), label=('label', 'first')))


def plot_comparison(models: dict, out_dir: str, title_suffix: str):
    """
    Given a dict of {name -> df} with 'score' & 'label', produce:
      - Mann-Whitney U test
      - Boxplot of positive-eQTL quantiles
      - ECDF of positive-eQTL quantiles
    All with zero-score filtering, no balancing.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Filter out zero scores and empty DataFrames
    for name in list(models.keys()):
        df = models[name]
        if df.empty:
            del models[name]
            continue
        models[name] = df[df['score'] > 0].copy()
        if models[name].empty:
            del models[name]

    if not models:
        return

    # Compute quantiles
    for df in models.values():
        df['quantile'] = df.score.rank(method='first', pct=True)

    pos_q = {name: df.loc[df.label == 1, 'quantile']
             for name, df in models.items()}

    # Filter out models with no positives
    pos_q = {k: v for k, v in pos_q.items() if len(v) > 0}
    if not pos_q:
        return

    # Mann-Whitney U
    reference = 'Shorkie_LM' if 'Shorkie_LM' in pos_q else 'Shorkie'
    print(f"\n  MWU: {reference} > others? {title_suffix}")
    if reference in pos_q:
        ref_q = pos_q[reference]
        for name, q in pos_q.items():
            if name == reference:
                continue
            u, p = mannwhitneyu(ref_q, q, alternative='greater')
            print(f"    {reference} vs {name:20s}: U={u:.1f}, p={p:.2e}")

    # Boxplot
    model_order = [n for n in ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init',
                                'DREAM-Atten', 'DREAM-CNN', 'DREAM-RNN']
                   if n in pos_q]
    palette = [MODEL_COLORS.get(n, 'gray') for n in model_order]

    df_box = pd.DataFrame({
        'model': np.repeat(model_order, [len(pos_q[n]) for n in model_order]),
        'quantile': np.concatenate([pos_q[n].values for n in model_order])
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y='quantile', data=df_box, palette=palette, order=model_order)
    plt.title(f'Positive-eQTL Quantile Comparison {title_suffix}')
    plt.ylabel('Quantile')
    plt.xlabel('')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pos_quantile_boxplot.png'), dpi=300)
    plt.close()

    # ECDF
    plt.figure(figsize=(6, 6))
    for name in model_order:
        q = pos_q[name]
        x = np.sort(q)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', label=name, color=MODEL_COLORS.get(name, 'gray'), lw=1.5)
    plt.title(f'ECDF of Positive-eQTL Quantiles {title_suffix}')
    plt.xlabel('Quantile')
    plt.ylabel('ECDF')
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pos_quantile_ecdf.png'), dpi=300)
    plt.close()


def plot_ensemble_comparison(all_neg_models: dict, out_dir: str, title_suffix: str):
    """
    Ensemble comparison across multiple negative sets.
    all_neg_models: {neg_set: {model_name: df}}
    Produces ensemble boxplot with per-set medians overlaid.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect per-set median quantiles
    # {model_name: [median_q_neg1, median_q_neg2, ...]}
    per_set_medians = {}
    # Also collect all positive quantiles for aggregate boxplot
    all_pos_quantiles = {}

    for ns, models in all_neg_models.items():
        for name in list(models.keys()):
            df = models[name]
            if df.empty:
                continue
            df = df[df['score'] > 0].copy()
            if df.empty:
                continue
            df['quantile'] = df.score.rank(method='first', pct=True)
            pos_q = df.loc[df.label == 1, 'quantile']
            if len(pos_q) == 0:
                continue

            per_set_medians.setdefault(name, []).append(pos_q.median())
            all_pos_quantiles.setdefault(name, []).append(pos_q.values)

    if not per_set_medians:
        return

    model_order = [n for n in ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init',
                                'DREAM-Atten', 'DREAM-CNN', 'DREAM-RNN']
                   if n in per_set_medians]
    palette = [MODEL_COLORS.get(n, 'gray') for n in model_order]

    # Mann-Whitney U on concatenated positives
    all_pos_concat = {}
    for name in model_order:
        if name in all_pos_quantiles:
            all_pos_concat[name] = np.concatenate(all_pos_quantiles[name])

    reference = 'Shorkie_LM' if 'Shorkie_LM' in all_pos_concat else 'Shorkie'
    print(f"\n  Ensemble MWU: {reference} > others? {title_suffix}")
    if reference in all_pos_concat:
        ref_q = all_pos_concat[reference]
        for name in model_order:
            if name == reference or name not in all_pos_concat:
                continue
            u, p = mannwhitneyu(ref_q, all_pos_concat[name], alternative='greater')
            print(f"    {reference} vs {name:20s}: U={u:.1f}, p={p:.2e}")

    # Ensemble Boxplot: aggregate all positives across neg sets
    df_box = pd.DataFrame({
        'model': np.repeat(model_order, [len(all_pos_concat.get(n, [])) for n in model_order]),
        'quantile': np.concatenate([all_pos_concat.get(n, np.array([])) for n in model_order])
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='model', y='quantile', data=df_box, palette=palette, order=model_order, ax=ax)

    # Overlay per-set medians as individual points
    for i, name in enumerate(model_order):
        medians = per_set_medians.get(name, [])
        ax.scatter([i] * len(medians), medians, color='black', s=40, zorder=5,
                   marker='D', edgecolors='white', linewidth=0.5)

    ax.set_title(f'Ensemble Positive-eQTL Quantile Comparison {title_suffix}')
    ax.set_ylabel('Quantile')
    ax.set_xlabel('')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ensemble_pos_quantile_boxplot.png'), dpi=300)
    plt.close()

    # Ensemble ECDF: use concatenated data
    plt.figure(figsize=(6, 6))
    for name in model_order:
        if name not in all_pos_concat:
            continue
        q = all_pos_concat[name]
        x = np.sort(q)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', label=name, color=MODEL_COLORS.get(name, 'gray'), lw=1.5)
    plt.title(f'Ensemble ECDF of Positive-eQTL Quantiles {title_suffix}')
    plt.xlabel('Quantile')
    plt.ylabel('ECDF')
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ensemble_pos_quantile_ecdf.png'), dpi=300)
    plt.close()

    # Summary bar chart: mean ± SEM of per-set median quantiles
    fig, ax = plt.subplots(figsize=(10, 5))
    means = [np.mean(per_set_medians.get(n, [0])) for n in model_order]
    sems = [sp_stats.sem(per_set_medians.get(n, [0])) if len(per_set_medians.get(n, [])) >= 2 else 0
            for n in model_order]

    bars = ax.bar(range(len(model_order)), means, yerr=sems,
                  capsize=5, color=palette, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha='right')
    ax.set_ylabel('Mean Median Quantile of Pos. eQTLs (± SEM)')
    ax.set_title(f'Ensemble Median Quantile Comparison {title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ensemble_median_quantile_bar.png'), dpi=300)
    plt.close()


def main():
    for neg in range(1, NEG_NUM + 1):
        # Per-experiment plots (single neg set)
        for exp in EXPS:
            out_exp = os.path.join(ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results',
                                   exp, f'set{neg}')
            models = {}
            for v in ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init']:
                try:
                    models[v] = load_shorkie_data(exp, neg, v)
                except FileNotFoundError:
                    pass
            for m in MPRA_MODELS:
                try:
                    models[MPRA_NAME_MAP[m]] = load_dream_exp(m, exp, neg)
                except FileNotFoundError:
                    pass
            plot_comparison(models, out_exp, f'({exp}, neg{neg})')

    # Ensemble across neg sets — per experiment
    for exp in EXPS:
        out_ens = os.path.join(ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results',
                                exp, 'ensemble')
        all_neg = {}
        for neg in range(1, NEG_NUM + 1):
            models = {}
            for v in ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init']:
                try:
                    models[v] = load_shorkie_data(exp, neg, v)
                except FileNotFoundError:
                    pass
            for m in MPRA_MODELS:
                try:
                    models[MPRA_NAME_MAP[m]] = load_dream_exp(m, exp, neg)
                except FileNotFoundError:
                    pass
            all_neg[neg] = models
        plot_ensemble_comparison(all_neg, out_ens, f'({exp})')

    # Ensemble across ALL exps + ALL neg sets
    out_all = os.path.join(ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results',
                           'merged', 'ensemble')
    all_neg_merged = {}
    for neg in range(1, NEG_NUM + 1):
        models_all = {}
        for v in ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init']:
            models_all[v] = load_shorkie_all(neg, v)
        for m in MPRA_MODELS:
            models_all[MPRA_NAME_MAP[m]] = load_dream_all(m, neg)
        all_neg_merged[neg] = models_all
    plot_ensemble_comparison(all_neg_merged, out_all, '(All Exps)')


if __name__ == '__main__':
    main()
