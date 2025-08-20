#!/usr/bin/env python3
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ─── Config ────────────────────────────────────────────────────────────────
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT       = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
NEG_NUM    = 4
EXPS       = ["Caudal_etal", "Kita_etal"]
MPRA_MODELS   = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

# ─── Helpers ──────────────────────────────────────────────────────────────
def load_shorkie_exp(exp: str, neg_set: int) -> pd.DataFrame:
    """Load Shorkie logSED_agg for a single experiment."""
    fn = f"{exp.lower()}_scores.tsv"
    path = os.path.join(
        ROOT, 'experiments','SUM_data_process','eQTL_all','results',
        f'negset_{neg_set}', fn
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Shorkie file: {path}")
    df = pd.read_csv(path, sep='\t', usecols=['Position_Gene','logSED_agg','label'])
    df = df.drop_duplicates('Position_Gene')
    df['score'] = df['logSED_agg'].abs()
    return df[['Position_Gene','score','label']]


def load_shorkie_all(neg_set: int) -> pd.DataFrame:
    """Merge Shorkie across both experiments by averaging absolute logSED_agg."""
    dfs = [load_shorkie_exp(exp, neg_set) for exp in EXPS]
    all_df = pd.concat(dfs, ignore_index=True)
    return (
        all_df
        .groupby('Position_Gene', as_index=False)
        .agg(score=('score','mean'), label=('label','first'))
    )


def load_dream_exp(model: str, exp: str, neg_set: int) -> pd.DataFrame:
    """Load MPRA predictions for one experiment and one neg_set."""
    if exp == "Caudal_etal":
        base = os.path.join(
            ROOT, 'experiments','SUM_data_process','eQTL_MPRA_models_eval','results'
        )
    else:
        base = os.path.join(
            ROOT, 'experiments','SUM_data_process','eQTL_MPRA_models_eval_kita_etal_select','results'
        )
    model_dir = os.path.join(base, model)
    pos_f = os.path.join(model_dir, 'final_pos_predictions.tsv')
    neg_f = os.path.join(model_dir, f'final_neg_predictions_{neg_set}.tsv')
    if not (os.path.exists(pos_f) and os.path.exists(neg_f)):
        raise FileNotFoundError(f"Missing MPRA files for {model}, neg{neg_set}, exp {exp}")
    pos = pd.read_csv(pos_f, sep='\t', usecols=['Position_Gene','logSED'])
    neg = pd.read_csv(neg_f, sep='\t', usecols=['Position_Gene','logSED'])
    pos['label'], neg['label'] = 1, 0
    for df in (pos, neg):
        df.rename(columns={'logSED':'score'}, inplace=True)
        df['score'] = df['score'].abs()
    return pd.concat([pos[['Position_Gene','score','label']], neg[['Position_Gene','score','label']]], ignore_index=True)


def load_dream_all(model: str, neg_set: int) -> pd.DataFrame:
    """Merge MPRA predictions across both experiments."""
    dfs = [load_dream_exp(model, exp, neg_set) for exp in EXPS]
    return pd.concat(dfs, ignore_index=True)


def plot_comparison(models: dict, out_dir: str, title_suffix: str):
    """Given a dict of name->df with 'score' & 'label', do MWU, boxplot, ECDF ignoring zeros."""
    os.makedirs(out_dir, exist_ok=True)

    # 1) filter out zero scores entirely
    for name in list(models.keys()):
        df = models[name]
        models[name] = df[df['score'] > 0].copy()

    # 2) quantiles (break ties consistently)
    for df in models.values():
        df['quantile'] = df.score.rank(method='first', pct=True)

    pos_q = {name: df.loc[df.label == 1, 'quantile']
             for name, df in models.items()}

    # 3) Mann–Whitney U
    print(f"\nMann–Whitney U: Shorkie > DREAM positives? {title_suffix}\n")
    sh_q = pos_q['Shorkie']
    for name, q in pos_q.items():
        if name == 'Shorkie': continue
        u, p = mannwhitneyu(sh_q, q, alternative='greater')
        print(f"  Shorkie vs {name:10s}: U={u:.1f}, p={p:.2e}")

    # 4) Boxplot
    df_box = pd.DataFrame({
        'model': np.repeat(list(pos_q.keys()), [len(q) for q in pos_q.values()]),
        'quantile': np.concatenate(list(pos_q.values()))
    })
    plt.figure(figsize=(5,5))
    sns.boxplot(x='model', y='quantile', data=df_box, palette='Set2')
    plt.title(f'Positive‑eQTL Quantile Comparison {title_suffix}')
    plt.ylabel('Quantile'); plt.xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pos_quantile_boxplot.png'), dpi=300)
    plt.close()

    # 5) ECDF
    plt.figure(figsize=(5,5))
    for name, q in pos_q.items():
        x = np.sort(q)
        y = np.arange(1, len(x)+1) / len(x)
        plt.step(x, y, where='post', label=name)
    plt.title(f'ECDF of Positive‑eQTL Quantiles {title_suffix}')
    plt.xlabel('Quantile'); plt.ylabel('ECDF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pos_quantile_ecdf.png'), dpi=300)
    plt.close()


def main():
    for neg in range(1, NEG_NUM+1):
        # per-experiment plots
        for exp in EXPS:
            out_exp = os.path.join('results', exp, f'set{neg}')
            sh = load_shorkie_exp(exp, neg)
            models = {'Shorkie': sh}
            for m in MPRA_MODELS:
                models[MPRA_NAME_MAP[m]] = load_dream_exp(m, exp, neg)
            plot_comparison(models, out_exp, f'({exp})')

        # merged across EXPS
        out_all = os.path.join('results', 'merged', f'set{neg}')
        sh_all = load_shorkie_all(neg)
        models_all = {'Shorkie': sh_all}
        for m in MPRA_MODELS:
            models_all[MPRA_NAME_MAP[m]] = load_dream_all(m, neg)
        plot_comparison(models_all, out_all, f'(Caudal et al. + Kita et al.)')

if __name__ == '__main__':
    main()
