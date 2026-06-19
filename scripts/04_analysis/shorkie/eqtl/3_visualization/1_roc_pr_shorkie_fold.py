#!/usr/bin/env python3
"""
ROC / PR curve generation for Shorkie + DREAM models.

Changes from previous version:
  - No class balancing (evaluate on natural distribution)
  - "All Sets" uses proper ensemble: per-set metrics → mean ± SEM,
    with interpolated mean curve and shaded ±1 SEM band
"""
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

# ─── Configuration ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_FOLDS = 8
FOLDS = [f"f{i}c0" for i in range(NUM_FOLDS)]

# MPRA models to compare
MPRA_MODELS = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

ROOT = args.root_dir
EXPS = ["caudal_etal", "kita_etal", "Renganaath_etal"]

# Common interpolation grids for ensemble curves
COMMON_FPR = np.linspace(0, 1, 200)
COMMON_REC = np.linspace(0, 1, 200)

# Style mapping
STYLE_MAP = {}  # filled dynamically

SHORKIE_COLORS = {
    'Shorkie':              '#2196F3',
    'Shorkie_LM':           '#E91E63',
    'Shorkie_Random_Init':  '#FF9800',
}
DREAM_COLORS = {
    'DREAM-Atten': '#4CAF50',
    'DREAM-CNN':   '#9C27B0',
    'DREAM-RNN':   '#795548',
}


def get_color(col):
    return {**SHORKIE_COLORS, **DREAM_COLORS}.get(col, 'gray')


def get_style(col):
    return ('-', 2.0) if 'Shorkie' in col else ('-.', 1.5)


def process_shorkie(exp: str, neg_set: int) -> pd.DataFrame:
    """Load Shorkie scores (Base, LM, Random_Init) for one dataset + neg_set."""
    base_dir = os.path.join(
        ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results',
        f'negset_{neg_set}'
    )

    # 1. Base Shorkie
    orig_path = os.path.join(base_dir, f"{exp}_Shorkie_scores.tsv")
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f"Missing: {orig_path}")
    df_orig = pd.read_csv(orig_path, sep='\t', usecols=['Position_Gene', 'logSED_agg', 'label'])
    df_orig['Position_Gene'] = df_orig['Position_Gene'].astype(str).str.strip()
    df_orig = df_orig.drop_duplicates(['Position_Gene'])
    df_orig['Shorkie'] = df_orig['logSED_agg'].abs()
    df_orig = df_orig.drop(columns=['logSED_agg'])

    # 2. LM Shorkie
    lm_path = os.path.join(base_dir, f"{exp}_Shorkie_LM_scores.tsv")
    if not os.path.exists(lm_path):
        raise FileNotFoundError(f"Missing: {lm_path}")
    df_lm = pd.read_csv(lm_path, sep='\t', usecols=['Position_Gene', 'LLR', 'label'])
    df_lm['Position_Gene'] = df_lm['Position_Gene'].astype(str).str.strip()
    df_lm = df_lm.drop_duplicates(['Position_Gene'])
    df_lm['Shorkie_LM'] = df_lm['LLR'].abs()
    df_lm = df_lm.drop(columns=['LLR', 'label'])

    # 3. Random_Init Shorkie
    ri_path = os.path.join(base_dir, f"{exp}_Shorkie_Random_Init_scores.tsv")
    if not os.path.exists(ri_path):
        raise FileNotFoundError(f"Missing: {ri_path}")
    df_ri = pd.read_csv(ri_path, sep='\t', usecols=['Position_Gene', 'logSED_agg'])
    df_ri['Position_Gene'] = df_ri['Position_Gene'].astype(str).str.strip()
    df_ri = df_ri.drop_duplicates(['Position_Gene'])
    df_ri['Shorkie_Random_Init'] = df_ri['logSED_agg'].abs()
    df_ri = df_ri.drop(columns=['logSED_agg'])

    # Merge on Position_Gene
    merged = pd.merge(df_orig, df_lm, on='Position_Gene', how='inner')
    merged = pd.merge(merged, df_ri, on='Position_Gene', how='inner')
    return merged


def process_mpra(model: str, neg_set: int, mpra_eval_base: str) -> pd.DataFrame:
    """Load MPRA predictions and match by Position_Gene."""
    model_dir = os.path.join(mpra_eval_base, model)
    pos_f = os.path.join(model_dir, 'final_pos_predictions.tsv')
    neg_f = os.path.join(model_dir, f'final_neg_predictions_{neg_set}.tsv')
    if not (os.path.exists(pos_f) and os.path.exists(neg_f)):
        raise FileNotFoundError(f"Missing MPRA files for {model} neg{neg_set}")

    pos = pd.read_csv(pos_f, sep='\t', usecols=['Position_Gene', 'logSED'])
    neg = pd.read_csv(neg_f, sep='\t', usecols=['Position_Gene', 'logSED'])
    pos['label'], neg['label'] = 1, 0
    for df in (pos, neg):
        df['Position_Gene'] = df['Position_Gene'].astype(str).str.strip()
        df.rename(columns={'logSED': 'score'}, inplace=True)
        df['score'] = df['score'].abs()
    return pd.concat([pos, neg], ignore_index=True)[['Position_Gene', 'score', 'label']]


def get_mpra_base(exp):
    if exp == "caudal_etal":
        return os.path.join(
            ROOT, 'experiments', 'SUM_data_process', 'eQTL', 'eqtl_MPRA_modeals_eval',
            'eQTL_MPRA_models_eval_caudal_etal', 'results'
        )
    elif exp == "kita_etal":
        return os.path.join(
            ROOT, 'experiments', 'SUM_data_process', 'eQTL', 'eqtl_MPRA_modeals_eval',
            'eQTL_MPRA_models_eval_kita_etal_select', 'results'
        )
    elif exp == "Renganaath_etal":
        return os.path.join(
            ROOT, 'revision_experiments', 'eQTL', 'eQTL_MPRA_models_eval_Renganaath_etal', 'results'
        )
    return None


def load_combined(exp, neg_set):
    """Load and merge all model scores for one experiment + one negative set."""
    combined_df = process_shorkie(exp, neg_set)
    mpra_base = get_mpra_base(exp)
    if mpra_base:
        for model in MPRA_MODELS:
            mpra_df = process_mpra(model, neg_set, mpra_base)
            mpra_df = mpra_df.rename(columns={'score': MPRA_NAME_MAP[model]})
            combined_df = combined_df.merge(
                mpra_df[['Position_Gene', MPRA_NAME_MAP[model]]],
                on='Position_Gene', how='inner'
            )
    return combined_df, mpra_base is not None


def plot_single_set_roc_pr(combined_df, score_cols, out_dir, exp, neg_set):
    """Plot ROC and PR for a single negative set (no balancing)."""
    combined_df = combined_df.dropna(subset=score_cols)

    # ROC
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], '--', alpha=0.5, color='gray')
    for col in score_cols:
        valid = combined_df[combined_df[col] > 0]
        if len(valid) == 0 or len(valid['label'].unique()) < 2:
            continue
        fpr, tpr, _ = roc_curve(valid['label'], valid[col])
        ls, lw = get_style(col)
        plt.plot(fpr, tpr, linestyle=ls, lw=lw, color=get_color(col),
                 label=f"{col} (AUC={auc(fpr, tpr):.3f})")
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f"ROC ({exp}, neg{neg_set})")
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'roc_set{neg_set}.png'), dpi=300)
    plt.close()

    # PR
    plt.figure(figsize=(5, 5))
    for col in score_cols:
        valid = combined_df[combined_df[col] > 0]
        if len(valid) == 0 or len(valid['label'].unique()) < 2:
            continue
        prec, rec, _ = precision_recall_curve(valid['label'], valid[col])
        ls, lw = get_style(col)
        plt.plot(rec, prec, linestyle=ls, lw=lw, color=get_color(col),
                 label=f"{col} (AUPRC={average_precision_score(valid['label'], valid[col]):.3f})")
    plt.ylim(0.45, 1.05)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f"PR ({exp}, neg{neg_set})")
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pr_set{neg_set}.png'), dpi=300)
    plt.close()


def plot_ensemble_roc_pr(exp, score_cols, output_base):
    """
    Ensemble approach: compute ROC/PR per negative set, then plot mean ± SEM.
    """
    # Collect per-set interpolated curves and scalar AUCs
    roc_data = {col: {'tprs': [], 'aucs': []} for col in score_cols}
    pr_data  = {col: {'precs': [], 'auprcs': []} for col in score_cols}

    for ns in range(1, 5):
        try:
            combined_df, _ = load_combined(exp, ns)
        except FileNotFoundError:
            continue
        combined_df = combined_df.dropna(subset=score_cols)

        for col in score_cols:
            valid = combined_df[combined_df[col] > 0]
            if len(valid) < 10 or len(valid['label'].unique()) < 2:
                continue

            # ROC
            fpr, tpr, _ = roc_curve(valid['label'], valid[col])
            auc_val = auc(fpr, tpr)
            tpr_interp = np.interp(COMMON_FPR, fpr, tpr)
            tpr_interp[0] = 0.0
            roc_data[col]['tprs'].append(tpr_interp)
            roc_data[col]['aucs'].append(auc_val)

            # PR
            prec, rec, _ = precision_recall_curve(valid['label'], valid[col])
            # precision_recall_curve returns in decreasing recall order;
            # flip for interp (need increasing x)
            rec_sorted = rec[::-1]
            prec_sorted = prec[::-1]
            prec_interp = np.interp(COMMON_REC, rec_sorted, prec_sorted)
            pr_data[col]['precs'].append(prec_interp)
            pr_data[col]['auprcs'].append(
                average_precision_score(valid['label'], valid[col])
            )

    # ─── ROC Ensemble Plot ───
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], '--', alpha=0.5, color='gray')

    for col in score_cols:
        tprs = roc_data[col]['tprs']
        aucs_list = roc_data[col]['aucs']
        if len(tprs) < 2:
            continue
        mean_tpr = np.mean(tprs, axis=0)
        sem_tpr = sp_stats.sem(tprs, axis=0)
        mean_auc_val = np.mean(aucs_list)
        sem_auc_val = sp_stats.sem(aucs_list)

        ls, lw = get_style(col)
        color = get_color(col)
        plt.plot(COMMON_FPR, mean_tpr, linestyle=ls, lw=lw, color=color,
                 label=f"{col} ({mean_auc_val:.3f}±{sem_auc_val:.3f})")
        plt.fill_between(COMMON_FPR,
                         np.clip(mean_tpr - sem_tpr, 0, 1),
                         np.clip(mean_tpr + sem_tpr, 0, 1),
                         alpha=0.15, color=color)

    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f"ROC Ensemble ({exp}, 4 neg sets)", fontsize=16)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f'roc_ensemble_all_sets.png'), dpi=300)
    plt.close()

    # ─── PR Ensemble Plot ───
    plt.figure(figsize=(5.5, 5))

    for col in score_cols:
        precs = pr_data[col]['precs']
        auprcs_list = pr_data[col]['auprcs']
        if len(precs) < 2:
            continue
        mean_prec = np.mean(precs, axis=0)
        sem_prec = sp_stats.sem(precs, axis=0)
        mean_auprc = np.mean(auprcs_list)
        sem_auprc = sp_stats.sem(auprcs_list)

        ls, lw = get_style(col)
        color = get_color(col)
        plt.plot(COMMON_REC, mean_prec, linestyle=ls, lw=lw, color=color,
                 label=f"{col} ({mean_auprc:.3f}±{sem_auprc:.3f})")
        plt.fill_between(COMMON_REC,
                         np.clip(mean_prec - sem_prec, 0, 1),
                         np.clip(mean_prec + sem_prec, 0, 1),
                         alpha=0.15, color=color)

    plt.ylim(0.45, 1.05)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f"PR Ensemble ({exp}, 4 neg sets)", fontsize=16)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f'pr_ensemble_all_sets.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    for exp in EXPS:
        print(f"\n=== {exp} ===")
        OUTPUT_BASE = os.path.join(
            ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results', exp, 'combined_plots'
        )
        os.makedirs(OUTPUT_BASE, exist_ok=True)

        mpra_base = get_mpra_base(exp)
        shorkie_cols = ['Shorkie', 'Shorkie_LM', 'Shorkie_Random_Init']
        mpra_cols = [MPRA_NAME_MAP[m] for m in MPRA_MODELS] if mpra_base else []
        score_cols = shorkie_cols + mpra_cols

        # Per-set plots
        for neg_set in range(1, 5):
            print(f"  neg{neg_set}...")
            out_dir = os.path.join(OUTPUT_BASE, f'set{neg_set}')
            os.makedirs(out_dir, exist_ok=True)
            try:
                combined_df, _ = load_combined(exp, neg_set)
                plot_single_set_roc_pr(combined_df, score_cols, out_dir, exp, neg_set)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")

        # Ensemble plot (mean ± SEM across 4 neg sets)
        print(f"  Ensemble...")
        plot_ensemble_roc_pr(exp, score_cols, OUTPUT_BASE)

    print("\nDone.")
