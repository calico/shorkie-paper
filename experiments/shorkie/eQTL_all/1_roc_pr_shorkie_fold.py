#!/usr/bin/env python3
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

ROOT = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
EXPS = ["Caudal_etal", "Kita_etal"]

# metrics to iterate over
METRICS = {
    'logSED_agg': 'agg',
    'logSED_avg': 'avg'
}


def balance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Down-sample the larger class so that positives == negatives."""
    pos = df[df['label'] == 1].copy()
    neg = df[df['label'] == 0].copy()
    return pd.concat([pos, neg], ignore_index=True)


def process_shorkie(exp: str, neg_set: int) -> pd.DataFrame:
    """
    Load Shorkie scores (agg + avg) for the given dataset and neg_set.
    """
    fname = f"{exp.lower()}_scores.tsv"
    path = os.path.join(
        ROOT, 'experiments', 'SUM_data_process', 'eQTL_all', 'results',
        f'negset_{neg_set}', fname
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Shorkie file for {exp} neg{neg_set}: {path}")

    df = pd.read_csv(path, sep='\t',
                     usecols=['Position_Gene', 'logSED_agg', 'logSED_avg', 'label'])
    df = df.drop_duplicates(['Position_Gene'])
    # absolute value
    df['logSED_agg'] = df['logSED_agg'].abs()
    df['logSED_avg'] = df['logSED_avg'].abs()
    df['label']       = df['label'].astype(int)
    return df[['Position_Gene', 'logSED_agg', 'logSED_avg', 'label']]


def process_mpra(model: str, neg_set: int, mpra_eval_base: str) -> pd.DataFrame:
    """Load MPRA predictions and match by Position_Gene."""
    model_dir = os.path.join(mpra_eval_base, model)
    pos_f = os.path.join(model_dir, 'final_pos_predictions.tsv')
    neg_f = os.path.join(model_dir, f'final_neg_predictions_{neg_set}.tsv')
    if not (os.path.exists(pos_f) and os.path.exists(neg_f)):
        raise FileNotFoundError(f"Missing MPRA files for {model} neg{neg_set}")

    pos = pd.read_csv(pos_f, sep='\t', usecols=['Position_Gene','logSED'])
    neg = pd.read_csv(neg_f, sep='\t', usecols=['Position_Gene','logSED'])
    pos['label'], neg['label'] = 1, 0

    for df in (pos, neg):
        df.rename(columns={'logSED':'score'}, inplace=True)
        df['score'] = df['score'].abs()

    return pd.concat([pos, neg], ignore_index=True)[['Position_Gene','score','label']]


if __name__ == '__main__':
    for exp in EXPS:
        # select MPRA results path
        if exp == "Caudal_etal":
            MPRA_EVAL_BASE = os.path.join(
                ROOT, 'experiments', 'SUM_data_process', 'eQTL_MPRA_models_eval', 'results'
            )
        else:
            MPRA_EVAL_BASE = os.path.join(
                ROOT, 'experiments', 'SUM_data_process',
                'eQTL_MPRA_models_eval_kita_etal_select', 'results'
            )

        for metric_col, metric_key in METRICS.items():
            print(f"\n=== Plotting {metric_col} ({metric_key}) for {exp} ===")
            OUTPUT_BASE = os.path.join('results', exp, 'merged_and_balanced', metric_key)
            os.makedirs(OUTPUT_BASE, exist_ok=True)

            # ─── per-negative-set ────────────────────────────────────────────────
            for neg_set in range(1, 5):
                print(f"--- {exp} NEG{neg_set} ({metric_key}) ---")
                out_dir = os.path.join(OUTPUT_BASE, f'set{neg_set}')
                os.makedirs(out_dir, exist_ok=True)

                eqtl_df = process_shorkie(exp, neg_set)
                # ** drop any rows where our metric is NaN **
                eqtl_df = eqtl_df.dropna(subset=[metric_col])

                # count
                n_pos = eqtl_df['label'].sum()
                n_neg = len(eqtl_df) - n_pos
                print(f" Shorkie neg{neg_set}: {len(eqtl_df)} samples ({n_pos}/{n_neg})")

                # ─── ROC ───────────────────────────────────────────────────────
                plt.figure(figsize=(5,5))
                plt.plot([0,1],[0,1],'--',alpha=0.7, color='gray')
                plotted = False

                for model in MPRA_MODELS:
                    mpra_df = process_mpra(model, neg_set, MPRA_EVAL_BASE)
                    common = (
                        eqtl_df[['Position_Gene', metric_col, 'label']]
                        .merge(mpra_df[['Position_Gene','score']], on='Position_Gene')
                    )
                    # also drop any leftover NaNs (just in case)
                    common = common.dropna(subset=[metric_col, 'score'])
                    common = common.rename(columns={metric_col:'ensemble'})
                    balanced = balance_df(common)

                    if not plotted:
                        fpr, tpr, _ = roc_curve(balanced['label'], balanced['ensemble'])
                        plt.plot(fpr, tpr, lw=2,# color='black',
                                 label=f"Shorkie (AUC={auc(fpr,tpr):.2f})", zorder=10)
                        plotted = True

                    fpr_m, tpr_m, _ = roc_curve(balanced['label'], balanced['score'])
                    plt.plot(fpr_m, tpr_m, lw=2, linestyle='-.',
                             label=f"{MPRA_NAME_MAP[model]} (AUC={auc(fpr_m,tpr_m):.2f})")

                plt.xlabel('FPR'); plt.ylabel('TPR')
                plt.title(f"ROC ({exp}, neg{neg_set}, {metric_key})")
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'roc_{metric_key}_set{neg_set}.png'), dpi=300)
                plt.close()

                # ─── PR ────────────────────────────────────────────────────────
                plt.figure(figsize=(5,5))
                plotted = False

                for model in MPRA_MODELS:
                    mpra_df = process_mpra(model, neg_set, MPRA_EVAL_BASE)
                    common = (
                        eqtl_df[['Position_Gene', metric_col, 'label']]
                        .merge(mpra_df[['Position_Gene','score']], on='Position_Gene')
                    ).dropna(subset=[metric_col, 'score'])
                    common = common.rename(columns={metric_col:'ensemble'})
                    balanced = balance_df(common)

                    if not plotted:
                        prec, rec, _ = precision_recall_curve(balanced['label'], balanced['ensemble'])
                        plt.plot(rec, prec, lw=2, #color='black',
                                 label=f"Shorkie (AUPRC={average_precision_score(balanced['label'],balanced['ensemble']):.2f})", zorder=10)
                        plotted = True

                    prec_m, rec_m, _ = precision_recall_curve(balanced['label'], balanced['score'])
                    plt.plot(rec_m, prec_m, lw=2, linestyle='-.',
                             label=f"{MPRA_NAME_MAP[model]} (AUPRC={average_precision_score(balanced['label'],balanced['score']):.2f})")

                plt.ylim(0.4,1.05)
                plt.xlabel('Recall'); plt.ylabel('Precision')
                plt.title(f"PR ({exp}, neg{neg_set}, {metric_key})")
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'pr_{metric_key}_set{neg_set}.png'), dpi=300)
                plt.close()

                print(f" Saved plots for neg{neg_set}, {metric_key}")

            # ─── combined across all negsets ────────────────────────────────────
            # gather & drop NaNs
            sh_all = pd.concat([process_shorkie(exp, ns) for ns in range(1,5)], ignore_index=True)
            sh_all = sh_all.dropna(subset=[metric_col]).rename(columns={metric_col:'score'})
            mpra_all = {
                MPRA_NAME_MAP[m]: pd.concat(
                    [process_mpra(m, ns, MPRA_EVAL_BASE) for ns in range(1,5)],
                    ignore_index=True
                )
                for m in MPRA_MODELS
            }

            bal_s = balance_df(sh_all)
            n_pos, n_neg = int(bal_s['label'].sum()), len(bal_s)-int(bal_s['label'].sum())

            # Combined ROC
            plt.figure(figsize=(5,5))
            plt.plot([0,1],[0,1],'--',alpha=0.7, color='gray')
            fpr_s, tpr_s, _ = roc_curve(bal_s['label'], bal_s['score'])
            plt.plot(fpr_s, tpr_s, lw=2, #color='black',
                     label=f"Shorkie (AUC={auc(fpr_s,tpr_s):.2f})", zorder=10)
            for name, df in mpra_all.items():
                b = balance_df(df)
                fpr_m, tpr_m, _ = roc_curve(b['label'], b['score'])
                plt.plot(fpr_m, tpr_m, lw=2, linestyle='-.',
                         label=f"{name} (AUC={auc(fpr_m,tpr_m):.2f})")

            plt.xlabel('FPR'); plt.ylabel('TPR')
            plt.title(f"Combined ROC ({exp}, {metric_key})")
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_BASE, f'roc_{metric_key}_all_sets.png'), dpi=300)
            plt.close()

            # Combined PR
            plt.figure(figsize=(5,5))
            prec_s, rec_s, _ = precision_recall_curve(bal_s['label'], bal_s['score'])
            plt.plot(rec_s, prec_s, lw=2, #color='black',
                     label=f"Shorkie (AUPRC={average_precision_score(bal_s['label'],bal_s['score']):.2f})", zorder=10)
            for name, df in mpra_all.items():
                b = balance_df(df)
                prec_m, rec_m, _ = precision_recall_curve(b['label'], b['score'])
                plt.plot(rec_m, prec_m, lw=2, linestyle='-.',
                         label=f"{name} (AUPRC={average_precision_score(b['label'],b['score']):.2f})")

            plt.ylim(0.4,1.05)
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title(f"Combined PR ({exp}, {metric_key})")
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_BASE, f'pr_{metric_key}_all_sets.png'), dpi=300)
            plt.close()

            print(f" Saved combined plots for {exp}, {metric_key}")
