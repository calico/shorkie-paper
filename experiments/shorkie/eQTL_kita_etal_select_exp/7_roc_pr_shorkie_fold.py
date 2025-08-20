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
NUM_FOLDS    = 8
FOLDS        = [f"f{i}c0" for i in range(NUM_FOLDS)]
ROOT         = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
SEED         = 42

MPRA_MODELS    = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP  = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

random.seed(SEED)
np.random.seed(SEED)
for neg_set in range(1,5):
    print(f"Using MPRA negative set {neg_set}")
    OUT_DIR      = f'results/set{neg_set}/'
    TSV_DIR      = os.path.join(ROOT, 'experiments', 'SUM_data_process', 'eQTL_kita_etal_select_exp', 'results', f'set{neg_set}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # ─── MPRA loader ──────────────────────────────────────────────────────────
    def process_mpra_data(pos_file, neg_file):
        pos_df = pd.read_csv(pos_file, sep="\t")
        neg_df = pd.read_csv(neg_file, sep="\t")
        pos_df['label'] = 1
        neg_df['label'] = 0
        pos_df.rename(columns={'logSED':'score'}, inplace=True)
        neg_df.rename(columns={'logSED':'score'}, inplace=True)
        return pd.concat([pos_df, neg_df], ignore_index=True)

    # ─── Load and balance all MPRA models ────────────────────────────────────
    mpra_balanced = {}
    for mdl in MPRA_MODELS:
        base   = os.path.join(ROOT, 'experiments', 'SUM_data_process', 'eQTL_MPRA_models_eval_kita_etal_select','results', mdl)
        pos_tsv = os.path.join(base, 'final_pos_predictions.tsv')
        neg_tsv = os.path.join(base, 'final_neg_predictions_1.tsv')
        df      = process_mpra_data(pos_tsv, neg_tsv).dropna(subset=['score','label'])
        df['score'] = df['score'].abs()
        pos_df = df[df['label']==1]
        neg_df = df[df['label']==0]
        print(f"[MPRA {mdl}] pos={len(pos_df)}, neg={len(neg_df)}")
        if len(pos_df) > len(neg_df):
            pos_df = pos_df.sample(len(neg_df), random_state=SEED)
        elif len(neg_df) > len(pos_df):
            neg_df = neg_df.sample(len(pos_df), random_state=SEED)
        mpra_balanced[mdl] = pd.concat([pos_df, neg_df], ignore_index=True)
        print(f"[MPRA {mdl}] balanced to {len(pos_df)} pos / {len(neg_df)} neg")

    # ─── Load all yeast per‑fold TSVs ────────────────────────────────────────
    yeast_dfs = {}
    for fold in FOLDS:
        path = os.path.join(TSV_DIR, f'yeast_eqtl_{fold}.tsv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing TSV for fold {fold}: {path}")
        df = pd.read_csv(path, sep='\t', usecols=['Position_Gene','score','label'])
        yeast_dfs[fold] = df.drop_duplicates(['Position_Gene'])

    # ─── Common positives & negatives across folds ──────────────────────────
    pos_sets    = [set(df[df['label']==1]['Position_Gene']) for df in yeast_dfs.values()]
    neg_sets    = [set(df[df['label']==0]['Position_Gene']) for df in yeast_dfs.values()]
    common_pos  = set.intersection(*pos_sets)
    common_neg  = set.intersection(*neg_sets)
    print(f"[Yeast] common pos={len(common_pos)}, common neg={len(common_neg)}")

    # build unified yeast DataFrame
    keys         = sorted(common_pos) + sorted(common_neg)
    yeast_scores = pd.DataFrame({'Position_Gene': keys})
    score_cols   = []
    # generate a sequence of lighter blues for folds
    fold_colors  = plt.cm.Greys(np.linspace(0.1, 0.6, NUM_FOLDS))

    for idx, (fold, df) in enumerate(yeast_dfs.items()):
        col = f'score_{fold}'
        score_cols.append(col)
        m = df.set_index('Position_Gene')['score']
        yeast_scores[col] = yeast_scores['Position_Gene'].map(m).abs()

    yeast_scores['label']    = yeast_scores['Position_Gene'].isin(common_pos).astype(int)
    yeast_scores['ensemble'] = yeast_scores[score_cols].mean(axis=1)

    # ─── Plot ROC ───────────────────────────────────────────────────────────
    plt.figure(figsize=(6,6))
    # diagonal
    plt.plot([0,1], [0,1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # per‑fold with lighter blues
    for idx, fold in enumerate(FOLDS):
        col   = score_cols[idx]
        color = fold_colors[idx]
        fpr, tpr, _ = roc_curve(yeast_scores['label'], yeast_scores[col])
        auc_        = auc(fpr, tpr)
        # plt.plot(fpr, tpr,
        #          color=color,
        #          linewidth=1.5,
        #          label=f"Shorkie {fold} (AUC={auc_:.2f})")

    # Shorkie ensemble in solid black
    fpr_e, tpr_e, _ = roc_curve(yeast_scores['label'], yeast_scores['ensemble'])
    auc_e           = auc(fpr_e, tpr_e)
    plt.plot(fpr_e, tpr_e,
            color='black', linestyle='-', linewidth=2,
            label=f"Shorkie ensemble (AUC={auc_e:.2f})")

    # MPRA models (all same style)
    for mdl, df in mpra_balanced.items():
        fpr_m, tpr_m, _ = roc_curve(df['label'], df['score'])
        auc_m           = auc(fpr_m, tpr_m)
        plt.plot(fpr_m, tpr_m,
                linestyle='-.', linewidth=2,
                label=f"{MPRA_NAME_MAP[mdl]} (AUC={auc_m:.2f})")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves: Shorkie vs DREAM models', fontsize=16)
    plt.legend(loc='best', fontsize=14)  # Increase font size of legend
    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Adjust y-axis limit
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'roc_eqtl_mpra_all.png'), dpi=300)
    plt.close()

    # ─── Plot Precision–Recall ──────────────────────────────────────────────
    plt.figure(figsize=(6,6))
    # diagonal
    plt.plot([1,0], [0,1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # per‑fold
    for idx, fold in enumerate(FOLDS):
        col   = score_cols[idx]
        color = fold_colors[idx]
        prec, rec, _ = precision_recall_curve(yeast_scores['label'], yeast_scores[col])
        ap_         = average_precision_score(yeast_scores['label'], yeast_scores[col])
        # plt.plot(rec, prec,
        #          color=color,
        #          linewidth=1.5,
        #          label=f"Shorkie {fold} (AUPRC={ap_:.2f})")

    # Shorkie ensemble in black
    prec_e, rec_e, _ = precision_recall_curve(yeast_scores['label'], yeast_scores['ensemble'])
    ap_e             = average_precision_score(yeast_scores['label'], yeast_scores['ensemble'])
    plt.plot(rec_e, prec_e,
            color='black', linestyle='-', linewidth=2,
            label=f"Shorkie ensemble (AUPRC={ap_e:.2f})")

    # MPRA models
    for mdl, df in mpra_balanced.items():
        prec_m, rec_m, _ = precision_recall_curve(df['label'], df['score'])
        # area under the curve
        auc_pr = auc(rec_m, prec_m)
        plt.plot(rec_m, prec_m,
                linestyle='-.', linewidth=2,
                label=f"{MPRA_NAME_MAP[mdl]} (AUPRC={auc_pr:.2f})")

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    # plt.ylim(0.4, 1)  # Adjust y-axis limit
    plt.ylim(0, 1)  # Adjust y-axis limit
    plt.xlim(0, 1)
    plt.title('PR curves: Shorkie vs DREAM models', fontsize=16)
    plt.legend(loc='lower left', fontsize=14)  # Increase font size of legend
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pr_eqtl_mpra_all.png'), dpi=300)
    plt.close()

    print("Done. Plots written to:")
    print(f"  {os.path.join(OUT_DIR,'roc_eqtl_mpra_all.png')}")
    print(f"  {os.path.join(OUT_DIR,'pr_eqtl_mpra_all.png')}")