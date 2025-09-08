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
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_FOLDS  = 8
ROOT       = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'

# New: path to your intersected CIS file
INTERSECT_CIS = os.path.join(
    ROOT, 'data','eQTL_kita_etal','fix',
    'selected_eQTL','intersected_CIS.tsv'
)
# mapping chromosomes to roman numerals I–XVI
chr_map = {
    f"chromosome{i}": f"chr{rn}"
    for i, rn in zip(
        range(1,17),
        ["I","II","III","IV","V","VI","VII","VIII",
         "IX","X","XI","XII","XIII","XIV","XV","XVI"]
    )
}

# Bases for MPRA-RNN (unchanged)
MPRA_BASE  = os.path.join(
    ROOT, 'experiments','SUM_data_process',
    'eQTL_MPRA_models_eval_kita_etal_select','results','DREAM_RNN'
)

OUTPUT_DIR = 'results/roc_pr_by_model_and_locType'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# plotting styles
GROUP_COLORS = {
    'Promoter': 'tab:red',
    'UTR3':     'tab:green',
    'UTR5':     'tab:grey',
    'ORF':      'tab:blue',
    'All':      'black',
    'Downstream': 'tab:orange',
    'Upstream': 'tab:purple',
    'DownstreamClose': 'tab:cyan',
}
MODEL_STYLES = {
    'Shorkie':   '-',
    'MPRA-RNN':  '-.',
}

# ─── load intersected CIS once ────────────────────────────────────────────
cis_df = pd.read_csv(
    INTERSECT_CIS, sep='\t',
    usecols=['Chr','ChrPos','locationType']
).drop_duplicates(['Chr','ChrPos'])
cis_df['ChrPos'] = cis_df['ChrPos'].astype(int)
# remap Chr values (e.g. "chromosome1" -> "chrI")
cis_df['Chr'] = cis_df['Chr'].map(chr_map)
print(f"First 5 rows of intersected CIS data:\n{cis_df.head()}")


def balance_df(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    if len(pos) > len(neg):
        pos = pos.sample(len(neg), random_state=SEED)
    else:
        neg = neg.sample(len(pos), random_state=SEED)
    return pd.concat([pos, neg], ignore_index=True)


def process_shorkie(neg_set: int) -> pd.DataFrame:
    """
    Reads the single TSV for Shorkie (negset_{neg_set}), merges in
    locationType from cis_df, and balances pos/neg.
    """
    tsv = os.path.join(
        ROOT, 'experiments','SUM_data_process',
        'eQTL_all','results',
        f'negset_{neg_set}','kita_etal_scores.tsv'
    )
    if not os.path.exists(tsv):
        raise FileNotFoundError(f"No file at {tsv}")

    df = pd.read_csv(
        tsv, sep='\t',
        usecols=[
            'Position_Gene', 'logSED_agg', 'label', 'Chr', 'ChrPos'
        ]
    )
    df = df.rename(columns={'logSED_agg':'score'})
    df['score'] = df['score'].abs()
    df['label'] = df['label'].astype(int)
    df['ChrPos'] = df['ChrPos'].astype(int)

    # First few rows for debugging
    print(f"First 5 rows of Shorkie data:\n{df.head()}")
    print(f"Shape of Shorkie data: {df.shape}")
    # merge in locationType
    df = df.merge(cis_df, on=['Chr','ChrPos'], how='left')
    if df['locationType'].isna().any():
        missing = df['locationType'].isna().sum()
        print(f"  → Warning: {missing} Shorkie rows have no locationType after merge")

    return balance_df(df)


def process_mpra_rnn(neg_set: int) -> pd.DataFrame:
    pos_f = os.path.join(MPRA_BASE, 'final_pos_predictions.tsv')
    neg_f = os.path.join(MPRA_BASE, f'final_neg_predictions_{neg_set}.tsv')
    if not (os.path.exists(pos_f) and os.path.exists(neg_f)):
        raise FileNotFoundError(f"Missing MPRA-RNN files for neg_set={neg_set}")

    pos = pd.read_csv(pos_f, sep='\t')
    neg = pd.read_csv(neg_f, sep='\t')
    pos['label'], neg['label'] = 1, 0

    for df in (pos, neg):
        df.rename(columns={'logSED':'score'}, inplace=True)
        df['score'] = df['score'].abs()

    mpra = pd.concat([pos, neg], ignore_index=True).dropna(subset=['score'])
    mpra = mpra.rename(columns={'chr':'Chr','pos':'ChrPos'})
    mpra['ChrPos'] = mpra['ChrPos'].astype(int)
    return balance_df(mpra)


def plot_curves(models: dict, neg_set: int):
    categories = ['Promoter','UTR3','UTR5','ORF','Downstream','Upstream','DownstreamClose']

    # ROC
    fig, ax = plt.subplots(figsize=(7,7))
    for model_name, df in models.items():
        print(f"Processing {model_name} data for ROC/PR curves...")
        for cat in categories:
            df_cat = df if cat=='All' else df[df.locationType==cat]
            if df_cat.label.sum() < 5:
                continue
            pos = df_cat[df_cat.label==1]
            neg = df[df.label==0].sample(len(pos), random_state=SEED)
            tmp = pd.concat([pos, neg])
            fpr, tpr, _ = roc_curve(tmp.label, tmp.score)
            val = auc(fpr, tpr)

            alpha_val = 0.6 if model_name == "MPRA-RNN" else 1.0

            ax.plot(fpr, tpr,
                    color=GROUP_COLORS[cat],
                    linestyle=MODEL_STYLES[model_name],
                    alpha = alpha_val,
                    label=f"{cat} \n({model_name}, AUC={val:.2f})")

    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_aspect('equal','box')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title("ROC: Shorkie vs DREAM-RNN")
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'roc_by_model_locType_neg{neg_set}.png'), dpi=300)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(7,7))
    for model_name, df in models.items():
        for cat in categories:
            df_cat = df if cat=='All' else df[df.locationType==cat]
            if df_cat.label.sum() < 5:
                continue
            pos = df_cat[df_cat.label==1]
            neg = df[df.label==0].sample(len(pos), random_state=SEED)
            tmp = pd.concat([pos, neg])
            precision, recall, _ = precision_recall_curve(tmp.label, tmp.score)
            val = average_precision_score(tmp.label, tmp.score)
            alpha_val = 0.6 if model_name == "MPRA-RNN" else 1.0
            ax.plot(recall, precision,
                    color=GROUP_COLORS[cat],
                    linestyle=MODEL_STYLES[model_name],
                    alpha = alpha_val,
                    label=f"{cat} \n({model_name}, AUPRC={val:.2f})")

    ax.plot([1,0],[0,1],'--', color='gray')
    ax.set_aspect('equal','box')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title("PR: Shorkie vs DREAM-RNN")
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'pr_by_model_locType_neg{neg_set}.png'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    for neg_set in range(1, NUM_FOLDS//2 + 1):  # or range(1,5)
        print(f"\n=== NEGATIVE SET {neg_set} ===")
        eqtl = process_shorkie(neg_set)
        print(f"Shorkie: {len(eqtl)} samples \n({eqtl.label.sum()} pos / {len(eqtl)-eqtl.label.sum()} neg)")
        print(f"Location types: {eqtl.locationType.unique()}")

        mpra = process_mpra_rnn(neg_set)
        both = mpra.merge(
            eqtl[['Chr','ChrPos','locationType']],
            on=['Chr','ChrPos'], how='inner'
        )
        print(f"MPRA-RNN matched: {len(both)} samples \n({both.label.sum()} pos / {len(both)-both.label.sum()} neg)")

        plot_curves({'Shorkie': eqtl, 'MPRA-RNN': both}, neg_set)
        print("Plots saved in", OUTPUT_DIR)
