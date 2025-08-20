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
NUM_FOLDS   = 8
FOLDS       = [f"f{i}c0" for i in range(NUM_FOLDS)]
ROOT        = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
SEED        = 42

MPRA_MODELS   = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

random.seed(SEED)
np.random.seed(SEED)

for neg_set in range(1, 5):
    print(f"\n=== Using MPRA negative set {neg_set} ===")
    OUT_DIR = f'results/set{neg_set}/'
    TSV_DIR = os.path.join(
        ROOT, 'experiments', 'SUM_data_process',
        'eQTL_kita_etal_select_exp', 'results', f'set{neg_set}'
    )
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
        base    = os.path.join(
            ROOT, 'experiments', 'SUM_data_process',
            'eQTL_MPRA_models_eval_kita_etal_select','results', mdl
        )
        pos_tsv = os.path.join(base, 'final_pos_predictions.tsv')
        neg_tsv = os.path.join(base, 'final_neg_predictions_1.tsv')
        df      = process_mpra_data(pos_tsv, neg_tsv).dropna(subset=['score','label'])
        df['score'] = df['score'].abs()
        pos_df = df[df['label']==1]
        neg_df = df[df['label']==0]
        if len(pos_df) > len(neg_df):
            pos_df = pos_df.sample(len(neg_df), random_state=SEED)
        elif len(neg_df) > len(pos_df):
            neg_df = neg_df.sample(len(pos_df), random_state=SEED)
        mpra_balanced[mdl] = pd.concat([pos_df, neg_df], ignore_index=True)
        print(f"[MPRA {mdl}] balanced to {len(pos_df)} pos / {len(neg_df)} neg")

    # ─── Load all yeast per-fold TSVs (with locationType) ────────────────────
    yeast_dfs = {}
    for fold in FOLDS:
        path = os.path.join(TSV_DIR, f'yeast_eqtl_{fold}.tsv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing TSV for fold {fold}: {path}")
        df = pd.read_csv(path, sep='\t',
                         usecols=['Position_Gene','score','label','locationType'])
        yeast_dfs[fold] = df.drop_duplicates(['Position_Gene'])

    # ─── Identify common positives & negatives ───────────────────────────────
    pos_sets   = [set(df[df['label']==1]['Position_Gene']) for df in yeast_dfs.values()]
    neg_sets   = [set(df[df['label']==0]['Position_Gene']) for df in yeast_dfs.values()]
    common_pos = set.intersection(*pos_sets)
    common_neg = set.intersection(*neg_sets)
    print(f"[Yeast] common pos={len(common_pos)}, common neg={len(common_neg)}")

    # ─── Build unified yeast DataFrame ───────────────────────────────────────
    keys         = sorted(common_pos) + sorted(common_neg)
    yeast_scores = pd.DataFrame({'Position_Gene': keys})
    score_cols   = []
    for idx, (fold, df) in enumerate(yeast_dfs.items()):
        col = f'score_{fold}'
        score_cols.append(col)
        m = df.set_index('Position_Gene')['score']
        yeast_scores[col] = yeast_scores['Position_Gene'].map(m).abs()

    yeast_scores['label']    = yeast_scores['Position_Gene'].isin(common_pos).astype(int)
    yeast_scores['ensemble'] = yeast_scores[score_cols].mean(axis=1)

    # ─── Map each positive gene to its locationType ───────────────────────────
    loc_map = {}
    for df in yeast_dfs.values():
        for pg, lt in df[df['label']==1][['Position_Gene','locationType']].itertuples(index=False):
            if pg not in loc_map and pd.notna(lt):
                loc_map[pg] = lt
    yeast_scores['locationType'] = yeast_scores['Position_Gene'].map(loc_map)

    # ─── Print counts per locationType ───────────────────────────────────────
    counts = yeast_scores.groupby('locationType')['label'].value_counts().unstack(fill_value=0)
    print("\nPositive / Negative counts per locationType:")
    for loc, row in counts.iterrows():
        pos = row.get(1, 0)
        neg = row.get(0, 0)
        print(f"  {loc}: {pos} positives, {neg} negatives")

    # ─── Sample negatives to match positives for "All" group & compute overall metrics ─────────────────────────────────────────
    pos_all        = yeast_scores[yeast_scores['label']==1]
    neg_all        = yeast_scores[yeast_scores['label']==0]
    total_pos      = len(pos_all)
    total_neg_orig = len(neg_all)
    # if total_neg_orig >= total_pos:
    #     neg_sample = neg_all.sample(n=total_pos, random_state=SEED)
    # else:
    #     extra_neg  = neg_all.sample(n=total_pos - total_neg_orig, replace=True, random_state=SEED)
    #     neg_sample = pd.concat([neg_all, extra_neg], ignore_index=True)
    neg_sample = neg_all
    df_all = pd.concat([pos_all, neg_sample], ignore_index=True)

    fpr_all, tpr_all, _ = roc_curve(df_all['label'], df_all['ensemble'])
    auc_all            = auc(fpr_all, tpr_all)
    prec_all, rec_all, _ = precision_recall_curve(df_all['label'], df_all['ensemble'])
    ap_all             = average_precision_score(df_all['label'], df_all['ensemble'])
    total_neg          = len(neg_sample)

    # ─── Plot ROC by locationType + overall (legend on right) ───────────────
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot([0,1], [0,1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # overall
    ax.plot(
        fpr_all, tpr_all,
        color='black', linestyle='-', linewidth=3,
        label=f"All (AUC={auc_all:.2f})"
        # label=f"All ({total_pos} pos / {total_neg} neg, AUC={auc_all:.2f})"
    )

    # each locationType (only if at least 100 positives)
    loc_types = sorted([lt for lt in yeast_scores['locationType'].unique() if pd.notna(lt)])
    for loc in loc_types:
        pos_df = yeast_scores[
            (yeast_scores['locationType'] == loc) &
            (yeast_scores['label'] == 1)
        ]
        if loc == "DownstreamClose":
            continue  # Skip this locationType as per original code
        if loc == "Upstream":
            loc = "Upstream TSS"
        elif loc == "Downstream":
            loc = "Downstream TES"
        # if len(pos_df) < 100:
        #     continue
        neg_df = yeast_scores[yeast_scores['label'] == 0].sample(
            n=len(pos_df), random_state=SEED
        )
        n_pos = len(pos_df)
        n_neg = len(neg_df)
        df_loc = pd.concat([pos_df, neg_df], ignore_index=True)
        fpr_l, tpr_l, _ = roc_curve(df_loc['label'], df_loc['ensemble'])
        auc_l           = auc(fpr_l, tpr_l)
        ax.plot(
            fpr_l, tpr_l,
            linewidth=2,
            label=f"{loc} \n({n_pos} pos / {n_neg} neg, AUC={auc_l:.2f})"
        )

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Shorkie ROC by locationType', fontsize=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=False)
    # plt.legend(loc='lower left', fontsize=9)  # Increase font size of legend

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'roc_by_locationType.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ─── Plot PR by locationType + overall (legend on right) ────────────────
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot([1,0], [0,1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # overall
    ax.plot(
        rec_all, prec_all,
        color='black', linestyle='-', linewidth=3,
        label=f"All ({total_pos} pos / {total_neg} neg, AUPRC={ap_all:.2f})"
    )

    # each locationType (only if at least 100 positives)
    for loc in loc_types:
        pos_df = yeast_scores[
            (yeast_scores['locationType'] == loc) &
            (yeast_scores['label'] == 1)
        ]
        if loc == "DownstreamClose":
            continue  # Skip this locationType as per original code
        if loc == "Upstream":
            loc = "Upstream TSS"
        elif loc == "Downstream":
            loc = "Downstream TES"

        # if len(pos_df) < 100:
        #     continue
        neg_df = yeast_scores[yeast_scores['label'] == 0].sample(
            n=len(pos_df), random_state=SEED
        )
        n_pos = len(pos_df)
        n_neg = len(neg_df)
        df_loc = pd.concat([pos_df, neg_df], ignore_index=True)
        prec_l, rec_l, _ = precision_recall_curve(df_loc['label'], df_loc['ensemble'])
        ap_l             = average_precision_score(df_loc['label'], df_loc['ensemble'])
        ax.plot(
            rec_l, prec_l,
            linewidth=2,
            label=f"{loc} \n({n_pos} pos / {n_neg} neg, AUPRC={ap_l:.2f})"
        )

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Shorkie PR by locationType', fontsize=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=False)
    # plt.legend(loc='lower left', fontsize=9)  # Increase font size of legend

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pr_by_locationType.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ─── (Optional) your MPRA vs Shorkie overall plots here ─────────────────
    # …

    print("Done. Plots written to:")
    print(f"  {os.path.join(OUT_DIR,'roc_by_locationType.png')}")
    print(f"  {os.path.join(OUT_DIR,'pr_by_locationType.png')}")
