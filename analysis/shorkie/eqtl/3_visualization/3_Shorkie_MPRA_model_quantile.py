#!/usr/bin/env python3
"""
Compute per-variant quantile rankings for each model.

Changes from previous version:
  - No class balancing
  - Adds summary TSV per experiment with mean ± SEM of positive-eQTL quantiles
    across the 4 negative sets
"""
import os
import re
import random

import numpy as np
import pandas as pd
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

ROOT = args.root_dir
EXPS = ["caudal_etal", "kita_etal", "Renganaath_etal"]
EXPS_NAME_MAP = {
    "caudal_etal":     "Caudal et al.",
    "kita_etal":       "Kita et al.",
    "Renganaath_etal": "Renganaath et al.",
}

NEG_SETS = [1, 2, 3, 4]

OUTPUT_DIR = os.path.join(ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results')
QUANT_DIR = os.path.join(OUTPUT_DIR, 'quantiles')
os.makedirs(QUANT_DIR, exist_ok=True)


# ─── Data loaders ─────────────────────────────────────────────────────────────

def process_shorkie(exp: str, neg_set: int) -> pd.DataFrame:
    base_dir = os.path.join(ROOT, 'revision_experiments', 'eQTL', 'viz_new', 'results',
                            f'negset_{neg_set}')
    # Base Shorkie
    df_orig = pd.read_csv(os.path.join(base_dir, f"{exp}_Shorkie_scores.tsv"),
                          sep='\t', usecols=['Position_Gene', 'logSED_agg', 'label', 'distance'])
    df_orig['Position_Gene'] = df_orig['Position_Gene'].astype(str).str.strip()
    df_orig = df_orig.drop_duplicates(['Position_Gene'])
    df_orig['Shorkie'] = df_orig['logSED_agg'].abs()
    df_orig = df_orig.drop(columns=['logSED_agg'])

    # LM
    df_lm = pd.read_csv(os.path.join(base_dir, f"{exp}_Shorkie_LM_scores.tsv"),
                         sep='\t', usecols=['Position_Gene', 'LLR', 'label'])
    df_lm['Position_Gene'] = df_lm['Position_Gene'].astype(str).str.strip()
    df_lm = df_lm.drop_duplicates(['Position_Gene'])
    df_lm['Shorkie_LM'] = df_lm['LLR'].abs()
    df_lm = df_lm.drop(columns=['LLR', 'label'])

    # Random Init
    df_ri = pd.read_csv(os.path.join(base_dir, f"{exp}_Shorkie_Random_Init_scores.tsv"),
                         sep='\t', usecols=['Position_Gene', 'logSED_agg'])
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
        df['score'] = df.score.abs()
    return pd.concat([pos, neg], ignore_index=True)


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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Storage for merged across all EXPS
    sh_store = {ns: [] for ns in NEG_SETS}
    mp_store = {mdl: {ns: [] for ns in NEG_SETS} for mdl in MPRA_MODELS}

    # Summary: per-model, per-negset median quantile of positives
    # Structure: {exp: {model: [median_q_neg1, median_q_neg2, ...]}}
    summary_data = {}

    for exp in EXPS:
        print(f"\n=== {exp} ===")
        MPRA_BASE = get_mpra_base(exp)
        summary_data[exp] = {}

        for ns in NEG_SETS:
            print(f"  neg{ns}...")

            # Load Shorkie Combined
            try:
                eqtl_df = process_shorkie(exp, ns)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue
            sh_store[ns].append(eqtl_df)

            # Load MPRA
            gene_sets = []
            if MPRA_BASE:
                for mdl in MPRA_MODELS:
                    try:
                        mpdf = process_mpra(mdl, ns, MPRA_BASE)
                        gene_sets.append(set(mpdf.Position_Gene))
                        mp_store[mdl][ns].append(mpdf)
                    except FileNotFoundError:
                        pass

            # Common genes intersection
            if gene_sets:
                common_genes = set(eqtl_df.Position_Gene).intersection(*gene_sets)
            else:
                common_genes = set(eqtl_df.Position_Gene)

            # Export per-Shorkie-variant quantiles
            for variant in ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init']:
                sh_common = eqtl_df[eqtl_df.Position_Gene.isin(common_genes)].copy()
                sh_common['score'] = sh_common[variant]
                # Filter zeros before ranking
                sh_common = sh_common[sh_common['score'] > 0].copy()
                sh_common['quantile'] = sh_common.score.rank(method='max', pct=True)
                sh_common = sh_common.sort_values('score', ascending=False)

                out_sh = os.path.join(QUANT_DIR, f"{exp}_{variant}_neg{ns}_quantiles.tsv")
                sh_common.to_csv(out_sh, sep='\t', index=False)
                print(f"    Wrote {variant} quantiles → {out_sh}")

                # Track median quantile of positives
                pos_q = sh_common.loc[sh_common.label == 1, 'quantile']
                median_q = pos_q.median() if len(pos_q) > 0 else np.nan
                summary_data[exp].setdefault(variant, []).append(median_q)

            # Export per-MPRA model quantiles
            if MPRA_BASE:
                for model in MPRA_MODELS:
                    display = MPRA_NAME_MAP[model]
                    if not mp_store[model][ns]:
                        continue
                    mpra_df = mp_store[model][ns][-1]

                    common = (
                        eqtl_df[['Position_Gene', 'label', 'distance']]
                        .merge(mpra_df[['Position_Gene', 'score']], on='Position_Gene')
                    )
                    common = common[common.Position_Gene.isin(common_genes)].copy()
                    common = common[common['score'] > 0].copy()
                    common['quantile'] = common.score.rank(method='max', pct=True)
                    common = common.sort_values('score', ascending=False)

                    out_mp = os.path.join(QUANT_DIR, f"{exp}_{model}_neg{ns}_quantiles.tsv")
                    common.to_csv(out_mp, sep='\t', index=False)
                    print(f"    Wrote {display} quantiles → {out_mp}")

                    pos_q = common.loc[common.label == 1, 'quantile']
                    median_q = pos_q.median() if len(pos_q) > 0 else np.nan
                    summary_data[exp].setdefault(display, []).append(median_q)

        # Write per-experiment summary (mean ± SEM across 4 neg sets)
        summary_rows = []
        for model_name, medians in summary_data[exp].items():
            valid = [v for v in medians if not np.isnan(v)]
            if len(valid) >= 2:
                summary_rows.append({
                    'model': model_name,
                    'n_sets': len(valid),
                    'mean_median_quantile': np.mean(valid),
                    'sem_median_quantile': sp_stats.sem(valid),
                    'per_set_medians': ','.join(f"{v:.4f}" for v in valid),
                })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_out = os.path.join(QUANT_DIR, f"{exp}_quantile_summary.tsv")
            summary_df.to_csv(summary_out, sep='\t', index=False)
            print(f"\n  Summary → {summary_out}")
            print(summary_df.to_string(index=False))

    # ─── Merged ALL EXPS quantiles ───────────────────────────────────────────
    global_summary = {}

    for ns in NEG_SETS:
        print(f"\nWriting merged quantiles for neg{ns}...")

        if not sh_store[ns]:
            continue
        merged_eqtl = pd.concat(sh_store[ns], ignore_index=True).drop_duplicates('Position_Gene')

        merged_mpra_all = {}
        for mdl in MPRA_MODELS:
            if mp_store[mdl][ns]:
                merged_mpra_all[mdl] = pd.concat(mp_store[mdl][ns], ignore_index=True).drop_duplicates('Position_Gene')

        if merged_mpra_all:
            common_genes = set(merged_eqtl.Position_Gene).intersection(
                *(set(df.Position_Gene) for df in merged_mpra_all.values())
            )
        else:
            common_genes = set(merged_eqtl.Position_Gene)

        for variant in ['Shorkie_LM', 'Shorkie', 'Shorkie_Random_Init']:
            shm = merged_eqtl[merged_eqtl.Position_Gene.isin(common_genes)].copy()
            shm['score'] = shm[variant]
            shm = shm[shm['score'] > 0].copy()
            shm['quantile'] = shm.score.rank(method='max', pct=True)
            shm = shm.sort_values('score', ascending=False)
            out_shm = os.path.join(QUANT_DIR, f"merged_{variant}_neg{ns}_quantiles.tsv")
            shm.to_csv(out_shm, sep='\t', index=False)
            print(f"  Wrote merged {variant} → {out_shm}")

            pos_q = shm.loc[shm.label == 1, 'quantile']
            median_q = pos_q.median() if len(pos_q) > 0 else np.nan
            global_summary.setdefault(variant, []).append(median_q)

        for mdl in MPRA_MODELS:
            if mdl not in merged_mpra_all:
                continue
            display = MPRA_NAME_MAP[mdl]
            merged_mpra = merged_mpra_all[mdl]
            mpra_c = merged_mpra[merged_mpra.Position_Gene.isin(common_genes)].copy()
            mpra_c = mpra_c[mpra_c['score'] > 0].copy()
            mpra_c['quantile'] = mpra_c.score.rank(method='max', pct=True)
            mpra_c = mpra_c.sort_values('score', ascending=False)
            out_mpm = os.path.join(QUANT_DIR, f"merged_{mdl}_neg{ns}_quantiles.tsv")
            mpra_c.to_csv(out_mpm, sep='\t', index=False)
            print(f"  Wrote merged {display} → {out_mpm}")

            pos_q = mpra_c.loc[mpra_c.label == 1, 'quantile']
            median_q = pos_q.median() if len(pos_q) > 0 else np.nan
            global_summary.setdefault(display, []).append(median_q)

    # Write global summary
    global_rows = []
    for model_name, medians in global_summary.items():
        valid = [v for v in medians if not np.isnan(v)]
        if len(valid) >= 2:
            global_rows.append({
                'model': model_name,
                'n_sets': len(valid),
                'mean_median_quantile': np.mean(valid),
                'sem_median_quantile': sp_stats.sem(valid),
                'per_set_medians': ','.join(f"{v:.4f}" for v in valid),
            })
    if global_rows:
        gdf = pd.DataFrame(global_rows)
        gout = os.path.join(QUANT_DIR, f"merged_quantile_summary.tsv")
        gdf.to_csv(gout, sep='\t', index=False)
        print(f"\nGlobal Summary → {gout}")
        print(gdf.to_string(index=False))


if __name__ == "__main__":
    main()
