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

# MPRA models to compare
MPRA_MODELS    = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP  = {
    "DREAM_Atten": "DREAM-Atten",
    "DREAM_CNN":   "DREAM-CNN",
    "DREAM_RNN":   "DREAM-RNN",
}

# Which experiments/version
ROOT            = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
EXPS            = ["Caudal_etal", "Kita_etal"]
EXPS_NAME_MAP   = {
    "Caudal_etal": "Caudal et al.",
    "Kita_etal":   "Kita et al.",
}

# Negative‐set indices
NEG_SETS    = [1, 2, 3, 4]

# Default distance bins (override per-exp)
BINS        = [0, 1000, 2000, 3000, 5000, 8000]
BIN_LABELS  = ['0-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb']
BIN_MAP     = dict(zip(BIN_LABELS, BINS[1:]))

# GTF file for TSS lookup
GTF_FILE    = os.path.join(ROOT, 'data','eQTL','neg_eQTLS','GCA_000146045_2.59.gtf')

# Where to save quantiles
OUTPUT_DIR  = os.path.join(ROOT, 'experiments','SUM_data_process','eQTL_all','results')
QUANT_DIR   = os.path.join(OUTPUT_DIR, 'quantiles')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(QUANT_DIR, exist_ok=True)


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
    try:
        coord, gid = pos_gene.rsplit('_', 1)
        chrom,pos = coord.split(':')
        pos = int(pos)
    except ValueError:
        return np.nan
    info = tss_data.get(gid)
    if not info or map_chromosome_to_roman(info['chrom']) != chrom:
        return np.nan
    return abs(pos - info['tss'])

def compute_metrics_by_bin(df: pd.DataFrame, model_name: str):
    out = []
    sub = df.copy()
    sub['distance_bin'] = pd.cut(sub.distance, bins=BINS, labels=BIN_LABELS)
    for b in BIN_LABELS:
        seg = sub[sub.distance_bin==b]
        if len(seg) < 10:
            continue
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
    return out


# ─── Data loaders ─────────────────────────────────────────────────────────────

def process_shorkie(exp: str, neg_set: int, tss_data: dict) -> pd.DataFrame:
    """
    Load Shorkie ensemble scores for the given dataset (exp) and neg_set
    from the new eQTL_all/results directory.
    """
    fname = f"{exp.lower()}_scores.tsv"
    path = os.path.join(
        OUTPUT_DIR, f'negset_{neg_set}', fname
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Shorkie scores file for {exp} neg{neg_set}: {path}")
    df = pd.read_csv(path, sep='\t', usecols=['Position_Gene','Chr','ChrPos','logSED_agg','label'])
    df = df.drop_duplicates(['Position_Gene'])
    df['ensemble'] = df['logSED_agg'].abs()
    df['label']    = df['label'].astype(int)
    df['distance'] = df.Position_Gene.map(lambda x: calculate_tss_distance(x, tss_data))
    return df[['Position_Gene','Chr','ChrPos','ensemble','label','distance']]

def process_mpra(model: str, neg_set: int, mpra_base: str) -> pd.DataFrame:
    model_dir = os.path.join(mpra_base, model)
    pos = pd.read_csv(os.path.join(model_dir, 'final_pos_predictions.tsv'),
                      sep='\t', usecols=['Position_Gene','logSED'])
    neg = pd.read_csv(os.path.join(model_dir, f'final_neg_predictions_{neg_set}.tsv'),
                      sep='\t', usecols=['Position_Gene','logSED'])
    pos['label'], neg['label'] = 1, 0
    for df in (pos, neg):
        df.rename(columns={'logSED':'score'}, inplace=True)
        df['score'] = df.score.abs()
    return pd.concat([pos, neg], ignore_index=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    tss_data = parse_gtf_for_tss(GTF_FILE)
    all_metrics = []

    # storage for merged Caudal+Kita
    sh_store = {ns: [] for ns in NEG_SETS}
    mp_store = {mdl: {ns: [] for ns in NEG_SETS} for mdl in MPRA_MODELS}

    for exp in EXPS:
        # adjust bins per experiment
        if exp == "Caudal_etal":
            BINS[:]       = [0,1000,2000,3000,5000,8000]
            BIN_LABELS[:] = ['0-1kb','1kb-2kb','2kb-3kb','3kb-5kb','5kb-8kb']
        else:
            BINS[:]       = [0,1000,1500,2000,3000]
            BIN_LABELS[:] = ['0-1kb','1kb-1.5kb','1.5kb-2kb','2kb-3kb']
        BIN_MAP.update(dict(zip(BIN_LABELS, BINS[1:])))

        # MPRA directories (unchanged)
        if exp == "Caudal_etal":
            MPRA_BASE  = os.path.join(ROOT,'experiments','SUM_data_process','eQTL_MPRA_models_eval','results')
        else:
            MPRA_BASE  = os.path.join(ROOT,'experiments','SUM_data_process',
                                      'eQTL_MPRA_models_eval_kita_etal_select','results')

        for ns in NEG_SETS:
            print(f"Processing {exp} neg{ns}…")

            # Shorkie
            eqtl_df = process_shorkie(exp, ns, tss_data)
            sh_df   = eqtl_df.rename(columns={'ensemble':'score'})
            sh_store[ns].append(eqtl_df)  # store for merged

            # full‐set metrics
            all_metrics += compute_metrics_by_bin(sh_df, f"Shorkie (neg{ns})")

            # filtered Shorkie quantiles (common with MPRA)
            gene_sets = []
            for mdl in MPRA_MODELS:
                mpdf = process_mpra(mdl, ns, MPRA_BASE)
                gene_sets.append(set(mpdf.Position_Gene))
                mp_store[mdl][ns].append(mpdf)  # store for merged

            common_genes = set(sh_df.Position_Gene).intersection(*gene_sets)
            sh_common   = sh_df[sh_df.Position_Gene.isin(common_genes)].copy()
            sh_common['quantile'] = sh_common.score.rank(method='max', pct=True)
            sh_common      = sh_common.sort_values('score', ascending=False)
            out_sh = os.path.join(QUANT_DIR, f"{exp}_Shorkie_neg{ns}_quantiles.tsv")
            sh_common.to_csv(out_sh, sep='\t', index=False)
            print(f"  Wrote filtered Shorkie quantiles → {out_sh}")

            # MPRA per‐exp
            for model in MPRA_MODELS:
                display = MPRA_NAME_MAP[model]
                mpra_df = mp_store[model][ns][-1]  # last appended
                common  = (
                    eqtl_df[['Position_Gene','ensemble','label','distance']]
                    .merge(mpra_df[['Position_Gene','score']], on='Position_Gene')
                )
                all_metrics += compute_metrics_by_bin(
                    common[['label','distance','score']],
                    f"{display} (neg{ns})"
                )
                mp_q = common[['Position_Gene','score','label','distance']].copy()
                mp_q['quantile'] = mp_q.score.rank(method='max',pct=True)
                mp_q      = mp_q.sort_values('score', ascending=False)
                out_mp = os.path.join(QUANT_DIR, f"{exp}_{model}_neg{ns}_quantiles.tsv")
                mp_q.to_csv(out_mp, sep='\t', index=False)
                print(f"  Wrote {display} quantiles → {out_mp}")

    # ─── Merged Caudal+Kita quantiles ───────────────────────────────────────────
    for ns in NEG_SETS:
        print(f"Writing merged quantiles for neg{ns}…")

        merged_eqtl = pd.concat(sh_store[ns], ignore_index=True).drop_duplicates('Position_Gene')
        merged_mpra_all = {
            mdl: pd.concat(mp_store[mdl][ns], ignore_index=True).drop_duplicates('Position_Gene')
            for mdl in MPRA_MODELS
        }

        common_genes = set(merged_eqtl.Position_Gene).intersection(
            *(set(df.Position_Gene) for df in merged_mpra_all.values())
        )

        # Shorkie merged
        shm = merged_eqtl.rename(columns={'ensemble':'score'})
        shm_common = shm[shm.Position_Gene.isin(common_genes)].copy()
        shm_common['quantile'] = shm_common.score.rank(method='max', pct=True)
        shm_common = shm_common.sort_values('score', ascending=False)
        out_shm = os.path.join(QUANT_DIR, f"merged_Shorkie_neg{ns}_quantiles.tsv")
        shm_common.to_csv(out_shm, sep='\t', index=False)
        print(f"  Wrote merged Shorkie quantiles → {out_shm}")

        # DREAM merged
        for mdl in MPRA_MODELS:
            merged_mpra = merged_mpra_all[mdl]
            mpra_common = merged_mpra[merged_mpra.Position_Gene.isin(common_genes)].copy()
            mpra_common['quantile'] = mpra_common.score.rank(method='max', pct=True)
            mpra_common = mpra_common.sort_values('score', ascending=False)
            out_mpm = os.path.join(QUANT_DIR, f"merged_{mdl}_neg{ns}_quantiles.tsv")
            mpra_common.to_csv(out_mpm, sep='\t', index=False)
            print(f"  Wrote merged {MPRA_NAME_MAP[mdl]} quantiles → {out_mpm}")


if __name__ == "__main__":
    main()
