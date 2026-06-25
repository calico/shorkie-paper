#!/usr/bin/env python3
"""Shared MPRA loaders for the Figure-6 deep-recheck builders (panels D–H).

Faithful ports of the source-of-truth analysis scripts:
  - single-seq Shorkie blue : MPRA_scatter_regression_single.py
  - single-seq DREAM  green : MPRA_ref_model_viz_single_index.py
  - dual-seq   Shorkie blue : MPRA_scatter_regression_dual_trim.py
  - dual-seq   DREAM  green : MPRA_ref_model_viz_dual_indices.py

Key facts locked from those scripts + the on-disk data:
  * Shorkie per-context logSED NPZ live in the COMPLETE stranded tree
    <work>/experiments/SUM_data_process/MPRA/MPRA_promoter_seqs/results/
    single_measurement_stranded/all_seq_types/<seq_type>/<SYM>_<ORF>_{pos|neg}_outputs/
    (single-seq files are `<ORF>_context_<i>_*.npz` with key `logSED`;
     dual-seq files are `<ORF>_ctx<i>_*.npz` with keys logSED_ALT_ORIG / logSED_REF_ORIG).
  * The published 6D uses THIS stranded tree (R=0.695); the old reproduction read the
    `scores_avg/results` tree and got 0.644 — that was the bug.
  * DREAM-RNN per-sequence predictions: data/random-promoter-dream-challenge-2022/data/DREAM-RNN_output.txt
  * Ground truth (MAUDE expression): data/MPRA/filtered_test_data_with_MAUDE_expression.txt
  * Index maps: yeast uses df['pos'] directly; random/challenging subsample via
    test_subset_ids/fix/<seq>_sample_ids.tsv (original_row_id, 1-indexed); SNV/motif use
    (alt_pos, ref_pos) pairs subsampled the same way.

Plot orientation matches the PUBLISHED panels: model prediction on X, expression on Y
(Pearson/Spearman are symmetric, so the printed correlations are unchanged).
"""
import os
import re
import csv
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from shorkie import config

config.load()
WORK = str(config.path("work_root"))
MPRA_DATA = f"{WORK}/data/MPRA"
GT_FILE = f"{MPRA_DATA}/filtered_test_data_with_MAUDE_expression.txt"
DREAM_FILE = f"{WORK}/data/random-promoter-dream-challenge-2022/data/DREAM-RNN_output.txt"
STRANDED = (f"{WORK}/experiments/SUM_data_process/MPRA/MPRA_promoter_seqs/"
            f"results/single_measurement_stranded/all_seq_types")

# Reporter-gene panel (symbol -> systematic ORF), exactly as the source scripts.
POS_GENES = {
    "GPM3": "YOL056W", "SLI1": "YGR212W", "VPS52": "YDR484W",
    "YMR160W": "YMR160W", "MRPS28": "YDR337W", "YCT1": "YLL055W",
    "RDL2": "YOR286W", "PHS1": "YJL097W", "RTC3": "YHR087W", "MSN4": "YKL062W",
}
NEG_GENES = {
    "COA4": "YLR218C", "ERI1": "YPL096C-A", "RSM25": "YIL093C",
    "ERD1": "YDR414C", "MRM2": "YGL136C", "SNT2": "YGL131C",
    "CSI2": "YOL007C", "RPE1": "YJL121C", "PKC1": "YBL105C",
    "AIM11": "YER093C-A", "MAE1": "YKL029C", "MRPL1": "YDR116C",
}

# Published palette: Shorkie blue (matplotlib C0), DREAM green, regression red.
COL_SHORKIE = "#1f77b4"
COL_DREAM = "green"
COL_REG = "red"

ORF = {**POS_GENES, **NEG_GENES}
GENE_STRAND = {**{s: "+" for s in POS_GENES}, **{s: "-" for s in NEG_GENES}}


# --------------------------------------------------------------------------- #
# Ground truth + DREAM per-sequence predictions
# --------------------------------------------------------------------------- #
def load_ground_truth():
    with open(GT_FILE) as f:
        lines = list(csv.reader(f, delimiter="\t"))
    return np.array([float(l[1]) for l in lines], dtype=np.float64)


def load_dream():
    with open(DREAM_FILE) as f:
        lines = list(csv.reader(f, delimiter="\t"))
    return np.array([float(l[1]) for l in lines], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Shorkie NPZ aggregation over the stranded tree
# --------------------------------------------------------------------------- #
def _ctx(fname):
    b = os.path.basename(fname)
    m = re.search(r"_ctx(\d+)_", b) or re.search(r"_context_(\d+)_", b)
    return 100 + int(m.group(1)) * 10 if m else 0


def _gene_dir(seq_type, sym, orf, strand):
    tag = "pos" if strand == "+" else "neg"
    return os.path.join(STRANDED, seq_type, f"{sym}_{orf}_{tag}_outputs")


# Published panels D-H use the 180 bp insertion context only (panel titles say
# "...180 bp"; the manuscript: "position 180 bp upstream was selected for subsequent
# analyses"). Averaging over all 11 contexts shifts 6D/6E/6H by up to 0.04; the 180 bp
# context reproduces every published Shorkie correlation to <0.001. SITE=180.
SITE = 180


def _site_files(d, orf, dual):
    pat = f"{orf}_ctx*.npz" if dual else f"{orf}_context_*.npz"
    files = sorted(glob.glob(os.path.join(d, pat)), key=_ctx)
    if not dual and not files:
        files = sorted(glob.glob(os.path.join(d, f"{orf}_ctx*.npz")), key=_ctx)
    return [f for f in files if _ctx(f) == SITE]


def _per_gene_single(seq_type, sym, orf, strand):
    """Per-sequence mean logSED at the 180 bp insertion context."""
    files = _site_files(_gene_dir(seq_type, sym, orf, strand), orf, dual=False)
    if not files:
        return None
    per_ctx = [np.mean(np.load(fp)["logSED"].astype(np.float64), axis=1) for fp in files]
    return np.mean(np.stack(per_ctx, axis=0), axis=0)


def _per_gene_dual(seq_type, sym, orf, strand):
    """Per-sequence mean (ALT-REF) logSED diff at the 180 bp insertion context."""
    files = _site_files(_gene_dir(seq_type, sym, orf, strand), orf, dual=True)
    if not files:
        return None
    per_ctx = []
    for fp in files:
        z = np.load(fp)
        diff = z["logSED_ALT_ORIG"].astype(np.float64) - z["logSED_REF_ORIG"].astype(np.float64)
        per_ctx.append(np.mean(diff, axis=1))
    return np.mean(np.stack(per_ctx, axis=0), axis=0)


def gene_site_single(seq_type, sym, orf, strand):
    """{insertion_site -> per-sequence mean logSED} for one gene (high/low panels B/C)."""
    d = _gene_dir(seq_type, sym, orf, strand)
    files = sorted(glob.glob(os.path.join(d, f"{orf}_context_*.npz")), key=_ctx)
    return {_ctx(fp): np.mean(np.load(fp)["logSED"].astype(np.float64), axis=1) for fp in files}


def aggregate_shorkie(seq_type, dual):
    """Mean of per-gene prediction vectors across all available genes (pos+neg)."""
    vecs = []
    for genes, strand in ((POS_GENES, "+"), (NEG_GENES, "-")):
        for sym, orf in genes.items():
            v = (_per_gene_dual(seq_type, sym, orf, strand) if dual
                 else _per_gene_single(seq_type, sym, orf, strand))
            if v is not None:
                vecs.append(v)
    if not vecs:
        raise RuntimeError(f"No Shorkie NPZ found for {seq_type} under {STRANDED}")
    return np.mean(np.stack(vecs, axis=0), axis=0), len(vecs)


# --------------------------------------------------------------------------- #
# Index maps (in NPZ row order)
# --------------------------------------------------------------------------- #
def single_indices(seq_type):
    """Absolute `pos` indices into the ground-truth array, in NPZ row order."""
    if seq_type == "yeast_seqs":
        return pd.read_csv(f"{MPRA_DATA}/test_subset_ids/{seq_type}.csv")["pos"].values
    df = pd.read_csv(f"{MPRA_DATA}/test_subset_ids/{seq_type}.csv")
    ids = pd.read_csv(f"{MPRA_DATA}/test_subset_ids/fix/{seq_type}_sample_ids.tsv", sep="\t")
    idx = ids["original_row_id"].astype(int).values - 1
    return df.iloc[idx]["pos"].values


def dual_pairs(seq_type):
    """List of (alt_pos, ref_pos) into the ground-truth array, in NPZ row order."""
    df = pd.read_csv(f"{MPRA_DATA}/test_subset_ids/{seq_type}.csv")
    ids = pd.read_csv(f"{MPRA_DATA}/test_subset_ids/fix/{seq_type}_sample_ids.tsv", sep="\t")
    idx = ids["original_row_id"].astype(int).values - 1
    pairs = list(zip(df["alt_pos"], df["ref_pos"]))
    return [pairs[i] for i in idx]


# --------------------------------------------------------------------------- #
# Correlation recipes (faithful to the source scripts)
# --------------------------------------------------------------------------- #
def shorkie_single(seq_type, gt_exp):
    """6D/6E Shorkie blue: pred=Shorkie logSED, gt=expression; finite mask only."""
    pred, n_genes = aggregate_shorkie(seq_type, dual=False)
    idx = single_indices(seq_type)
    gt = gt_exp[idx].astype(np.float64)
    if len(gt) != len(pred):
        raise RuntimeError(f"{seq_type}: gt len {len(gt)} != pred len {len(pred)}")
    m = np.isfinite(gt) & np.isfinite(pred)
    gt, pred = gt[m], pred[m]
    return pred, gt, pearsonr(gt, pred)[0], spearmanr(gt, pred)[0], len(gt), n_genes


def dream_single(seq_type, gt_exp, dream_exp):
    """6D/6E DREAM green: pred=DREAM, gt=expression; finite mask only."""
    idx = single_indices(seq_type).astype(int)
    gt = gt_exp[idx].astype(np.float64)
    pred = dream_exp[idx].astype(np.float64)
    m = np.isfinite(gt) & np.isfinite(pred)
    gt, pred = gt[m], pred[m]
    return pred, gt, pearsonr(gt, pred)[0], spearmanr(gt, pred)[0], len(gt)


def shorkie_dual(seq_type, gt_exp):
    """6F/6G/6H Shorkie blue: pred=Shorkie (Alt-Ref) logSED diff, gt=expression diff;
    finite mask THEN drop exact zeros (matches analyze_and_plot_group)."""
    pred, n_genes = aggregate_shorkie(seq_type, dual=True)
    pairs = dual_pairs(seq_type)
    gt = np.array([gt_exp[a] - gt_exp[r] for a, r in pairs], dtype=np.float64)
    if len(gt) != len(pred):
        raise RuntimeError(f"{seq_type}: gt len {len(gt)} != pred len {len(pred)}")
    m = np.isfinite(gt) & np.isfinite(pred)
    gt, pred = gt[m], pred[m]
    nz = (gt != 0) & (pred != 0)
    gt, pred = gt[nz], pred[nz]
    return pred, gt, pearsonr(gt, pred)[0], spearmanr(gt, pred)[0], len(gt), n_genes


def _f16(x):
    x = np.clip(x, np.finfo(np.float16).min, np.finfo(np.float16).max)
    return x.astype("float16")


def dream_dual(seq_type, gt_exp, dream_exp):
    """6F/6G/6H DREAM green: pred=DREAM (Alt-Ref) diff, gt=expression diff;
    float16-clip, skip non-finite (no zero-drop) — matches compute_dual_scores."""
    pairs = dual_pairs(seq_type)
    gt_list, pred_list = [], []
    for a, r in pairs:
        gd = gt_exp[a] - gt_exp[r]
        pd_ = dream_exp[a] - dream_exp[r]
        if not (np.isfinite(gd) and np.isfinite(pd_)):
            continue
        gt_list.append(float(_f16(np.array(gd))))
        pred_list.append(float(_f16(np.array(pd_))))
    gt = np.array(gt_list, dtype=np.float64)
    pred = np.array(pred_list, dtype=np.float64)
    return pred, gt, pearsonr(gt, pred)[0], spearmanr(gt, pred)[0], len(gt)
