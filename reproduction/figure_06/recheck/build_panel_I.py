#!/usr/bin/env python3
"""Figure 6 panel I — endogenous RNA-seq coverage prediction (held-out test set).

Faithful port of MPRA_RNASeq/9_combined_density_subplot.py: two gaussian-kde 2D-density
subplots of log2(prediction+1) vs log2(Mean T0 RNA-Seq coverage+1) — Shorkie (all 8
folds, valid+test) on the left, DREAM-RNN (180bp upstream, all splits) on the right,
red trend line, Pearson/Spearman annotated, shared viridis "Density" colorbar.

Published targets: Shorkie Pearson r=0.895, Spearman ρ=0.837; DREAM r=0.249, ρ=0.261.

Inputs (cached on disk; no GPU):
  Shorkie : <SB>/f{0..7}c0/RNA-Seq/{valid,test}/gene_{preds,targets}_stats.tsv  (col 'mean')
  DREAM   : MPRA_RNASeq/predictions/upstream_180bp_predictions.tsv ('prediction')
            target = <SB>/f0c0/RNA-Seq/{train,valid,test}/gene_targets_stats.tsv ('mean')

Outputs reproduced/Figure_6I.png and recheck/fig6_I_R.csv.
"""
import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, gaussian_kde

from shorkie import config

config.load()
WORK = str(config.path("work_root"))
REPRO = config.repo_root() / "reproduction" / "figure_06" / "reproduced"
REPRO.mkdir(parents=True, exist_ok=True)
RECHECK = config.repo_root() / "reproduction" / "figure_06" / "recheck"

SB = (f"{WORK}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/"
      f"self_supervised_unet_small_bert_drop/Shorkie_all_gene_eval_rc")
FOLDS = [f"f{i}c0" for i in range(8)]
SUB_DIR = "RNA-Seq"
DREAM_BASE = f"{SB}/f0c0/RNA-Seq"
DREAM_PRED_TSV = f"{WORK}/experiments/SUM_data_process/MPRA/MPRA_RNASeq/predictions/upstream_180bp_predictions.tsv"


def load_shorkie(stat="mean"):
    dfs = []
    for fold in FOLDS:
        for split in ("valid", "test"):
            pf = os.path.join(SB, fold, SUB_DIR, split, "gene_preds_stats.tsv")
            tf = os.path.join(SB, fold, SUB_DIR, split, "gene_targets_stats.tsv")
            if not (os.path.isfile(pf) and os.path.isfile(tf)):
                continue
            dp = pd.read_csv(pf, sep="\t", usecols=["gene_id", stat]).rename(columns={stat: f"{stat}_pred"})
            dt = pd.read_csv(tf, sep="\t", usecols=["gene_id", stat]).rename(columns={stat: f"{stat}_target"})
            m = pd.merge(dt, dp, on="gene_id")
            if not m.empty:
                dfs.append(m)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_dream(pred_col="prediction"):
    dp = pd.read_csv(DREAM_PRED_TSV, sep="\t", usecols=["gene_id", pred_col]).rename(columns={pred_col: "mean_pred"})
    dfs = []
    for split in ("train", "valid", "test"):
        tf = os.path.join(DREAM_BASE, split, "gene_targets_stats.tsv")
        if not os.path.isfile(tf):
            continue
        dt = pd.read_csv(tf, sep="\t", usecols=["gene_id", "mean"]).rename(columns={"mean": "mean_target"})
        m = pd.merge(dt, dp, on="gene_id", how="inner")
        if not m.empty:
            dfs.append(m)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def process(df, stat="mean"):
    if df.empty:
        return None
    y = np.log2(df[f"{stat}_target"].astype(float) + 1)   # Mean T0 RNA-Seq coverage
    x = np.log2(df[f"{stat}_pred"].astype(float) + 1)      # Prediction
    z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    s, b = np.polyfit(x, y, 1)
    xv = np.array([x.min(), x.max()])
    return dict(x=x, y=y, z=z, r=r, rho=rho, xv=xv, yv=s * xv + b, n=len(x))


def subplot(ax, d, title, vmin, vmax):
    if d is None:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center"); return None
    sc = ax.scatter(d["x"], d["y"], c=d["z"], s=15, alpha=0.6, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.plot(d["xv"], d["yv"], color="red", lw=2)
    ax.set_xlabel("Prediction (log₂ scale)", fontsize=12)
    ax.set_ylabel("Mean T₀ RNA-Seq Coverage (log₂ scale)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_box_aspect(1)
    ax.text(0.05, 0.95, f"Pearson r = {d['r']:.3f}\nSpearman ρ = {d['rho']:.3f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=12)
    return sc


def main():
    sh = process(load_shorkie())
    dr = process(load_dream())
    zs = [d["z"] for d in (sh, dr) if d is not None]
    vmin, vmax = (np.concatenate(zs).min(), np.concatenate(zs).max()) if zs else (None, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sc1 = subplot(axes[0], sh, "Shorkie Prediction vs RNA-Seq coverage\n(all folds)", vmin, vmax)
    sc2 = subplot(axes[1], dr, "DREAM-RNN Prediction vs RNA-Seq coverage\n(180bp, all splits)", vmin, vmax)
    plt.subplots_adjust(wspace=0.3)
    if sc1 or sc2:
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        fig.colorbar(sc1 if sc1 else sc2, cax=cbar_ax, label="Density")
    fig.suptitle("RNA-Seq Coverage Prediction (held-out test set)", fontsize=14)
    fig.text(0.02, 0.97, "I", fontsize=20, fontweight="bold", va="top", ha="left")
    out = REPRO / "Figure_6I.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[6I] Shorkie n={sh['n']} Pearson r={sh['r']:.3f} Spearman rho={sh['rho']:.3f} (pub 0.895/0.837)")
    print(f"[6I] DREAM   n={dr['n']} Pearson r={dr['r']:.3f} Spearman rho={dr['rho']:.3f} (pub 0.249/0.261)")
    print("saved", out)
    with open(RECHECK / "fig6_I_R.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pearson_repro", "spearman_repro", "n", "published_pearson", "published_spearman"])
        w.writerow(["Shorkie", f"{sh['r']:.4f}", f"{sh['rho']:.4f}", sh["n"], 0.895, 0.837])
        w.writerow(["DREAM-RNN", f"{dr['r']:.4f}", f"{dr['rho']:.4f}", dr["n"], 0.249, 0.261])


if __name__ == "__main__":
    main()
