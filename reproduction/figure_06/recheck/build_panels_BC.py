#!/usr/bin/env python3
"""Figure 6 panels B/C (AUROC/AUPRC of high- vs low-expression sequences across
insertion sites) and the 6A insertion schematic.

For each of the 18 reporter genes (6 per expression quantile: 5-25 / 25-75 / 75-95),
Shorkie's per-sequence logSED classifies high- vs low-expression library sequences at
each insertion site (100-200 bp). AUROC (6B) and AUPRC (6C) per site are plotted as
per-gene dashed lines plus the three quantile aggregates (mean +/- SE), matching the
published panels. Both stay > 0.95 across all sites.

Outputs reproduced/Figure_6BC.png, reproduced/Figure_6A_schematic.png, and
recheck/fig6_BC.csv (mean AUROC/AUPRC).
"""
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpra_common as mc
from shorkie import config

config.load()
REPRO = config.repo_root() / "reproduction" / "figure_06" / "reproduced"
REPRO.mkdir(parents=True, exist_ok=True)
RECHECK = config.repo_root() / "reproduction" / "figure_06" / "recheck"

SITES = list(range(100, 201, 10))
QUANTILES = {
    "5-25":  ["GPM3", "SLI1", "VPS52", "COA4", "ERI1", "RSM25"],
    "25-75": ["YMR160W", "MRPS28", "YCT1", "ERD1", "MRM2", "SNT2"],
    "75-95": ["RDL2", "PHS1", "RTC3", "CSI2", "RPE1", "PKC1"],
}
AGG_COLOR = {"5-25": "darkgreen", "25-75": "red", "75-95": "black"}


def gene_site_metrics(sym):
    orf, strand = mc.ORF[sym], mc.GENE_STRAND[sym]
    hi = mc.gene_site_single("high_exp_seqs", sym, orf, strand)
    lo = mc.gene_site_single("low_exp_seqs", sym, orf, strand)
    out = {}
    for c in SITES:
        if c not in hi or c not in lo:
            continue
        sc = np.concatenate([hi[c], lo[c]])
        lab = np.concatenate([np.ones(len(hi[c])), np.zeros(len(lo[c]))])
        out[c] = (roc_auc_score(lab, sc), average_precision_score(lab, sc))
    return out


def build_bc(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = matplotlib.colormaps["tab20"].resampled(18)
    for ax, (name, mi, ylab) in zip(axes, [("AUROC", 0, "AUROC"), ("AUPRC", 1, "AUPRC")]):
        ci = 0
        for grp, genes in QUANTILES.items():
            for g in genes:
                if g not in metrics:
                    ci += 1; continue
                xs = [c for c in SITES if c in metrics[g]]
                ys = [metrics[g][c][mi] for c in xs]
                ax.plot(xs, ys, ls="--", lw=0.9, alpha=0.55, color=cmap(ci),
                        label=f"{g} ({'pos' if mc.GENE_STRAND[g]=='+' else 'neg'})")
                ci += 1
        for grp, genes in QUANTILES.items():
            gs = [g for g in genes if g in metrics]
            ys = np.array([[metrics[g][c][mi] for c in SITES if c in metrics[g]] for g in gs])
            if len(ys) == 0:
                continue
            mean = ys.mean(axis=0)
            se = ys.std(axis=0) / np.sqrt(len(ys))
            ax.errorbar(SITES[:len(mean)], mean, yerr=se, color=AGG_COLOR[grp], lw=2.2,
                        marker="o", ms=4, capsize=2, label=f"{grp} Aggregate")
        ax.set_xlabel("Insertion Position (nt upstream)", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f"Rafi et al. High vs Low Expression Sequences\n{name} trend for three gene expression quantiles", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=5.5, ncol=2, loc="lower right")
    fig.tight_layout()
    out = REPRO / "Figure_6BC.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def build_6a():
    fig, ax = plt.subplots(figsize=(11, 3))
    # gene/TSS at right
    ax.add_patch(plt.Rectangle((330, 0.30), 70, 0.30, color="#1f5fb4"))
    ax.text(365, 0.45, ">  >  >", ha="center", va="center", fontsize=10, color="white")
    ax.text(420, 0.45, "TSS", ha="left", va="center", fontsize=10)
    # inserted-sequence bars (a couple, faded) + insertion triangles 100..200
    ax.add_patch(plt.Rectangle((40, 0.78), 260, 0.06, color="#86c172", alpha=0.7))
    ax.add_patch(plt.Rectangle((60, 0.70), 260, 0.06, color="#9aa0a6", alpha=0.7))
    ax.add_patch(plt.Rectangle((150, 0.62), 230, 0.06, color="#7aa6c2", alpha=0.7))
    ax.text(305, 0.90, "...", fontsize=12)
    cmap = matplotlib.colormaps["turbo"].resampled(11)
    for i, d in enumerate(range(200, 99, -10)):
        x = 330 - d
        ax.plot(x, 0.50, marker="^", ms=12, color=cmap(i), mec="black", mew=0.4)
        if d in (200, 180, 160, 140, 120, 100):
            ax.text(x, 0.36, f"{d}bp", ha="center", fontsize=8)
    # legend
    ax.plot(360, 0.95, marker="^", ms=10, color="#1f5fb4")
    ax.text(372, 0.95, "Inserted position", fontsize=8, va="center")
    ax.add_patch(plt.Rectangle((352, 0.84), 16, 0.05, color="#7aa6c2"))
    ax.text(372, 0.865, "Inserted sequence", fontsize=8, va="center")
    ax.set_xlim(110, 470); ax.set_ylim(0.25, 1.02); ax.axis("off")
    ax.set_title("Figure 6A (reproduced) — MPRA insertion schematic (100-200 bp upstream, 10-bp steps)", fontsize=10)
    fig.tight_layout()
    out = REPRO / "Figure_6A_schematic.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    metrics = {g: gene_site_metrics(g) for grp in QUANTILES.values() for g in grp}
    metrics = {g: m for g, m in metrics.items() if m}
    all_auroc = [metrics[g][c][0] for g in metrics for c in metrics[g]]
    all_auprc = [metrics[g][c][1] for g in metrics for c in metrics[g]]
    mean_auroc, mean_auprc = float(np.mean(all_auroc)), float(np.mean(all_auprc))
    min_auroc, min_auprc = float(np.min(all_auroc)), float(np.min(all_auprc))
    bc = build_bc(metrics)
    a = build_6a()
    print(f"[6B/6C] genes={len(metrics)}  mean AUROC={mean_auroc:.4f} (min {min_auroc:.3f})  "
          f"mean AUPRC={mean_auprc:.4f} (min {min_auprc:.3f})  -> {bc.name}")
    print(f"[6A] schematic -> {a.name}")
    with open(RECHECK / "fig6_BC.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "min", "threshold"])
        w.writerow(["AUROC", f"{mean_auroc:.4f}", f"{min_auroc:.4f}", 0.95])
        w.writerow(["AUPRC", f"{mean_auprc:.4f}", f"{min_auprc:.4f}", 0.95])


if __name__ == "__main__":
    main()
