#!/usr/bin/env python3
"""Figure 6 panels B/C (AUROC/AUPRC of high- vs low-expression sequences across
insertion sites).

For each of the 18 reporter genes (6 per expression quantile: 5-25 / 25-75 / 75-95),
Shorkie's per-sequence logSED classifies high- vs low-expression library sequences at
each insertion site (100-200 bp). AUROC (6B) and AUPRC (6C) per site are plotted as
per-gene dashed lines with 'o' markers plus the three quantile aggregates (mean +/- STD,
fmt='o-'), faithful to 3_MPRA_classifier_merge.py::plot_combined_trend_quantiles. The
quantile aggregates stay > 0.95 across all sites (individual genes dip at 100 bp).

The panel figsize matches the published B/C content aspect ratio (~1.93, i.e. 8 x 8/1.93);
bbox_inches="tight" is intentionally NOT used so the saved PNG aspect ratio equals the
figsize ratio deterministically (tight_layout fits the title/legend within the canvas).

Outputs reproduced/Figure_6B.png, reproduced/Figure_6C.png, and recheck/fig6_BC.csv
(mean AUROC/AUPRC).
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
# Exact aggregate colors from 3_MPRA_classifier_merge.py::plot_combined_trend_quantiles.
AGG_COLOR = {"5-25": "#006400", "25-75": "#8B0000", "75-95": "#000000"}


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


def build_metric(metrics, mi, name, panel, letter):
    """One published panel (B=AUROC, C=AUPRC), faithful to
    3_MPRA_classifier_merge.py::plot_combined_trend_quantiles: per-gene dashed lines
    with 'o' markers (single tab20 over 18 genes), and three quantile aggregates with
    STD error bars (fmt='o-', ms=8, capsize=5) in dark-green/dark-red/black."""
    cmap = matplotlib.colormaps["tab20"].resampled(18)
    # figsize aspect 8 / (8/1.93) = 1.93 matches the published B/C panel content ratio.
    fig, ax = plt.subplots(figsize=(8, 8 / 1.93))
    handles, labels = [], []
    ci = 0
    for grp, genes in QUANTILES.items():
        # per-gene dashed + marker lines
        for g in genes:
            if g not in metrics:
                ci += 1
                continue
            xs = [c for c in SITES if c in metrics[g]]
            ys = [metrics[g][c][mi] for c in xs]
            line, = ax.plot(xs, ys, linestyle="--", marker="o", alpha=0.5, color=cmap(ci),
                            label=f"{g} ({'pos' if mc.GENE_STRAND[g]=='+' else 'neg'})")
            handles.append(line); labels.append(f"{g} ({'pos' if mc.GENE_STRAND[g]=='+' else 'neg'})")
            ci += 1
        # quantile aggregate: mean +/- STD
        gs = [g for g in genes if g in metrics]
        ys = np.array([[metrics[g][c][mi] for c in SITES if c in metrics[g]] for g in gs])
        if len(ys) == 0:
            continue
        mean = ys.mean(axis=0)
        std = ys.std(axis=0)
        cont = ax.errorbar(SITES[:len(mean)], mean, yerr=std, fmt="o-", color=AGG_COLOR[grp],
                           markersize=8, capsize=5, label=f"{grp} Aggregate")
        handles.append(cont.lines[0]); labels.append(f"{grp} Aggregate")
    ax.set_xlabel("Insertion Position (nt upstream)")
    ax.set_ylabel(name)
    ax.set_title(f"Rafi et al. High vs Low Expression Sequences\n{name} trend for three gene expression quantiles")
    ax.grid(True)
    ax.legend(handles, labels, loc="best", fontsize=8, ncol=3)
    ax.annotate(letter, xy=(-0.07, 1.12), xycoords="axes fraction",
                fontsize=20, fontweight="bold", va="top", ha="left")
    fig.tight_layout()
    out = REPRO / f"Figure_{panel}.png"
    # No bbox_inches="tight": save the full fixed canvas so the PNG aspect ratio == figsize ratio (1.93).
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def build_bc(metrics):
    b = build_metric(metrics, 0, "AUROC", "6B", "B")
    c = build_metric(metrics, 1, "AUPRC", "6C", "C")
    return b, c


def main():
    metrics = {g: gene_site_metrics(g) for grp in QUANTILES.values() for g in grp}
    metrics = {g: m for g, m in metrics.items() if m}
    all_auroc = [metrics[g][c][0] for g in metrics for c in metrics[g]]
    all_auprc = [metrics[g][c][1] for g in metrics for c in metrics[g]]
    mean_auroc, mean_auprc = float(np.mean(all_auroc)), float(np.mean(all_auprc))
    min_auroc, min_auprc = float(np.min(all_auroc)), float(np.min(all_auprc))
    b, c = build_bc(metrics)
    print(f"[6B/6C] genes={len(metrics)}  mean AUROC={mean_auroc:.4f} (min {min_auroc:.3f})  "
          f"mean AUPRC={mean_auprc:.4f} (min {min_auprc:.3f})  -> {b.name}, {c.name}")
    with open(RECHECK / "fig6_BC.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "min", "threshold"])
        w.writerow(["AUROC", f"{mean_auroc:.4f}", f"{min_auroc:.4f}", 0.95])
        w.writerow(["AUPRC", f"{mean_auprc:.4f}", f"{min_auprc:.4f}", 0.95])


if __name__ == "__main__":
    main()
