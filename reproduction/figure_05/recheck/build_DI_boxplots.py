#!/usr/bin/env python3
"""Figure 5 panels D (MSN2) and I (MSN4): normalized-Pearson's-R per-timepoint boxplots,
surfaced as top-level reproduced/Figure_5{D,I}_*.png.

These panels otherwise live only as the source-script output under
reproduced/eval_{TF}/<gene>/pearsonr_norm_by_timepoint_boxplot.png. This builder re-renders
them faithfully into the reproduced/ root (like the A/C/F/H panels), reading the per-track
metrics table eval.txt.

Faithful to the source `…/motif_shorkie__time_series/1_time_track_metrics_viz.py` (lines 190-225):
timepoint parsed from the 'description' column (-T\\d+); plt.boxplot of `pearsonr_norm` grouped
by timepoint (matplotlib default → orange medians, as published); per-timepoint `n=` annotation;
ylabel "Normalized Pearson's R"; title "Timepoint-Resolved Normalized Pearson's R for\\nMeasured
Genes in {TF} Induction Tracks"; dpi=300.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig05_lib import RD

PANELS = [("5D", "MSN2", "Figure_5D_MSN2_boxplot.png"),
          ("5I", "MSN4", "Figure_5I_MSN4_boxplot.png")]


def build(panel, tf, png_name):
    ev = pd.read_csv(RD / f"eval_{tf}" / "eval.txt", sep="\t")
    ev["timepoint"] = (ev["description"].str.extract(r"-T(\d+)", expand=False)
                       .pipe(pd.to_numeric, errors="coerce").astype("Int64"))
    data, labels = [], []
    for t in sorted(ev["timepoint"].dropna().unique()):
        vals = ev.loc[ev["timepoint"] == t, "pearsonr_norm"].dropna().values
        if len(vals):
            data.append(vals); labels.append(int(t))
    fig = plt.figure(); ax = plt.gca()
    plt.boxplot(data, labels=labels)
    ymin, ymax = ax.get_ylim()
    for i, vals in enumerate(data):
        ax.text(i + 1, ymin + 0.03 * (ymax - ymin), f"n={len(vals)}", ha="center", va="top", fontsize=8)
    plt.xlabel("Timepoint")
    plt.ylabel("Normalized Pearson's R")
    plt.title(f"Timepoint-Resolved Normalized Pearson's R for\nMeasured Genes in {tf} Induction Tracks")
    plt.tight_layout()
    out = RD / png_name
    fig.savefig(out, dpi=300); plt.close(fig)
    med = float(ev["pearsonr_norm"].dropna().median())
    print(f"[{panel} {tf}] median={med:.3f}  n-counts={[len(v) for v in data]}  -> {out}")


if __name__ == "__main__":
    for panel, tf, png in PANELS:
        build(panel, tf, png)
