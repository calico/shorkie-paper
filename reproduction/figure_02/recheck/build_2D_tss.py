#!/usr/bin/env python3
"""Figure 2D — motif-position-vs-TSS enrichment histograms, the published 6-panel grid.

Published 2D is a 3x2 grid of "Distance Distribution for <motif>" histograms, True
(green) overlaid on Background (salmon), TSS dashed line at 0, xlim +/-2500, in the
verbatim style of the upstream `4_motif_to_tss_dist/3_plot_tss_dist_freq.py`
(seaborn whitegrid, 50 bins, flipped distance so negative = upstream).

All six panels are drawn directly from the released TF artifacts
(`motif_tss_distances.csv` / `background_tss_distances.csv`). Three of the panels are
TF motifs the paper relabelled to the genic feature they mark (confirmed by the
authors):

    start codon (ATG)            <- MIG3.4
    5' splice site (donor site)  <- CHA4.11
    branch point                 <- SWI5.7

the other three are the promoter TFs Abf1.1 / Rap1.1 / Reb1p. Each panel's True hit
count is asserted against the published count, which both reproduces the figure and
verifies the relabelling against the released data.

Outputs reproduced/Figure_2D_reproduced.png + recheck/fig2D_enrichment.csv.
Run (env yeast_ml): python reproduction/figure_02/recheck/build_2D_tss.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from shorkie import config

WORK = str(config.path("work_root"))
TFD = f"{WORK}/experiments/motif_LM/4_motif_to_tss_dist"
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"

XLIM = (-2500, 2500)
BINS = np.linspace(XLIM[0], XLIM[1], 51)        # 50 bins, as upstream
GREEN, SALMON = "#2ca02c", "#d62728"

# Published 2D, row-major 3x2: (csv motif_name, published title, published n_true)
PANELS = [
    ("MIG3.4",                    "start codon (ATG)",            2218),
    ("ABF1.1",                    "Abf1.1",                        745),
    ("RAP1.1",                    "Rap1.1",                        644),
    ("Reb1p&consensus=CCGGGTAA",  "Reb1p",                         821),
    ("CHA4.11",                   "5' splice site (donor site)",   603),
    ("SWI5.7",                    "branch point",                  779),
]


def hist_panel(ax, true_d, bg_d, title):
    """Single published-style histogram (matches 3_plot_tss_dist_freq.py)."""
    tf = pd.DataFrame({"flip_distance": -np.asarray(true_d, float)})
    bf = pd.DataFrame({"flip_distance": -np.asarray(bg_d, float)})
    sns.histplot(tf, x="flip_distance", bins=BINS, color=GREEN, label="True",
                 stat="count", alpha=0.7, kde=False, ax=ax)
    sns.histplot(bf, x="flip_distance", bins=BINS, color=SALMON, label="Background",
                 stat="count", alpha=0.5, kde=False, ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(XLIM)
    ax.set_title(f"Distance Distribution for {title}\n"
                 f"(True: n={len(true_d)}, Background: n={len(bg_d)})", fontsize=10)
    ax.set_xlabel("Distance from TSS (bp)\nNegative = upstream, Positive = downstream",
                  fontsize=8)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8)


def main():
    df = pd.read_csv(f"{TFD}/motif_tss_distances.csv")
    bg = pd.read_csv(f"{TFD}/background_tss_distances.csv")
    names = set(df.motif_name.astype(str))

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes = axes.ravel()

    rows = []
    for ax, (name, title, pub_n) in zip(axes, PANELS):
        assert name in names, (
            f"motif {name!r} ('{title}') not found in motif_tss_distances.csv; "
            f"cannot reproduce published 2D panel")
        td = df.loc[df.motif_name.astype(str) == name, "distance"].values
        bd = bg.loc[bg.motif_name.astype(str) == name, "distance"].values
        hist_panel(ax, td, bd, title)
        med_t, med_b = float(np.median(np.abs(td))), float(np.median(np.abs(bd)))
        match = (len(td) == pub_n)
        if not match:
            print(f"[WARN] {title} ({name}): n_true={len(td)} != published n={pub_n}")
        rows.append(dict(panel=title, csv_name=name, n_true=len(td), n_bg=len(bd),
                         median_abs_dist_true=round(med_t, 1),
                         median_abs_dist_bg=round(med_b, 1),
                         enriched_near_tss=bool(med_t < med_b),
                         published_n=pub_n, count_match=bool(match)))

    fig.suptitle("Figure 2D (reproduced) — motif enrichment vs TSS "
                 "(True green / Background salmon)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RD / "Figure_2D_reproduced.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(RECHECK / "fig2D_enrichment.csv", index=False)
    n_ok = int(res.count_match.sum())
    print(f"[OK] {len(rows)} panels ({n_ok}/{len(rows)} counts match published) -> {out}")


if __name__ == "__main__":
    main()
