#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 3 (visual refinement): panel-2D TSS-enrichment
histograms in the published style.

Published 2D = six histograms of motif-position vs TSS, True (green) overlaid on a
Background (salmon), TSS dashed line at 0, xlim +/-2500, titled
"Distance Distribution for <motif> (True: n=.., Background: n=..)" — verbatim style
of `4_motif_to_tss_dist/3_plot_tss_dist_freq.py`. The published panels are the TF
motifs **Abf1.1, Rap1.1, Reb1p** plus the genic features **start codon (ATG),
5' splice site (donor), branch point**.

The 3 TF panels are reproduced EXACTLY from the released TF artifacts
(`motif_tss_distances.csv` / `background_tss_distances.csv`); their hit counts match
the published panel (Abf1.1 n=745, Rap1.1 n=644, Reb1p[CCGGGTAA] n=821).

The 3 genic features come from a separate modisco-pattern-indexed analysis
(`0_motif_genomic_region_ratio/motif_tss_distance/.../motif_<k>_tss_stats.txt`),
where the authors manually labelled which modisco pattern is the 5'SS donor / branch
point / start codon. We auto-identify the 5'SS-donor and branch-point patterns by
consensus (GT[AG]AGT / TACTAAC) and render them too; the start codon is not a clean
modisco PWM and is documented rather than guessed.

Outputs reproduced/Figure_2D_reproduced.png + recheck/fig2D_enrichment.csv.
Run (env yeast_ml): python reproduction/figure_02/recheck/build_2D_tss.py
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

WORK = str(config.path("work_root"))
TFD = f"{WORK}/experiments/motif_LM/4_motif_to_tss_dist"
GRD = f"{WORK}/experiments/motif_LM/0_motif_genomic_region_ratio/motif_tss_distance/unet_small_bert_drop"
H5 = f"{WORK}/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5"
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
XLIM = (-2500, 2500)
BINS = np.linspace(XLIM[0], XLIM[1], 51)
GREEN, SALMON = "#2ca02c", "#d62728"
ALPH = list("ACGT")

# published-panel TF motifs -> CSV name
TF_PANELS = [("Abf1.1", "ABF1.1"), ("Rap1.1", "RAP1.1"), ("Reb1p", "Reb1p&consensus=CCGGGTAA")]
# genic-feature consensus to auto-identify among modisco patterns
GENIC = [("5' splice site (donor site)", "GTAAGT"), ("branch point", "TACTAAC")]


def hist_panel(ax, true_d, bg_d, title):
    true_f, bg_f = -np.asarray(true_d), -np.asarray(bg_d)   # flip: negative = upstream
    ax.hist(true_f, bins=BINS, color=GREEN, alpha=0.7, label=f"True")
    ax.hist(bg_f, bins=BINS, color=SALMON, alpha=0.5, label=f"Background")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlim(XLIM)
    ax.set_title(f"Distance Distribution for {title}\n(True: n={len(true_d)}, Background: n={len(bg_d)})", fontsize=9)
    ax.set_xlabel("Distance from TSS (bp)\nNegative = upstream, Positive = downstream", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)


def find_pattern_for_consensus(consensus):
    """Return modisco pattern name (e.g. 'pos_patterns_pattern_3') best matching a consensus."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    rc = "".join(comp[b] for b in reversed(consensus))
    idx = {b: i for i, b in enumerate(ALPH)}
    best = (None, 0.0)
    with h5py.File(H5, "r") as f:
        for pn in f["pos_patterns"].keys():
            ppm = np.asarray(f["pos_patterns"][pn]["sequence"][:], float)
            ppm = ppm / (ppm.sum(1, keepdims=True) + 1e-9)
            for cons in (consensus, rc):
                k = len(cons); v = [idx[b] for b in cons]
                for s in range(ppm.shape[0] - k + 1):
                    sc = float(np.mean([ppm[s + j, v[j]] for j in range(k)]))
                    if sc > best[1]:
                        best = (f"pos_patterns_pattern_{pn.split('_')[-1]}", sc)
    return best


def genic_distances(pattern_name):
    """Read distance (last col) from the motif_<k>_tss_stats file whose 4th col matches pattern_name."""
    for f in glob.glob(f"{GRD}/motif_*_tss_stats.txt"):
        if "_bg_" in f:
            continue
        with open(f) as fh:
            first = fh.readline().split("\t")
        if len(first) > 3 and first[3].replace("_fwd", "").replace("_rev", "") == pattern_name:
            d = pd.read_csv(f, sep="\t", header=None)
            bgf = f.replace("_tss_stats", "_bg_tss_stats")
            dbg = pd.read_csv(bgf, sep="\t", header=None) if os.path.exists(bgf) else None
            return d.iloc[:, -1].values, (dbg.iloc[:, -1].values if dbg is not None else None)
    return None, None


def main():
    df = pd.read_csv(f"{TFD}/motif_tss_distances.csv")
    bg = pd.read_csv(f"{TFD}/background_tss_distances.csv")

    panels = []   # (title, true_dist, bg_dist)
    rows = []
    for label, name in TF_PANELS:
        td = df[df.motif_name == name]["distance"].values
        bd = bg[bg.motif_name == name]["distance"].values
        panels.append((label, td, bd))
        med_t, med_b = float(np.median(np.abs(td))), float(np.median(np.abs(bd)))
        rows.append(dict(panel=label, n_true=len(td), n_bg=len(bd),
                         median_abs_dist_true=round(med_t, 1), median_abs_dist_bg=round(med_b, 1),
                         enriched_near_tss=bool(med_t < med_b)))

    # genic features (5'SS / branch): auto-identified by consensus and recorded for
    # documentation, but NOT plotted in the main panel — their published hit-set is a
    # manually-curated subset (5'SS n=603, branch n=779) whose exact filtering is not in
    # the released artifacts, so the raw modisco-hit distributions (n=3077 / n=12859) do
    # not match the published counts. The start codon (ATG) is not a clean modisco PWM.
    for label, cons in GENIC:
        pat, sc = find_pattern_for_consensus(cons)
        td, bd = (None, None)
        if pat is not None and sc >= 0.80:
            td, bd = genic_distances(pat)
        if td is not None and bd is not None:
            rows.append(dict(panel=f"{label} [genic; documented, not plotted]", n_true=len(td), n_bg=len(bd),
                             median_abs_dist_true=round(float(np.median(np.abs(td))), 1),
                             median_abs_dist_bg=round(float(np.median(np.abs(bd))), 1),
                             enriched_near_tss=None))
            print(f"[genic] {label}: pattern {pat} (score {sc:.2f}) n={len(td)} "
                  f"(published n differs -> manual curation; documented, not plotted)")
        else:
            print(f"[genic] {label}: no confident pattern (best {sc:.2f}) -> documented")

    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, td, bd) in zip(axes, panels):
        hist_panel(ax, td, bd, title)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Figure 2D (reproduced) — motif enrichment vs TSS (True green / Background salmon)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RD / "Figure_2D_reproduced.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(RECHECK / "fig2D_enrichment.csv", index=False)
    print(f"[OK] {n} panels -> {out}")


if __name__ == "__main__":
    main()
