#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 1: rebuild the panel-2C conservation grid.

Published 2C is a 6-dataset x 11-motif presence/absence grid showing that the
de-novo TF-MoDISco motifs recovered by Shorkie LM are conserved across fungal
evolution, with conservation declining beyond the Saccharomycetales order
(manuscript: "Motif conservation declined beyond the Saccharomycetales order";
"Mcm1.1 was absent in Schizosaccharomycetales, which lack a direct homolog").

We reconstruct the grid DATA-DRIVENLY from the same artifacts the authors used:
the per-tier `modisco report` TOMTOM match tables (`report_*/motifs.html`, columns
match0/1/2 + qval0/1/2 against the yeast motif DB). For each of the 9 TF motifs in
the published grid we take the best (min) TOMTOM q-value across all recovered
modisco patterns; a motif is "recovered" in a tier if min-qval < QTHRESH.

This reproduces the published CONSERVATION STRUCTURE (recovered-motif count
declines with evolutionary distance; Mcm1 lost in Schizosaccharomycetales; the
promoter TFs Rap1/Abf1/Dot6 lost beyond Saccharomycetales). It is NOT identical
to the published grid cell-for-cell, because the published grid was additionally
visually curated — TOMTOM matches some promiscuous motifs (e.g. Sfp1, a low-
complexity G-rich motif) confidently in every tier, and matches the low-complexity
TATA/TBP element only weakly. Those curation differences are reported, not hidden.

Outputs: recheck/fig2C_qval_grid.csv, recheck/fig2C_presence_grid.csv,
reproduced/Figure_2C_reproduced.png.

Run (env yeast_ml): python reproduction/figure_02/recheck/build_2C_grid.py
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

WORK = str(config.path("work_root"))
ML = f"{WORK}/experiments/motif_LM__unseen_species"
IND = f"{WORK}/experiments/motif_LM"
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

QTHRESH = 0.10  # TOMTOM q-value cutoff for "recovered"

# 6 datasets (published row order) -> modisco report
TIERS = [
    ("S. cerevisiae R64", f"{IND}/saccharomycetales_viz_seq/unet_small_bert_drop/report_w16384_n100000/motifs.html"),
    ("4 S. cerevisiae strains", f"{ML}/strains_select_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("5 Saccharomycetales", f"{ML}/saccharomycetales_select_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("4 Ascomycota", f"{ML}/ascomycota_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("4 Orbiliales", f"{ML}/orbiliales_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("4 Schizosaccharomycetales", f"{ML}/schizosaccharomycetales_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
]

# 9 TF motifs from the published grid -> yeast-DB name aliases (normalised, no trailing 'P')
TF_MOTIFS = {
    "Cbf1p": ["CBF1"], "Reb1.1": ["REB1"], "Snf1.1": ["SNF1"], "Mcm1.1": ["MCM1"],
    "Rap1.1": ["RAP1"], "Sfp1.2": ["SFP1"], "Abf1.1": ["ABF1"], "Dot6": ["DOT6"],
    "TBP": ["TBP", "SPT15", "TBP1"],
}
# the other 2 published columns (5' splice donor, branch point) are sequence motifs,
# not TF-DB entries; the published grid shows them conserved in all 6 tiers.

# published curated presence pattern (1=shown, 0="--"); for cross-check only
PUBLISHED = {
    "TBP":    [1, 1, 1, 1, 1, 1],
    "Cbf1p":  [1, 1, 1, 1, 1, 1],
    "Reb1.1": [1, 1, 1, 1, 1, 1],
    "Snf1.1": [1, 1, 1, 1, 1, 1],
    "Mcm1.1": [1, 1, 1, 1, 1, 0],
    "Rap1.1": [1, 1, 1, 0, 0, 0],
    "Sfp1.2": [1, 1, 1, 0, 0, 0],
    "Abf1.1": [1, 1, 1, 0, 0, 0],
    "Dot6":   [1, 1, 1, 0, 0, 0],
}


def norm_name(m):
    if not isinstance(m, str):
        return ""
    base = m.split("&")[0].split(".")[0].upper()
    return base.rstrip("P") if "&" in m else base


def min_qval(df, aliases):
    targets = set(a.rstrip("P") for a in aliases) | set(aliases)
    best = np.nan
    for mi in (0, 1, 2):
        mc, qc = f"match{mi}", f"qval{mi}"
        if mc not in df.columns:
            continue
        for _, r in df.iterrows():
            if norm_name(r[mc]) in targets:
                q = r[qc]
                if pd.notna(q) and (np.isnan(best) or q < best):
                    best = q
    return best


def main():
    motifs = list(TF_MOTIFS.keys())
    tier_names = [t[0] for t in TIERS]
    qgrid = pd.DataFrame(index=motifs, columns=tier_names, dtype=float)
    npat = {}
    for tname, path in TIERS:
        assert os.path.exists(path), f"missing report {path}"
        df = pd.read_html(path)[0]
        npat[tname] = len(df)
        for mname, aliases in TF_MOTIFS.items():
            qgrid.loc[mname, tname] = min_qval(df, aliases)

    presence = (qgrid < QTHRESH).astype(int)

    print("=" * 88)
    print("FIGURE 2C — data-driven TOMTOM conservation grid (min q-value per cell)")
    print(f"recovered if min-qval < {QTHRESH}.  npat/tier:", {k: npat[k] for k in tier_names})
    print("=" * 88)
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.2g}"):
        print(qgrid)
    print("\npresence (1=recovered):")
    print(presence)

    # recovered-count per tier -> conservation decline
    counts = presence.sum(axis=0)
    print("\nrecovered TF-motif count per tier:", dict(counts))

    # cross-check vs the published curated grid
    pub = pd.DataFrame(PUBLISHED, index=tier_names).T.reindex(motifs)
    agree = int((presence.values == pub.values).sum())
    total = presence.size
    print(f"\ncell agreement with published curation: {agree}/{total}")
    diffs = []
    for m in motifs:
        for j, t in enumerate(tier_names):
            if presence.loc[m, t] != pub.loc[m, t]:
                diffs.append((m, t, int(presence.loc[m, t]), int(pub.loc[m, t]),
                              float(qgrid.loc[m, t]) if pd.notna(qgrid.loc[m, t]) else None))
    if diffs:
        print("cells differing from published curation (motif, tier, data, published, qval):")
        for d in diffs:
            print("  ", d)

    # ---- verification claims ----
    claims = {}
    claims["TFs_recovered_in_R64(>=8/9)"] = int(presence[tier_names[0]].sum())
    claims["conservation_declines(Sacc>=Asco>=Schizo)"] = bool(
        counts[tier_names[2]] >= counts[tier_names[3]] >= counts[tier_names[5]]
        and counts[tier_names[2]] > counts[tier_names[5]])
    # Mcm1: confident through Orbiliales, absent in Schizosaccharomycetales
    claims["Mcm1_present_through_Orbiliales"] = bool(presence.loc["Mcm1.1", tier_names[4]] == 1)
    claims["Mcm1_absent_in_Schizosacc"] = bool(presence.loc["Mcm1.1", tier_names[5]] == 0)
    # promoter TFs (Rap1/Abf1/Dot6) present in first 3 tiers, absent beyond Saccharomycetales
    prom = ["Rap1.1", "Abf1.1", "Dot6"]
    claims["promoterTFs_present_in_Sacc_tiers"] = bool(
        all(presence.loc[m, tier_names[k]] == 1 for m in prom for k in (0, 1, 2)))
    claims["promoterTFs_lost_beyond_Sacc"] = bool(
        all(presence.loc[m, tier_names[k]] == 0 for m in prom for k in (3, 4, 5)))
    print("\nVERIFY claims:")
    for k, v in claims.items():
        print(f"  [{'PASS' if (v is True or (isinstance(v,int) and v>=8)) else 'CHECK'}] {k} = {v}")

    qgrid.to_csv(RECHECK / "fig2C_qval_grid.csv")
    presence.to_csv(RECHECK / "fig2C_presence_grid.csv")

    # ---- figure: -log10(qval) heatmap with presence ring ----
    M = -np.log10(qgrid.astype(float).clip(lower=1e-12))
    M = M.where(qgrid.notna(), other=np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(M.values, aspect="auto", cmap="viridis", vmin=0, vmax=8)
    ax.set_xticks(range(len(tier_names)))
    ax.set_xticklabels(tier_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(motifs)))
    ax.set_yticklabels(motifs, fontsize=10)
    for i, m in enumerate(motifs):
        for j, t in enumerate(tier_names):
            q = qgrid.loc[m, t]
            txt = "—" if pd.isna(q) or q >= QTHRESH else f"{q:.0e}"
            color = "white" if (pd.notna(q) and q < 1e-3) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Figure 2C (reproduced) — TF-MoDISco motif conservation across fungal tiers\n"
                 f"(TOMTOM min q-value; '—' = not recovered, q≥{QTHRESH})", fontsize=11)
    fig.colorbar(im, ax=ax, label="-log10(min q-value)", shrink=0.7)
    fig.tight_layout()
    out = RECHECK / "Figure_2C_qval_heatmap.png"   # heatmap kept here; the panel figure is the logo grid (build_2C_logos.py)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[OK] -> {out}")
    print(f"[OK] -> {RECHECK/'fig2C_qval_grid.csv'} , {RECHECK/'fig2C_presence_grid.csv'}")


if __name__ == "__main__":
    main()
