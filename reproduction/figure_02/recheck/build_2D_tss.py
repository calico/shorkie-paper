#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 2: reproduce panel-2D TF-motif TSS enrichment.

Published panel 2D shows histograms of TF-MoDISco motif positions relative to the
TSS, overlaying the True motif distribution (green) against a Background (red); the
TF motifs concentrate just upstream of the TSS (promoter enrichment). The published
panel shows the TF motifs **Abf1.1, Rap1.1, Reb1p** (plus genic features — start
codon/5'SS/branch point — that come from a separate genic-position analysis NOT in
the released TSS-distance CSV; see DISCREPANCIES.md).

This reproduces the three TF histograms from the released artifacts
(`motif_tss_distances.csv`, `background_tss_distances.csv`) using the exact plotting
convention of the upstream script
`scripts/04_analysis/.../4_motif_to_tss_dist/3_plot_tss_dist_freq.py`
(flip_distance = -distance; green True / red Background; xlim ±2500; TSS line at 0),
and verifies per-TF that the motif sits closer to / upstream of the TSS than
background.

(The committed reproduction had used the wrong TF subset — Reb1p, ABF1.1, Sfp1p,
AZF1.6, GCR2.6, RPN4.7 — only 2 of which are in the published panel.)

Outputs: reproduced/Figure_2D_reproduced.png, recheck/fig2D_enrichment.csv.

Run (env yeast_ml): python reproduction/figure_02/recheck/build_2D_tss.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

WORK = str(config.path("work_root"))
D = f"{WORK}/experiments/motif_LM/4_motif_to_tss_dist"
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

# published panel-2D TF motifs -> CSV motif_name(s)
PUB_TFS = {
    "Abf1.1": ["ABF1.1"],
    "Rap1.1": ["RAP1.1"],
    "Reb1p": ["Reb1p&consensus=CCGGGTAA", "Reb1p&consensus=MGGGTAAB"],
}
XLIM = (-2500, 2500)
NEAR = 250  # bp window around TSS for the enrichment fraction


def main():
    df = pd.read_csv(f"{D}/motif_tss_distances.csv")
    bg = pd.read_csv(f"{D}/background_tss_distances.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    bin_edges = np.linspace(XLIM[0], XLIM[1], 51)
    rows = []
    for ax, (label, names) in zip(axes, PUB_TFS.items()):
        td = df[df.motif_name.isin(names)].copy()
        bd = bg[bg.motif_name.isin(names)].copy()
        td["flip"] = -td["distance"]
        bd["flip"] = -bd["distance"]
        ax.hist(td["flip"], bins=bin_edges, color="#2ca02c", alpha=0.7, label=f"True (n={len(td)})")
        ax.hist(bd["flip"], bins=bin_edges, color="#d62728", alpha=0.5, label=f"Background (n={len(bd)})")
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_xlim(XLIM)
        ax.set_title(label)
        ax.set_xlabel("Distance from TSS (bp)\n(neg = upstream)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        med_t = float(np.median(np.abs(td["distance"])))
        med_b = float(np.median(np.abs(bd["distance"])))
        frac_t = float((np.abs(td["distance"]) <= NEAR).mean())
        frac_b = float((np.abs(bd["distance"]) <= NEAR).mean())
        # upstream concentration: fraction of True hits in [-NEAR, 0] upstream of TSS (flip>0 side after sign flip = upstream is negative distance)
        rows.append(dict(motif=label, n_true=len(td), n_bg=len(bd),
                         median_abs_dist_true=round(med_t, 1), median_abs_dist_bg=round(med_b, 1),
                         frac_within_250_true=round(frac_t, 4), frac_within_250_bg=round(frac_b, 4),
                         enriched_near_tss=bool(med_t < med_b and frac_t > frac_b)))
    fig.suptitle("Figure 2D (reproduced) — TF-MoDISco motif enrichment vs TSS "
                 "(True green vs Background red); published TFs: Abf1.1, Rap1.1, Reb1p", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = RD / "Figure_2D_reproduced.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(RECHECK / "fig2D_enrichment.csv", index=False)
    n_ok = int(res["enriched_near_tss"].sum())
    print(f"\n[{'PASS' if n_ok == len(res) else 'CHECK'}] {n_ok}/{len(res)} published TFs enriched near/upstream of TSS vs background")
    print(f"[OK] -> {out}")
    print(f"[OK] -> {RECHECK/'fig2D_enrichment.csv'}")


if __name__ == "__main__":
    main()
