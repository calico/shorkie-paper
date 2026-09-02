#!/usr/bin/env python3
"""Revision experiment 02, step 4 — render the per-motif ISM recovery contrast
.

Two panels, both built so that motifs where Shorkie_Random_Init does BETTER are as visible
as the ones where it does worse -- the published claim averaged those away.

  A  dumbbell plot: for each motif named in the paper, the median recovery score of each
     model, joined by a line. Reading across shows which model attributes more, and how
     far apart they are on a common (per-window standardised) scale.
  B  contrast forest: the paired Shorkie - Shorkie_Random_Init difference with its
     bootstrap interval, ordered by effect. Motifs significant after Holm correction are
     filled; the rest are open, so "not resolvable" is visually distinct from "no
     difference".

Writes ``results/motif_recovery.png``. CPU only, seconds.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shorkie import config

COL_SHORKIE = "#377eb8"
COL_RANDOM = "#ff7f00"
LABELS = {
    "cons_RRPE": "RRPE consensus", "cons_PAC": "PAC consensus", "cons_TATA": "TATA box",
    "cons_donor": "5' splice donor", "cons_branch": "branch point",
    "SPT15": "TBP (Spt15)", "STB3": "RRPE (Stb3 PWM)", "DOT6": "PAC (Dot6 PWM)",
}
NAMED = ["cons_TATA", "cons_donor", "cons_branch", "cons_RRPE", "cons_PAC",
         "RAP1", "REB1", "SPT15", "CBF1", "ABF1", "MCM1", "FHL1", "SFP1",
         "UME6", "DOT6", "STB3", "TBF1", "MSN2", "MSN4", "SWI4", "RPN4"]

def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-motif ISM recovery contrasts.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--top_n", type=int, default=25,
                        help="Extra motifs to include in panel B, by |contrast|")
    return parser.parse_args()

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    csv = out_dir / "motif_recovery.csv"
    if not csv.exists():
        sys.exit(f"error: {csv} not found -- run 3_recovery_scores.py first")
    res = pd.read_csv(csv)
    res["label"] = res.motif.map(lambda m: LABELS.get(m, m))

    named = res[res.motif.isin(NAMED)].sort_values("median_contrast")
    extra = (res[~res.motif.isin(NAMED)]
.reindex(res.median_contrast.abs().sort_values(ascending=False).index)
.dropna(subset=["motif"]).head(args.top_n))
    panelB = pd.concat([named, extra]).drop_duplicates("motif").sort_values("median_contrast")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.5, max(6.0, 0.30 * len(panelB) + 2.2)),
        gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    # --- A: dumbbell of the two medians, motifs named in the paper --------------------
    y = np.arange(len(named))
    for yi, (_, r) in zip(y, named.iterrows()):
        axA.plot([r.median_Shorkie_Random_Init, r.median_Shorkie], [yi, yi],
                 color="0.75", lw=1.6, zorder=1)
    axA.scatter(named.median_Shorkie_Random_Init, y, s=48, color=COL_RANDOM, zorder=3,
                label="Shorkie_Random_Init", edgecolor="white", linewidth=0.6)
    axA.scatter(named.median_Shorkie, y, s=48, color=COL_SHORKIE, zorder=3,
                label="Shorkie", edgecolor="white", linewidth=0.6)
    axA.axvline(0, color="0.4", lw=0.9, ls="--")
    axA.set_yticks(y)
    axA.set_yticklabels([f"{r.label}  (n={int(r.n_sites)})" for _, r in named.iterrows()],
                        fontsize=8)
    axA.set_xlabel("median recovery score\n(standardised |saliency| in motif − in flanks)")
    axA.set_title("A · attribution placed on each motif", fontsize=11, loc="left")
    axA.grid(axis="x", alpha=0.3)
    axA.legend(fontsize=8, loc="lower right")

    # --- B: paired contrast with bootstrap interval ------------------------------------
    yb = np.arange(len(panelB))
    for yi, (_, r) in zip(yb, panelB.iterrows()):
        sig = bool(r.significant)
        colour = COL_SHORKIE if r.median_contrast > 0 else COL_RANDOM
        axB.plot([r.contrast_ci_lo, r.contrast_ci_hi], [yi, yi], color=colour,
                 lw=1.6, alpha=0.9 if sig else 0.5, zorder=2)
        axB.scatter([r.median_contrast], [yi], s=52 if sig else 34,
                    facecolor=colour if sig else "white", edgecolor=colour,
                    linewidth=1.4, zorder=3)
    axB.axvline(0, color="0.4", lw=0.9, ls="--")
    axB.set_yticks(yb)
    axB.set_yticklabels([r.label for _, r in panelB.iterrows()], fontsize=8)
    axB.set_xlabel("paired contrast: Shorkie − Shorkie_Random_Init\n"
                   "(median, 95% bootstrap CI; filled = significant after Holm)")
    axB.set_title("B · which model attributes more, and is the difference resolvable?",
                  fontsize=11, loc="left")
    axB.grid(axis="x", alpha=0.3)

    n_sig = int(res.significant.sum())
    fig.suptitle(
        f"ISM motif recovery, Shorkie vs Shorkie_Random_Init — {len(res)} motifs, "
        f"{n_sig} significant after Holm correction", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / "motif_recovery.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("saved", out_png, flush=True)

if __name__ == "__main__":
    main()
