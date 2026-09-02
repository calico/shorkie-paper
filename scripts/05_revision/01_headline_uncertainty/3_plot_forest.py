#!/usr/bin/env python3
"""Revision experiment 01, step 3 — render the fold-level uncertainty for the paper
.

Two figures from ``results/headline_ci.csv``:

  * ``headline_forest.png``  -- per panel statistic and track type, the point estimate
    with its 95% cross-fold interval, for Shorkie and both random-init baselines. Makes
    the separation (and where it narrows) readable at a glance.
  * ``headline_paired_folds.png`` -- the eight paired per-fold values for the RNA-seq
    bin-level statistic, drawn as connected lines, with the paired mean difference and
    its Wilcoxon p annotated. This is the figure that shows the improvement holds in
    every individual fold rather than only on average.

Colours follow the published Figure 3 palette (Shorkie #377eb8, random-init #ff7f00);
the un-tuned baseline gets a lighter tint of the same orange so the two baselines read
as a family.

CPU only, seconds.
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

COLORS = {
    "Shorkie": "#377eb8",
    "Shorkie_Random_Init": "#ff7f00",
    "Shorkie_Random_Init_untuned": "#fdbf6f",
}
LABELS = {
    "Shorkie": "Shorkie",
    "Shorkie_Random_Init": "Shorkie_Random_Init (lr 5e-4, Fig 3C)",
    "Shorkie_Random_Init_untuned": "Shorkie_Random_Init (lr 1e-4, text '0.67')",
}
MODEL_ORDER = list(COLORS)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot cross-fold CIs for the headline metrics.")
    parser.add_argument("--in_csv", default=None, help="headline_ci.csv from step 2")
    parser.add_argument("--fold_csv", default=None, help="paired_fold_deltas.csv from step 2")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def plot_forest(df, out_png):
    rows = df[df.model.isin(MODEL_ORDER)].copy()
    rows["key"] = rows["panel"] + "  " + rows["track_type"]
    keys = list(dict.fromkeys(rows["key"]))

    fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(keys) + 2.2))
    offsets = {m: o for m, o in zip(MODEL_ORDER, (0.24, 0.0, -0.24))}
    for m in MODEL_ORDER:
        sub = rows[rows.model == m]
        ys, xs, lo, hi = [], [], [], []
        for _, r in sub.iterrows():
            ys.append(keys.index(r["key"]) + offsets[m])
            xs.append(r["point"]); lo.append(r["fold_ci95_lo"]); hi.append(r["fold_ci95_hi"])
        if not ys:
            continue
        xs, lo, hi = np.array(xs), np.array(lo), np.array(hi)
        ax.errorbar(xs, ys, xerr=[xs - lo, hi - xs], fmt="o", ms=5, lw=1.6, capsize=3,
                    color=COLORS[m], label=LABELS[m])

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson's R  (point estimate, 95% CI over the 8 test folds)")
    ax.set_title("Figure 3 headline metrics with cross-fold confidence intervals", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("saved", out_png, flush=True)

def plot_paired(fold_df, ci_df, out_png, panel="3C", track_type="RNA-Seq"):
    sub = fold_df[(fold_df.panel == panel) & (fold_df.track_type == track_type)]
    piv = sub.pivot_table(index="fold", columns="model", values="value")
    present = [m for m in MODEL_ORDER if m in piv.columns]
    if len(present) < 2:
        print("SKIPPED: not enough models for the paired plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x = np.arange(len(present))
    for fold, row in piv.iterrows():
        ax.plot(x, [row[m] for m in present], "-", color="0.75", lw=1, zorder=1)
    for i, m in enumerate(present):
        ax.scatter([i] * len(piv), piv[m], s=42, color=COLORS[m], zorder=3,
                   edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m].replace(" (", "\n(") for m in present], fontsize=8)
    ax.set_ylabel(f"{panel} {track_type}: Pearson's R per fold")
    ax.set_title(f"Per-fold paired comparison (n=8 test folds)\n{panel} — {track_type}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    notes = []
    for baseline in present[1:]:
        hit = ci_df[(ci_df.panel == panel) & (ci_df.track_type == track_type)
                    & (ci_df.model == f"Shorkie_minus_{baseline}")]
        if len(hit):
            r = hit.iloc[0]
            p = r.get("wilcoxon_p", float("nan"))
            notes.append(f"Δ vs {baseline}: {r['point']:+.3f} "
                         f"[{r['fold_ci95_lo']:+.3f}, {r['fold_ci95_hi']:+.3f}], "
                         f"Wilcoxon p={p:.4f}")
    if notes:
        ax.text(0.5, -0.28, "\n".join(notes), transform=ax.transAxes, ha="center",
                va="top", fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_png, flush=True)

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ci_csv = Path(args.in_csv) if args.in_csv else out_dir / "headline_ci.csv"
    fold_csv = Path(args.fold_csv) if args.fold_csv else out_dir / "paired_fold_deltas.csv"
    for p in (ci_csv, fold_csv):
        if not p.exists():
            sys.exit(f"error: {p} not found -- run 2_bootstrap_ci.py first")

    ci_df = pd.read_csv(ci_csv)
    fold_df = pd.read_csv(fold_csv)
    plot_forest(ci_df, out_dir / "headline_forest.png")
    plot_paired(fold_df, ci_df, out_dir / "headline_paired_folds.png")

if __name__ == "__main__":
    main()
