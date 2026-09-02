#!/usr/bin/env python3
"""Revision experiment 05, step 1 — redraw Figure 6B/6C so the series are
distinguishable.

The numbers are NOT touched. The AUROC/AUPRC computation is imported unchanged from
``reproduction/figure_06/recheck/build_panels_BC.py`` via its ``mpra_common`` loaders, and
this script asserts the recomputed means against the committed ``fig6_BC.csv``. Only the
visual encoding changes:

  * the three quantile aggregates move off dark-green / dark-red / black -- which differ
    little in hue AND barely at all in lightness -- onto Paul Tol's high-contrast triple
    (#004488 / #BB5566 / #DDAA33), chosen so the three stay separable under deuteranopia,
    protanopia and tritanopia and in greyscale;
  * marker shape and line style become redundant encodings of the same grouping, so the
    series are readable even if colour is lost entirely;
  * the 18 per-gene lines stop drawing from a resampled ``tab20`` (which repeats hues once
    stretched from 10 to 18) and are instead tinted by the quantile they belong to, with
    marker shape separating genes within a quantile. This also makes the panel's actual
    structure -- three groups of six -- visible, which the original palette hid.

The improvement is measured, not asserted: ``results/palette_accessibility.csv`` reports
pairwise CIE76 colour distance and lightness separation for the published and revised
palettes under normal vision and three simulated colour-vision deficiencies, using the
Machado et al. (2009) severity-1.0 matrices.

Outputs ``results/Figure_6B_revised.png``, ``results/Figure_6C_revised.png``,
``results/palette_accessibility.csv``, ``results/verify_revision_05_fig6bc.csv``.

CPU only, ~1 min (dominated by loading the MPRA logSED NPZ tree).
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shorkie import config

config.load()
_F6 = Path(config.repo_root()) / "reproduction" / "figure_06" / "recheck"
sys.path.insert(0, str(_F6))
sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
import mpra_common as mc                      # noqa: E402
from build_panels_BC import (                 # noqa: E402  -- reuse, do not re-derive
    SITES, QUANTILES, AGG_COLOR as PUBLISHED_AGG_COLOR, gene_site_metrics,
)
from compare import Check, write_verdicts, summary  # noqa: E402

# Paul Tol high-contrast qualitative scheme: distinct in hue AND lightness, and designed
# to stay distinguishable under the common colour-vision deficiencies.
REVISED_AGG_COLOR = {"5-25": "#004488", "25-75": "#BB5566", "75-95": "#DDAA33"}
AGG_MARKER = {"5-25": "o", "25-75": "s", "75-95": "^"}
AGG_LINESTYLE = {"5-25": "-", "25-75": "--", "75-95": "-."}
# Redundant marker encoding for the six genes inside each quantile.
GENE_MARKERS = ["o", "s", "^", "D", "v", "P"]

# Machado, Oliveira & Fernandes (2009) severity-1.0 CVD simulation matrices (linear RGB).
CVD_MATRICES = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Redraw Figure 6B/6C with an accessible palette.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

# --------------------------------------------------------------------------- colour --
def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_xyz(rgb):
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return m @ rgb

def _xyz_to_lab(xyz):
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

def to_lab(hex_color, cvd=None):
    rgb = _srgb_to_linear(matplotlib.colors.to_rgb(hex_color))
    if cvd is not None:
        rgb = np.clip(CVD_MATRICES[cvd] @ rgb, 0.0, 1.0)
    return _xyz_to_lab(_linear_to_xyz(rgb))

def palette_metrics(name, palette):
    """Pairwise CIE76 distance and lightness gap, under normal vision and each CVD."""
    rows = []
    for a, b in itertools.combinations(sorted(palette), 2):
        for vision in ("normal", "protanopia", "deuteranopia", "tritanopia"):
            cvd = None if vision == "normal" else vision
            la, lb = to_lab(palette[a], cvd), to_lab(palette[b], cvd)
            rows.append(dict(palette=name, pair=f"{a} vs {b}", vision=vision,
                             delta_E76=round(float(np.linalg.norm(la - lb)), 1),
                             delta_lightness=round(float(abs(la[0] - lb[0])), 1)))
    return rows

# --------------------------------------------------------------------------- panels --
def build_panel(metrics, mi, name, letter, out_png):
    fig, ax = plt.subplots(figsize=(8, 8 / 1.93))
    handles, labels = [], []
    for grp, genes in QUANTILES.items():
        base = REVISED_AGG_COLOR[grp]
        tint = matplotlib.colors.to_rgb(base) + (0.42,)      # same hue, low alpha
        for gi, g in enumerate(genes):
            if g not in metrics:
                continue
            xs = [c for c in SITES if c in metrics[g]]
            ys = [metrics[g][c][mi] for c in xs]
            lab = f"{g} ({'pos' if mc.GENE_STRAND[g] == '+' else 'neg'})"
            line, = ax.plot(xs, ys, linestyle=":", linewidth=1.0,
                            marker=GENE_MARKERS[gi % len(GENE_MARKERS)], markersize=4,
                            color=tint, label=lab)
            handles.append(line); labels.append(lab)

        gs = [g for g in genes if g in metrics]
        if not gs:
            continue
        ys = np.array([[metrics[g][c][mi] for c in SITES if c in metrics[g]] for g in gs])
        mean, std = ys.mean(axis=0), ys.std(axis=0)
        cont = ax.errorbar(SITES[:len(mean)], mean, yerr=std, color=base,
                           marker=AGG_MARKER[grp], linestyle=AGG_LINESTYLE[grp],
                           linewidth=2.4, markersize=8, capsize=5, zorder=5,
                           markeredgecolor="white", markeredgewidth=0.8,
                           label=f"{grp} Aggregate")
        handles.append(cont.lines[0]); labels.append(f"{grp} Aggregate")

    ax.set_xlabel("Insertion Position (nt upstream)")
    ax.set_ylabel(name)
    ax.set_title("Rafi et al. High vs Low Expression Sequences\n"
                 f"{name} trend for three gene expression quantiles")
    ax.grid(True, alpha=0.35)
    ax.legend(handles, labels, loc="best", fontsize=7, ncol=3)
    ax.annotate(letter, xy=(-0.07, 1.12), xycoords="axes fraction",
                fontsize=20, fontweight="bold", va="top", ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("saved", out_png, flush=True)

def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {g: gene_site_metrics(g) for grp in QUANTILES.values() for g in grp}
    metrics = {g: m for g, m in metrics.items() if m}
    if not metrics:
        sys.exit("error: no MPRA logSED NPZ found -- check results.mpra_viz / work_root")

    all_auroc = [metrics[g][c][0] for g in metrics for c in metrics[g]]
    all_auprc = [metrics[g][c][1] for g in metrics for c in metrics[g]]
    mean_auroc, mean_auprc = float(np.mean(all_auroc)), float(np.mean(all_auprc))
    print(f"genes={len(metrics)}  mean AUROC={mean_auroc:.4f}  mean AUPRC={mean_auprc:.4f}",
          flush=True)

    build_panel(metrics, 0, "AUROC", "B", out_dir / "Figure_6B_revised.png")
    build_panel(metrics, 1, "AUPRC", "C", out_dir / "Figure_6C_revised.png")

    rows = palette_metrics("published (dark green / dark red / black)", PUBLISHED_AGG_COLOR)
    rows += palette_metrics("revised (Tol high-contrast)", REVISED_AGG_COLOR)
    acc = pd.DataFrame(rows)
    acc.to_csv(out_dir / "palette_accessibility.csv", index=False)
    print("\nWorst-case separation across all pairs and vision types:")
    print(acc.groupby("palette")[["delta_E76", "delta_lightness"]].min().to_string())

    # The redraw must not move any number: check against the committed fig6_BC.csv.
    published = {r["metric"]: float(r["mean"]) for r in
                 csv.DictReader(open(_F6 / "fig6_BC.csv"))}
    checks = [
        Check(panel="6B", metric="mean AUROC (unchanged by the redraw)",
              reported=published.get("AUROC"), reproduced=round(mean_auroc, 4), rtol=0.001),
        Check(panel="6C", metric="mean AUPRC (unchanged by the redraw)",
              reported=published.get("AUPRC"), reproduced=round(mean_auprc, 4), rtol=0.001),
    ]
    write_verdicts(checks, out_dir / "verify_revision_05_fig6bc.csv")
    print()
    print(summary(checks))

if __name__ == "__main__":
    main()
