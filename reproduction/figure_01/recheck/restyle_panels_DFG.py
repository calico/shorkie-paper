#!/usr/bin/env python3
"""Figure 1 deep-recheck — restyle panels 1D, 1F & 1G to match the published render.

The reproduced numbers are already exact (12/12 Δ=0). This script regenerates the
*figures* so the side-by-side against the published panels is visually faithful in
both **style** and **panel proportions** (the user asked D/F/G to match the published
crops exactly). All three are pure renders of byte-identical cached data — no GPU,
no recompute.

  1D  Mash distance, two stacked panels matching published/Figure_1_D_pub.png
      (1329x1659 px, portrait ~0.80 w/h):
        * top    = 165_Saccharomycetales, linear y 0..1
        * bottom = 80_Strains with a BROKEN y-axis (lower 0.000-0.010 holds every
          bar; upper 0.990-1.000 is empty — it just shows the full Mash range),
          diagonal break marks, an arrow to the R64-1-1 (Mash distance 0) bar, and
          curly-brace "All other ... genomes" annotations under each panel.
        Bars: skyblue, gapped (width 0.75). Published-exact titles. No numeric
        x-ticks (a "Target Genomes" label + brace replace them).
      NOTE: the released scripts/.../mash/2_mash_genome_viz.py emits only a plain
      skyblue bar chart; the published broken-axis/brace styling was a manual
      enhancement, faithfully reproduced here (the underlying distances are unchanged
      and byte-identical on re-run).

  1F  validation-loss curves: dashed lines per corpus (blue/orange/green/red), raw
      per-epoch loss, faint vertical min-marker at the argmin batch, legend
      "<label>; loss = <min>", x = epoch * 64 (extends to ~320k for 165_Sacc).
      figsize widened to the published F aspect (3012x1340 ~ 2.25:1).

  1G  region-specific perplexity: gene (blue) / intergenic (orange) grouped bars,
      zoomed y-axis, title "Region-specific Perplexity (Gene vs Intergenic)".
      figsize widened to the published G aspect (2611x1181 ~ 2.21:1).

Published display labels differ from the internal tier keys (kingdom tier is shown
as "1341_Fungus"; the F legend capitalises "R64_Yeast"). Those are applied here.

Writes the canonical reproduced PNGs (so the committed figure matches) AND a copy
under recheck/ for the composites.

Run (env yeast_ml):
    python reproduction/figure_01/recheck/restyle_panels_DFG.py
"""
from __future__ import annotations
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

from shorkie import config

REPRO = Path(config.repo_root()) / "reproduction" / "figure_01"
RD = REPRO / "reproduced"
RECHECK = REPRO / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)
LMR = Path(config.path("lm_experiment_root")) / "test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI"

TIERS = ["R64_yeast", "80_Strains", "165_Saccharomycetales", "1342_Fungus"]
COL = {"R64_yeast": "tab:blue", "80_Strains": "tab:orange",
       "165_Saccharomycetales": "tab:green", "1342_Fungus": "tab:red"}
# published display labels
LABEL_F = {"R64_yeast": "R64_Yeast", "80_Strains": "80_Strains",
           "165_Saccharomycetales": "165_Saccharomycetales", "1342_Fungus": "1341_Fungus"}
LABEL_G = {"R64_yeast": "R64_yeast", "80_Strains": "80_Strains",
           "165_Saccharomycetales": "165_Saccharomycetales", "1342_Fungus": "1341_Fungus"}

SUBS = {
    "R64_yeast": "lm_r64_gtf/lm_r64_gtf_unet_small",
    "80_Strains": "lm_strains_gtf/lm_strains_gtf_unet_small",
    "165_Saccharomycetales": "LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small",
    "1342_Fungus": "LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small",
}
PSUBS = dict(SUBS)
PSUBS["165_Saccharomycetales"] += "_bert_drop"
PSUBS["1342_Fungus"] += "_bert_drop"

VLRE = re.compile(r"valid_loss:\s*([0-9.]+)")
BATCHES_PER_EPOCH = 64

MASH = RD / "panelD_mash"
BAR_BLUE = "skyblue"


# --------------------------------------------------------------------------- #
# Panel D helpers — curly brace + broken-axis bars
# --------------------------------------------------------------------------- #
def _draw_brace_down(ax, x0, x1, ytop, depth, text, fontsize=10, lw=1.4, color="black"):
    """A downward-opening curly brace spanning x0..x1 (data x) drawn *below* the
    axes, with `text` centred underneath. y is in axes-fraction (blended transform:
    data x, axes-fraction y), so negative ytop sits below the plotting area.

    Standard 4-cubic-Bezier curly brace (two outer humps + central downward spike)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    xm = 0.5 * (x0 + x1)
    yb = ytop - depth          # tip / outer-hump bottom
    ym = ytop - depth * 0.5
    xq1, xq3 = 0.5 * (x0 + xm), 0.5 * (xm + x1)
    Path = mpath.Path
    verts = [
        (x0, ytop),
        (x0, ym), (xq1, ytop), (xq1, ym),       # left outer hump
        (xq1, yb), (xm, ym), (xm, yb),          # left inner -> central spike
        (xm, ym), (xq3, yb), (xq3, ym),         # right inner
        (xq3, ytop), (x1, ym), (x1, ytop),      # right outer hump
    ]
    codes = [Path.MOVETO] + [Path.CURVE4] * 12
    ax.add_patch(PathPatch(Path(verts, codes), transform=trans, fill=False,
                           lw=lw, color=color, clip_on=False))
    ax.text(xm, yb - 0.045, text, transform=trans, ha="center", va="top",
            fontsize=fontsize, clip_on=False)


def _annotate_mash_panel(ax, n, brace_text):
    """Shared below-axis annotations for a Mash subplot (same on both panels in the
    published figure): a steel-blue arrow to the index-0 (R64-1-1, Mash distance 0)
    bar, a curly brace over the other genomes, and a centred 'Target Genomes' label.
    Drawn in a blended transform (data x, axes-fraction y) so it works regardless of
    the panel's y-data scale (Sacc 0-1 vs the strains' broken 0-0.01)."""
    blend = blended_transform_factory(ax.transData, ax.transAxes)
    # arrow to the index-0 (R64-1-1, distance 0) bar, label at the lower-left
    ax.annotate("$S.\\ Cerevisiae$; R64-1-1\n(with Mash Distance 0)",
                xy=(0, 0.04), xycoords=blend,
                xytext=(1.5, -0.52), textcoords=blend,
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.4),
                fontsize=8, ha="left", va="top", style="italic", clip_on=False)
    # curly brace over the other genomes, just below the axis
    _draw_brace_down(ax, 1, n - 1, -0.13, 0.14, brace_text, fontsize=9.5)
    # "Target Genomes" centred below the brace caption
    ax.text(0.5 * (n - 1), -0.46, "Target Genomes", transform=blend,
            ha="center", va="top", fontsize=12, clip_on=False)


def restyle_1D():
    sacc = pd.read_csv(MASH / "saccharomycetales_dist.tab", sep="\t", header=None,
                       names=["ref", "q", "dist", "p", "sh"])["dist"].sort_values().values
    strn = pd.read_csv(MASH / "strains_dist.tab", sep="\t", header=None,
                       names=["ref", "q", "dist", "p", "sh"])["dist"].sort_values().values

    # Wide-and-short plot areas to match the published panel (top Sacc plot ~4.5:1
    # w:h). Explicit spacer rows (hspace=0) give the title/brace/arrow/"Target
    # Genomes" annotations fixed room that does NOT shrink with the short axes.
    fig = plt.figure(figsize=(8.6, 8.0))
    gs = fig.add_gridspec(6, 1, height_ratios=[1.5,    # 0 ax_sacc plot
                                               1.8,    # 1 spacer (Sacc annotations + strains title)
                                               0.40,   # 2 ax_top  (strains 0.990-1.000)
                                               0.05,   # 3 break gap
                                               1.5,    # 4 ax_bot  (strains 0.000-0.010)
                                               1.20],  # 5 spacer (strains annotations)
                          hspace=0.0)
    ax_sacc = fig.add_subplot(gs[0])
    ax_top = fig.add_subplot(gs[2])   # strains break upper (0.990-1.000)
    ax_bot = fig.add_subplot(gs[4])   # strains break lower (0.000-0.010)

    # --- top panel: 165_Saccharomycetales, linear 0..1 ---------------------- #
    ax_sacc.bar(np.arange(len(sacc)), sacc, width=0.75, color=BAR_BLUE)
    ax_sacc.set_ylim(0, 1.0)
    ax_sacc.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_sacc.set_ylabel("Mash Distance Score", fontsize=10)
    ax_sacc.set_title("Mash Distance Score for Yeast R64 vs. Target 165_Saccharomycetales Genomes",
                      fontsize=11.5)
    ax_sacc.set_xlim(-1, len(sacc))
    ax_sacc.set_xticks([])
    for sp in ("top", "right"):
        ax_sacc.spines[sp].set_visible(False)
    _annotate_mash_panel(ax_sacc, len(sacc),
                         "All other Saccharomycetales genomes (non–$S.\\ cerevisiae$)")

    # --- bottom panel: 80_Strains with a broken y-axis ---------------------- #
    x = np.arange(len(strn))
    for ax in (ax_top, ax_bot):
        ax.bar(x, strn, width=0.75, color=BAR_BLUE)
        ax.set_xlim(-1, len(strn))
    ax_top.set_ylim(0.990, 1.000)
    ax_top.set_yticks([0.990, 0.995, 1.000])
    ax_bot.set_ylim(0.000, 0.010)
    ax_bot.set_yticks([0.000, 0.005, 0.010])
    ax_top.set_title("Mash Distance Score for Yeast R64 vs. Target 80_Strains Genomes",
                     fontsize=11.5)
    # hide the spines facing the break + the top panel's x-ticks
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.set_xticks([])
    # shared y-label centred on the broken pair (sits at the top of ax_bot ~ the break)
    ax_bot.set_ylabel("Mash Distance Score", fontsize=10)
    ax_bot.yaxis.set_label_coords(-0.085, 0.85)

    # diagonal break marks (sized in axes-fraction; ax_top is short so scale them down)
    kw = dict(color="k", clip_on=False, lw=1.2)
    dxt, dyt = 0.010, 0.06    # ax_top is short -> larger fractional y
    dxb, dyb = 0.010, 0.022
    ax_top.plot((-dxt, +dxt), (-dyt, +dyt), transform=ax_top.transAxes, **kw)
    ax_top.plot((1 - dxt, 1 + dxt), (-dyt, +dyt), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-dxb, +dxb), (1 - dyb, 1 + dyb), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1 - dxb, 1 + dxb), (1 - dyb, 1 + dyb), transform=ax_bot.transAxes, **kw)

    _annotate_mash_panel(ax_bot, len(strn),
                         "All other $S.\\ cerevisiae$ strain genomes (non–R64-1-1)")

    for out in (RD / "Figure_1D_reproduced.png", RECHECK / "Figure_1D_restyled.png"):
        fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[1D] sacc n={len(sacc)} max={sacc.max():.4f} (>0.99: {(sacc > 0.99).sum()}); "
          f"strains n={len(strn)} max={strn.max():.5f}")


# Smoothing used by the original Figure-1F generator
# (scripts/03_eval/lm/lm_model_eval/3_dataset_comparison.py): a symmetric moving
# average with adaptive edges (moving_average L142-157) + a tail trim of the last
# `trim_end` points (maybe_trim_end L160-166). Defaults window_size=21, trim_end=5.
SMOOTH_WINDOW = 21
TRIM_END = 5


def _moving_average(x, window_size=SMOOTH_WINDOW):
    """Symmetric moving average with adaptive (truncated) edges — verbatim port of the
    legacy plot_losses smoothing so 1F matches the original published curves."""
    x = np.asarray(x, dtype=float)
    if window_size <= 1 or x.size == 0:
        return x.copy()
    out = np.empty_like(x)
    half = window_size // 2
    n = len(x)
    for j in range(n):
        lo, hi = max(0, j - half), min(n - 1, j + half)
        out[j] = x[lo:hi + 1].mean()
    return out


def _trim_end(x, trim_end=TRIM_END):
    if trim_end <= 0 or x.size == 0:
        return x
    return x[:1] if trim_end >= len(x) else x[:-trim_end]


def restyle_1F():
    fig, ax = plt.subplots(figsize=(10.0, 4.45))
    mins = {}
    for t in TIERS:
        vls = np.array([float(VLRE.search(ln).group(1)) for ln in open(LMR / SUBS[t] / "train" / "train.out")
                        if ln.strip().startswith("Epoch") and VLRE.search(ln)])
        # Same plot as the legacy 3_dataset_comparison.py: plot the SMOOTHED curve
        # (moving_average(21) -> trim last 5) but take the legend min from the raw
        # (trimmed) data, so the printed min is unchanged (verify stays 12/12).
        va_orig = _trim_end(vls)
        va_smooth = _trim_end(_moving_average(vls))
        x = np.arange(1, len(va_smooth) + 1) * BATCHES_PER_EPOCH
        mins[t] = float(va_orig.min())
        ax.plot(x, va_smooth, ls="--", color=COL[t], lw=1.2, alpha=0.9,
                label=f"{LABEL_F[t]}; loss = {va_orig.min():.4f}", zorder=-100)
        # faint vertical min-marker at the (raw) argmin batch (legacy: alpha 0.25, lw 1)
        ax.axvline((int(va_orig.argmin()) + 1) * BATCHES_PER_EPOCH,
                   color=COL[t], ls="--", lw=1.0, alpha=0.25, zorder=-200)
    ax.set_xlabel("# Training Batches")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Losses")
    # y starts low enough to show the FULL green descent (165_Sacc min 0.4018) — the
    # previous 0.404 floor clipped it; 0.02 ticks (0.40/0.42/0.44). x to ~320k (the
    # smoothed+trimmed green reaches ~319.7k), 50k ticks; legend centred at the top.
    ax.set_ylim(0.400, 0.446)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_xlim(0, 322000)
    ax.set_xticks(np.arange(0, 300001, 50000))
    ax.legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    for out in (RD / "Figure_1F_reproduced.png", RECHECK / "Figure_1F_restyled.png"):
        fig.savefig(out, dpi=140)
    plt.close(fig)
    print("[1F] min valid loss:", {LABEL_F[t]: round(mins[t], 4) for t in TIERS})
    return mins


def restyle_1G():
    gene, inter = {}, {}
    for t in TIERS:
        f = LMR / PSUBS[t] / "test_testset_perplexity_region" / "test_testset_perplexity_region.out"
        rows, inreg = [], False
        for ln in open(f):
            s = ln.strip()
            if s.startswith("Region-specific"):
                inreg = True
                continue
            elif inreg and s:
                rows.append(s)
        body = [r for r in rows if re.match(r"^\d+\s+\w+", r)]
        df = pd.read_csv(StringIO("\n".join(rows[:1] + body)), sep=r"\s+")
        gene[t] = float(df[df.region == "gene"]["perplexity"].iloc[0])
        inter[t] = float(df[df.region == "intergenic"]["perplexity"].iloc[0])
    x = np.arange(len(TIERS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.bar(x - w / 2, [gene[t] for t in TIERS], w, label="gene", color="tab:blue")
    ax.bar(x + w / 2, [inter[t] for t in TIERS], w, label="intergenic", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_G[t] for t in TIERS], fontsize=9)
    ax.set_ylim(3.52, 3.78)
    ax.set_ylabel("Perplexity")
    ax.set_title("Region-specific Perplexity (Gene vs Intergenic)")
    ax.legend()
    fig.tight_layout()
    for out in (RD / "Figure_1G_reproduced.png", RECHECK / "Figure_1G_restyled.png"):
        fig.savefig(out, dpi=140)
    plt.close(fig)
    print("[1G] gene:", {LABEL_G[t]: round(gene[t], 4) for t in TIERS})
    print("[1G] inter:", {LABEL_G[t]: round(inter[t], 4) for t in TIERS})
    return gene, inter


if __name__ == "__main__":
    restyle_1D()
    restyle_1F()
    restyle_1G()
    print("[OK] restyled 1D/1F/1G -> reproduced/ + recheck/")
