#!/usr/bin/env python3
"""Figure 1 deep-recheck — close the 1G provenance gap by pixel-measuring the
published panel-G bar heights and confirming they are an affine function of the
reproduced gene/intergenic perplexities.

The manuscript text states only the *qualitative* 1G ordering ("165_Saccharo-
mycetales achieved the lowest test perplexity ... outperformed the 1341_Fungal
LM"); the exact numbers live only in the figure. We therefore tie the reproduced
numbers to the *published bars* directly:

  1. detect the 8 grouped bars (4 gene=blue, 4 intergenic=orange) in
     published/Figure_1_G_pub.png, taking each bar's top pixel row;
  2. linear-fit  top_row_pixel = a * perplexity + b  over all 8 bars;
  3. a near-perfect fit (R^2 ~ 1, residual << tick spacing) proves the published
     bar heights ARE the reproduced perplexities, up to the axis' affine scaling.

This is non-circular: 8 (value, pixel) pairs vs 2 fitted DOF; 6 residual DOF.

Run (env yeast_ml):
    python reproduction/figure_01/recheck/measure_panelG.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from shorkie import config

REPRO = Path(config.repo_root()) / "reproduction" / "figure_01"
PUB = REPRO / "published" / "Figure_1_G_pub.png"
RECHECK = REPRO / "recheck"

TIERS = ["R64_yeast", "80_Strains", "165_Saccharomycetales", "1342_Fungus"]
# reproduced (== on-disk .out) perplexities, in left-to-right tier order
GENE = {"R64_yeast": 3.7561, "80_Strains": 3.7342, "165_Saccharomycetales": 3.5488, "1342_Fungus": 3.6043}
INTER = {"R64_yeast": 3.7386, "80_Strains": 3.7225, "165_Saccharomycetales": 3.6360, "1342_Fungus": 3.6851}

TAB_BLUE = np.array([31, 119, 180])
TAB_ORANGE = np.array([255, 127, 14])


def _bar_top_from_baseline(col_mask):
    """Top row of the contiguous coloured run anchored at the column's lowest
    coloured pixel. This is the true bar top and ignores a disconnected legend
    swatch sitting higher up in the same column."""
    rows = np.flatnonzero(col_mask)
    if rows.size == 0:
        return None
    rowset = set(rows.tolist())
    top = bottom = rows[-1]
    while (top - 1) in rowset:   # walk up while contiguous from the baseline
        top -= 1
    return top


def detect_bars(arr, target_rgb, tol=60):
    """Return [(center_x, top_row)] for each bar of the given colour, left->right.

    Each bar's top is measured as the contiguous coloured run anchored at the
    plot baseline, so the legend swatch (a separate higher run) is excluded.
    """
    H, W, _ = arr.shape
    dist = np.sqrt(((arr.astype(int) - target_rgb) ** 2).sum(axis=2))
    mask = dist < tol                      # HxW boolean
    # a real bar's colour extends below 70% of the image height (legend does not)
    reaches_base = np.array([mask[int(0.70 * H):, c].any() for c in range(W)])
    in_bar = mask.any(axis=0) & reaches_base
    bars = []
    c = 0
    while c < W:
        if in_bar[c]:
            c0 = c
            while c < W and in_bar[c]:
                c += 1
            c1 = c
            if c1 - c0 >= max(3, int(0.005 * W)):   # ignore sliver noise
                tops = [_bar_top_from_baseline(mask[:, cc]) for cc in range(c0, c1)]
                tops = [t for t in tops if t is not None]
                bars.append((0.5 * (c0 + c1), float(np.median(tops))))
        else:
            c += 1
    return bars


def main():
    if not PUB.exists():
        sys.exit(f"missing published crop {PUB} (run extract_panels crop first)")
    arr = np.asarray(Image.open(PUB).convert("RGB"))
    blue = detect_bars(arr, TAB_BLUE)
    orange = detect_bars(arr, TAB_ORANGE)
    print(f"detected {len(blue)} gene(blue) bars, {len(orange)} intergenic(orange) bars")
    if len(blue) != 4 or len(orange) != 4:
        print("  blue:", [(round(x), round(t)) for x, t in blue])
        print("  orange:", [(round(x), round(t)) for x, t in orange])
        sys.exit("expected exactly 4+4 bars; adjust crop/tolerance")

    blue.sort()
    orange.sort()
    vals, tops, labels = [], [], []
    for (x, top), t in zip(blue, TIERS):
        vals.append(GENE[t]); tops.append(top); labels.append(f"gene[{t}]")
    for (x, top), t in zip(orange, TIERS):
        vals.append(INTER[t]); tops.append(top); labels.append(f"inter[{t}]")
    vals = np.array(vals)
    tops = np.array(tops)

    # linear fit  top_pixel = a*value + b
    a, b = np.polyfit(vals, tops, 1)
    pred = a * vals + b
    ss_res = float(((tops - pred) ** 2).sum())
    ss_tot = float(((tops - tops.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    resid_ppl = np.abs(tops - pred) / abs(a)     # residual expressed in perplexity units

    print(f"\nlinear fit  top_pixel = {a:.2f} * perplexity + {b:.1f}")
    print(f"R^2 = {r2:.6f}   max residual = {resid_ppl.max():.4f} perplexity units "
          f"(tick spacing 0.05)\n")
    print(f"  {'bar':<26} {'perplexity':>10} {'top_px':>8} {'pred_px':>8} {'resid_ppl':>10}")
    for lab, v, tp, pr, rp in zip(labels, vals, tops, pred, resid_ppl):
        print(f"  {lab:<26} {v:>10.4f} {tp:>8.0f} {pr:>8.0f} {rp:>10.4f}")

    ok = (r2 > 0.99) and (resid_ppl.max() < 0.02)
    verdict = "PASS" if ok else "REVIEW"
    print(f"\n[{verdict}] published 1G bar heights are an affine map of the reproduced "
          f"perplexities (R^2={r2:.4f}, max resid {resid_ppl.max():.4f} ppl).")

    out = RECHECK / "panelG_barheight_fit.csv"
    with open(out, "w") as f:
        f.write("bar,perplexity,top_pixel,pred_pixel,resid_perplexity\n")
        for lab, v, tp, pr, rp in zip(labels, vals, tops, pred, resid_ppl):
            f.write(f"{lab},{v:.4f},{tp:.0f},{pr:.1f},{rp:.4f}\n")
        f.write(f"# fit a={a:.4f} b={b:.4f} R2={r2:.6f} max_resid_ppl={resid_ppl.max():.4f} verdict={verdict}\n")
    print(f"[OK] -> {out}")
    return r2, resid_ppl.max()


if __name__ == "__main__":
    main()
