#!/usr/bin/env python3
"""Figure 4 — clean uniform ISM-saliency grid (recheck).

A single, annotation-free view of *every* per-base saliency logo Figure 4 is built
from, rendered so that:

  1. **Uniform scale** — every logo occupies the *identical physical box* (same width
     and height in inches); regions of different length therefore have different
     bp-per-inch (the user's chosen reading of "same subplot scale").
  2. **Exact region match** — each logo's x-axis is the *published* panel window
     (fig4_common.PUB_WIN, read off published/Figure_4_full.png). Where the released
     ISM scores.h5 window is offset (FUN12, MMS2) only the covered intersection is
     drawn and the covered fraction is recorded.
  3. **All saliency rows that exist** — per gene we stack Shorkie LM + Shorkie ISM +
     Shorkie Random_Init ISM, but *only* for the sources whose precomputed data is on
     disk (data-driven). No Reference-DB, gene models, TF boxes, splice boxes, dividers
     or curated text — just the logos + a minimal row label and genomic coords.

Precomputed only: reads ISM scores.h5 / LM preds.npz (via the per-row caches) — **no
ISM re-run, no GPU, no model load**.

Resulting rows (released data): RPL26A -> LM+ISM+Random (3); FUN12 -> LM+ISM (2);
KRE33 -> LM+ISM (2); DTD1/MMS2/HOP2 -> ISM (1).  (The published FUN12/KRE33 Random and
MMS2 LM/Random rows came from data not in the released registry — documented residual.)

Output: reproduced/Figure_4_ISM_grid_reproduced.png + recheck/fig4_ism_grid_metrics.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F

# ---- identical physical box for every logo (inches) ---------------------------------
BOX_W, BOX_H = 11.0, 0.85
LEFT, RIGHT, TOP, BOT = 2.7, 0.35, 0.55, 0.6
GAP, GROUP_GAP = 0.30, 0.60            # between rows of a gene / between genes


def _row(gene, panel, source, sal):
    """sal = (v, chrom_or_contig, data_start, data_end). Slice to the published window."""
    chrom, ps, pe = F.PUB_WIN[gene]
    v, _, dstart, _ = sal
    v_sub, x0, cov = F.slice_to_window(v, dstart, ps, pe)
    return dict(gene=gene, panel=panel, source=source, chrom=chrom,
                pub_start=ps, pub_end=pe, v_sub=v_sub, x0=x0,
                covered_frac=round(cov, 4), localization=round(F.localization(v), 2))


def gather_rows():
    """All saliency rows whose precomputed data exists, in published panel order."""
    rows = []
    for spec in F.PROM:
        gene = spec["gene"]
        # Shorkie LM (preds.npz via per-row cache) — its window == the published window
        lm = F.lm_saliency(spec["lm_set"], spec["lm_row"])
        rows.append(_row(gene, spec["panel"], "Shorkie LM", lm))
        # Shorkie ISM (fine-tuned logSED)
        ism = F.ism_saliency("motif_shorkie_RP_TSS", spec["sub"], spec["part"], spec["idx"])
        if ism is not None:
            rows.append(_row(gene, spec["panel"], "Shorkie ISM", ism))
        # Shorkie Random_Init ISM (only the RP sub is released -> RPL26A)
        if spec["random"]:
            rnd = F.ism_saliency("motif_random_init_RP_TSS", "gene_exp_motif_test_RP",
                                 spec["part"], spec["idx"])
            if rnd is not None:
                rows.append(_row(gene, spec["panel"], "Shorkie Random_Init ISM", rnd))
    for spec in F.SPLICE:
        gene = spec["gene"]
        ism = F.ism_saliency("motif_shorkie_RP_TSS", "gene_exp_motif_test_SS", spec["part"], 0)
        if ism is not None:
            rows.append(_row(gene, spec["panel"], "Shorkie ISM", ism))
    return rows


def draw_row(ax, r, first_of_gene, last_of_gene):
    ps, pe = r["pub_start"], r["pub_end"]
    v = r["v_sub"]
    if v.shape[0] > 0:
        F.draw_logo(ax, v, x0=r["x0"])
    else:                                                  # no overlap (shouldn't happen here)
        ax.axhline(0, color="black", lw=0.8); ax.set_ylim(-1, 1)
        ax.text((ps + pe) / 2, 0, "no ISM data in window", ha="center", va="center",
                fontsize=7, color="gray")
    ax.set_xlim(ps, pe); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # row label (gene + source), just left of the box
    ax.text(-0.012, 0.5, f"{r['gene']}\n{r['source']}", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.5)
    # compact coord + coverage, top-left inside the box
    cov = r["covered_frac"]
    covnote = "" if cov >= 0.999 else f"   (covered {cov:.0%})"
    ax.text(0.004, 0.96, f"{r['chrom']}:{ps:,}-{pe:,}{covnote}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.3, color="#555555")
    # genomic x ticks only on the last row of each gene (rows share the gene's window)
    if last_of_gene:
        ticks = np.linspace(ps, pe, 4)
        ax.set_xticks(ticks); ax.set_xticklabels([f"{int(t):,}" for t in ticks], fontsize=6.5)
    else:
        ax.set_xticks([])


def main():
    rows = gather_rows()
    n = len(rows)
    # figure height = uniform boxes + per-row gaps + extra gap between genes
    extra = sum(GAP + (GROUP_GAP if rows[i]["gene"] != rows[i - 1]["gene"] else 0.0)
                for i in range(1, n))
    fig_w = LEFT + BOX_W + RIGHT
    fig_h = TOP + n * BOX_H + extra + BOT
    fig = plt.figure(figsize=(fig_w, fig_h))

    panel_marks = []           # (panel_letter, y_center_fraction) for the gene's first row
    y = fig_h - TOP
    for i, r in enumerate(rows):
        if i > 0:
            y -= GAP + (GROUP_GAP if r["gene"] != rows[i - 1]["gene"] else 0.0)
        y -= BOX_H
        ax = fig.add_axes([LEFT / fig_w, y / fig_h, BOX_W / fig_w, BOX_H / fig_h])
        first = (i == 0 or rows[i - 1]["gene"] != r["gene"])
        last = (i == n - 1 or rows[i + 1]["gene"] != r["gene"])
        draw_row(ax, r, first, last)
        if first:
            panel_marks.append((r["panel"], (y + BOX_H / 2) / fig_h))

    for panel, yc in panel_marks:
        fig.text(0.012, yc, panel, ha="left", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 1 - 0.18 / fig_h, "Figure 4 (reproduced) — all ISM saliency logos, "
             "uniform scale, region-matched (no annotations)", ha="center", va="top", fontsize=11)

    out = F.RD / "Figure_4_ISM_grid_reproduced.png"
    fig.savefig(out, dpi=150)                              # NO bbox_inches='tight' -> identical boxes
    plt.close(fig)

    meta = pd.DataFrame([{k: r[k] for k in
                          ("panel", "gene", "source", "chrom", "pub_start", "pub_end",
                           "covered_frac", "localization")} for r in rows])
    meta["box_w_in"] = BOX_W; meta["box_h_in"] = BOX_H    # record the uniform box size
    meta.to_csv(F.RECHECK / "fig4_ism_grid_metrics.csv", index=False)
    print("saved", out, f"| {n} uniform logos ({BOX_W}x{BOX_H} in each)")
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
