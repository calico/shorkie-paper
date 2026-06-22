#!/usr/bin/env python3
"""Render Figure 3 panels H/I/J from reproduced coverage npz (run_coverage.py output),
matching the PUBLISHED styling:

  * three stacked filled coverage tracks per locus column — Experiment Ground Truth
    (observed bigwig, purple), **Fine-tuned model = Shorkie (orange)**, Scratch-trained
    = Random_Init (blue). (The previous version used red for Shorkie.)
  * a gene-annotation track above each column drawn from the GTF (deepskyblue gene
    bodies + strand chevrons + darkblue exon boxes + gene names), as in the published
    panels (HNM1/RPL7A; SUB2/RPS16B/RPL13A/RPP1A; ERG26/EFM5/SWC4).
  * dashed vertical lines at exon/intron boundaries spanning the coverage rows.
  * per-column shared y-scale, display cropped to the published window.

Per-locus Pearson R vs observed (over the full model output region) is recorded to
recheck/recheck_checks_coverage.csv (not printed on the figure — the published panels
carry none). Saves reproduced/Figure_3HIJ_coverage.png.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../reproduction
from common.compare import Check, write_verdicts  # noqa
from shorkie import config
config.load()
REPRO = config.repo_root() / "reproduction" / "figure_03"
COV = REPRO / "reproduced" / "coverage"
GTF = str(config.path("genome.gtf"))

LOCI = [
    ("rpl7a",         "chrVII:362,180-366,023 (RNA-Seq tracks); fold 3"),
    ("rps16b_rpl13a", "chrIV:305,657-310,505 (RNA-Seq tracks); fold 3"),
    ("efm5",          "chrVII:495,374-499,965 (RNA-Seq tracks); fold 6"),
]
ROWS = [("cov_obs",  "Experiment Ground Truth (Avg)", "tab:purple"),
        ("cov_self", "Fine-tuned model (Avg)",        "tab:orange"),
        ("cov_ri",   "Scratch-trained (Avg)",         "tab:blue")]
PANEL = {"rpl7a": "3H", "rps16b_rpl13a": "3I", "efm5": "3J"}


def parse_gtf_attrs(s):
    d = {}
    for part in str(s).strip(";").split("; "):
        if not part:
            continue
        k, _, v = part.partition(" ")
        d[k] = v.strip('"')
    return d


def genes_in(chrom, lo, hi):
    """Return (genes, exons) overlapping [lo,hi] on `chrom` (GTF uses bare Roman)."""
    seqid = chrom.replace("chr", "")
    cols = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attr"]
    g = pd.read_csv(GTF, sep="\t", comment="#", names=cols, usecols=range(9),
                    dtype={"seqid": str})
    g = g[(g.seqid == seqid) & (g.end >= lo) & (g.start <= hi) & g.type.isin(["gene", "exon"])]
    genes, exons = [], []
    for _, r in g.iterrows():
        a = parse_gtf_attrs(r["attr"])
        if r["type"] == "gene":
            genes.append((int(r["start"]), int(r["end"]), r["strand"],
                          a.get("gene_name", a.get("gene_id", "?"))))
        else:
            exons.append((int(r["start"]), int(r["end"]), a.get("gene_name", a.get("gene_id", ""))))
    return genes, exons


def draw_gene_track(ax, chrom, lo, hi):
    genes, exons = genes_in(chrom, lo, hi)
    span = hi - lo
    for gs, ge, strand, name in genes:
        x0, x1 = max(gs, lo), min(ge, hi)
        ax.add_patch(Rectangle((x0, 0.38), x1 - x0, 0.24, fc="deepskyblue",
                               ec="steelblue", lw=0.5, zorder=2))
        # strand chevrons along the gene body
        n_ch = max(1, int((x1 - x0) / (span * 0.05)))
        for k in range(1, n_ch):
            cx = x0 + (x1 - x0) * k / n_ch
            dx = span * 0.006 * (1 if strand == "+" else -1)
            ax.plot([cx - dx, cx + dx, cx - dx], [0.44, 0.50, 0.56], color="white", lw=0.8, zorder=3)
        ax.text((x0 + x1) / 2, 0.78, name, ha="center", va="center", fontsize=8, zorder=4)
    for es, ee, _ in exons:  # exons as taller dark boxes (mRNA structure)
        x0, x1 = max(es, lo), min(ee, hi)
        if x1 > x0:
            ax.add_patch(Rectangle((x0, 0.30), x1 - x0, 0.40, fc="navy", ec="none", zorder=3))
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1.05)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return exons


def main():
    fig, axes = plt.subplots(4, 3, figsize=(16, 8),
                             gridspec_kw=dict(height_ratios=[0.55, 1, 1, 1]),
                             sharex="col")
    checks = []
    for j, (name, title) in enumerate(LOCI):
        d = np.load(COV / f"{name}.npz", allow_pickle=True)
        stride = int(d["stride"]); start = int(d["seq_out_start"])
        chrom = str(d["chrom"]); win = [int(v) for v in d["win"]]
        n = len(d["cov_obs"]); x = start + np.arange(n) * stride
        obs = d["cov_obs"]
        lo, hi = win  # crop display to the published window
        m = (x >= lo) & (x <= hi)
        ymax = max(float(np.nanmax(d[k][m])) for k, _, _ in ROWS) * 1.08

        # gene-annotation track (row 0)
        exons = draw_gene_track(axes[0, j], chrom, lo, hi)
        axes[0, j].set_title(title, fontsize=11, pad=4)
        # intron/exon boundary dashed lines
        bnds = sorted({b for es, ee, _ in exons for b in (es, ee) if lo <= b <= hi})

        for i, (key, lbl, col) in enumerate(ROWS):
            ax = axes[i + 1, j]
            ax.fill_between(x[m], 0, d[key][m], color=col, alpha=0.85, linewidth=0)
            ax.set_ylim(0, ymax); ax.set_xlim(lo, hi)
            for b in bnds:
                ax.axvline(b, color="0.4", ls="--", lw=0.6, alpha=0.7, zorder=0)
            if j == 0:
                ax.set_ylabel(lbl, fontsize=8)
            if i + 1 == 3:
                ax.set_xlabel(f"{chrom} position (bp)", fontsize=9)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            # quantitative check vs observed (full output region, not just the crop)
            if key != "cov_obs":
                r = float(np.corrcoef(d[key], obs)[0, 1])
                model = "Shorkie" if key == "cov_self" else "Random_Init"
                checks.append(Check(PANEL[name], f"coverage_Pearson_R[{model},obs](>0.5)",
                                    0.5, round(r, 4), mode="ge"))

    handles = [Line2D([0], [0], color=c, lw=8, alpha=0.85) for _, _, c in ROWS]
    fig.legend(handles, [lbl for _, lbl, _ in ROWS], loc="upper center", ncol=3,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = REPRO / "reproduced" / "Figure_3HIJ_coverage.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("saved", out)
    write_verdicts(checks, REPRO / "recheck" / "recheck_checks_coverage.csv")
    for c in checks:
        print(f"  {c.panel} {c.metric} repro={c.reproduced} {c.verdict}")


if __name__ == "__main__":
    main()
