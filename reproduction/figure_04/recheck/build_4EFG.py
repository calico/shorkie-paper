#!/usr/bin/env python3
"""Figure 4 panels E/F/G (recheck) — Shorkie splicing ISM for DTD1 / MMS2 / HOP2.

Each panel: a publication-quality Shorkie-ISM letter logo over the gene window
(logSED, T0-averaged, mean-centred, projected on the reference one-hot) above a
strand-aware gene model, with dashed annotation boxes at the Start Codon, 5' splice
donor, Branch point, 3' splice acceptor and Stop Codon — coordinates derived from
the R64 GTF exon/intron boundaries.

Output: reproduced/Figure_4EFG_reproduced.png  +  recheck/fig4EFG_metrics.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F


def main():
    rows = []
    n = len(F.SPLICE)
    fig = plt.figure(figsize=(16, 3.1 * n))
    gs = fig.add_gridspec(n * 2, 1, height_ratios=[3.0, 1.0] * n, hspace=0.05)
    for k, spec in enumerate(F.SPLICE):
        gene, part, panel = spec["gene"], spec["part"], spec["panel"]
        res = F.ism_saliency("motif_shorkie_RP_TSS", "gene_exp_motif_test_SS", part, 0)
        ax_logo = fig.add_subplot(gs[2 * k])
        ax_gene = fig.add_subplot(gs[2 * k + 1], sharex=ax_logo)
        if res is None:
            ax_logo.text(0.5, 0.5, f"4{panel} {gene}: scores.h5 not found", ha="center")
            ax_logo.axis("off"); ax_gene.axis("off"); continue
        v, chrom, start, end = res
        loc = F.localization(v)
        ovl = bool(F.gene_features(gene) and not (end < F.gene_features(gene)["start"] - 600
                                                  or start > F.gene_features(gene)["end"] + 100))
        ymin, ymax = F.draw_logo(ax_logo, v, x0=start)
        ax_logo.set_xlim(start, end)
        ax_logo.set_title(f"{gene} ({chrom}:{start:,}-{end:,})  —  Shorkie ISM   "
                          f"[localization {loc:.1f}×]", fontsize=10, loc="left")
        ax_logo.set_yticks([])
        # gene model
        F.draw_gene_model(ax_gene, gene, start, end)
        ax_gene.set_xlabel(f"genomic position on {chrom}", fontsize=8)
        # dashed annotation lines/boxes at each splice/codon feature, labels staggered
        feats = [(lab, pos) for lab, pos in F.splice_annotations(gene) if start <= pos <= end]
        feats.sort(key=lambda t: t[1])
        span = ymax - ymin
        for fi, (label, pos) in enumerate(feats):
            for ax in (ax_logo, ax_gene):
                lo, hi = ax.get_ylim()
                ax.add_patch(Rectangle((pos - 5, lo), 10, hi - lo,
                                       fill=False, ec="red", ls="--", lw=1.0, zorder=5))
            ytext = ymax + span * (0.10 + 0.22 * (fi % 3))   # stagger over 3 levels
            ax_logo.annotate(label, xy=(pos, ymax), xytext=(pos, ytext),
                             ha="center", va="bottom", fontsize=6.5, color="red",
                             arrowprops=dict(arrowstyle="-", color="red", lw=0.5))
        ax_logo.set_ylim(ymin, ymax + span * 0.85)
        rows.append(dict(panel=f"4{panel}", gene=gene, chrom=chrom, start=start, end=end,
                         max_abs_saliency=round(float(np.abs(v).max()), 4),
                         localization=round(loc, 2), window_overlaps_gene=int(ovl)))
    fig.suptitle("Figure 4E–G (reproduced) — Shorkie splicing ISM + gene model + splice annotations",
                 y=0.995, fontsize=12)
    out = F.RD / "Figure_4EFG_reproduced.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(F.RECHECK / "fig4EFG_metrics.csv", index=False)
    print("saved", out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
