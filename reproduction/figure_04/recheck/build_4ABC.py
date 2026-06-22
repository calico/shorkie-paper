#!/usr/bin/env python3
"""Figure 4 panels A/B/C (recheck) — promoter ISM saliency, 4 rows per gene.

For RPL26A / FUN12 / KRE33 over the published TSS-anchored 500 bp window (450 up + 50
down), four stacked DNA-letter-logo rows aligned on a shared genomic x-axis:
  1. Shorkie LM             — LM masked-prediction IC logo (preds.npz, 2_modisco_DNA_logo recipe)
  2. Shorkie ISM            — fine-tuned logSED (scores.h5), T0-avg, mean-centred, ref-projected
  3. Shorkie_Random_Init ISM— scratch logSED (RPL26A/RP only; B/C have no Random_Init tree)
  4. Reference DB           — database motif logos placed at TomTom-matched LM-MoDISco seqlet hits
plus dashed TF boxes (Fhl1/Rap1/...) on rows 1-2, the 450/50 TSS divider, and a
strand-aware gene model at the bottom.

Output: reproduced/Figure_4ABC_reproduced.png  +  recheck/fig4ABC_metrics.csv
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
import match_tfs as M

ROWS = ["Shorkie LM", "Shorkie ISM", "Shorkie_Random_Init ISM", "Reference DB"]


def main():
    rows_meta = []
    for spec in F.PROM:
        gene, panel = spec["gene"], spec["panel"]
        # display window = published TSS-anchored LM window
        v_lm, chrom, rs, re_ = F.lm_saliency(spec["lm_set"], spec["lm_row"])
        bed = F._lm_bed(spec["lm_set"]).iloc[spec["lm_row"]]
        bstart, strand = int(bed["start"]), bed["strand"]
        tss = (bstart + int(bed["end"])) // 2

        # row data
        ism = F.ism_saliency("motif_shorkie_RP_TSS", spec["sub"], spec["part"], spec["idx"])
        rnd = (F.ism_saliency("motif_random_init_RP_TSS", "gene_exp_motif_test_RP",
                              spec["part"], spec["idx"]) if spec["random"] else None)
        # Reference DB = FIMO-like scan of the panel's published TFs in the window
        hits = F.scan_db_motifs(chrom, rs, re_, tf_names=spec["ref_tfs"],
                                max_hits=len(spec["ref_tfs"]), force=True)

        fig = plt.figure(figsize=(16, 7.0))
        gs = fig.add_gridspec(5, 1, height_ratios=[1.4, 1.4, 1.4, 1.4, 1.0], hspace=0.18)
        axes = [fig.add_subplot(gs[i]) for i in range(5)]
        for ax in axes:
            ax.set_xlim(rs, re_); ax.set_yticks([])

        # 1) Shorkie LM
        ax = axes[0]; ymL = F.draw_logo(ax, v_lm, x0=rs); ax.set_xlim(rs, re_)
        loc_lm = F.localization(v_lm)
        # 2) Shorkie ISM
        ax = axes[1]
        loc_ism = float("nan"); ovl = 0
        if ism is not None:
            v, c2, istart, iend = ism
            F.draw_logo(ax, v, x0=istart); ax.set_xlim(rs, re_)
            loc_ism = F.localization(v)
            ovl = int(not (iend < F.gene_features(gene)["start"] - 600 or istart > F.gene_features(gene)["end"] + 100))
        else:
            ax.text((rs+re_)/2, 0, "ISM scores.h5 not found", ha="center")
        # 3) Random_Init ISM
        ax = axes[2]; loc_rnd = float("nan")
        if rnd is not None:
            vr, _, rst, _ = rnd; F.draw_logo(ax, vr, x0=rst); ax.set_xlim(rs, re_)
            loc_rnd = F.localization(vr)
        else:
            ax.text((rs + re_) / 2, 0.0, "no Random_Init tree (RP sub only)",
                    ha="center", va="center", fontsize=10, color="gray"); ax.set_ylim(-1, 1)
        # 4) Reference DB — DB motif logos at hit positions
        ax = axes[3]
        ymax_db = 2.2
        for h in hits:
            dbarr = F.db_motif_ic(h["tf"])
            if dbarr is None:
                continue
            w = dbarr.shape[0]; cx = (h["gstart"] + h["gend"]) / 2.0
            F.draw_logo(ax, dbarr, x0=cx - w / 2.0, ymin=0, ymax=ymax_db)
        ax.set_xlim(rs, re_); ax.set_ylim(0, ymax_db)
        if not hits:
            ax.text((rs + re_) / 2, ymax_db/2, "no TomTom-matched DB motif hit in window",
                    ha="center", va="center", fontsize=9, color="gray")

        # TF dashed boxes on LM + ISM rows; labels above (curated published TF names)
        for h in hits:
            for ax in (axes[0], axes[1]):
                lo, hi = ax.get_ylim()
                ax.add_patch(Rectangle((h["gstart"], lo), h["gend"] - h["gstart"], hi - lo,
                                       fill=False, ec="red", ls="--", lw=1.0, zorder=5))
            axes[0].annotate(F.TF_DISPLAY.get(h["tf"], h["tf"]),
                             xy=((h["gstart"] + h["gend"]) / 2, axes[0].get_ylim()[1]),
                             ha="center", va="bottom", fontsize=8, color="red")
        # gene-feature annotations (Start Codon / 5' splice donor) within the window
        for label, pos in F.splice_annotations(gene):
            if not (rs <= pos <= re_) or label.startswith(("Branch", "3'", "Stop")):
                continue
            for ax in (axes[1], axes[4]):
                lo, hi = ax.get_ylim()
                ax.add_patch(Rectangle((pos - 4, lo), 8, hi - lo, fill=False,
                                       ec="purple", ls=":", lw=1.0, zorder=6))
            axes[1].annotate(label, xy=(pos, axes[1].get_ylim()[0]),
                             ha="center", va="top", fontsize=6.5, color="purple")

        # 450/50 TSS divider on all logo rows
        for ax in axes[:4]:
            ax.axvline(tss, color="black", ls=":", lw=0.8)
        # gene model
        F.draw_gene_model(axes[4], gene, rs, re_)
        axes[4].set_xlabel(f"genomic position on {chrom}", fontsize=8)

        # row labels + scale annotations
        for i, lbl in enumerate(ROWS):
            axes[i].set_ylabel(lbl, fontsize=8, rotation=0, ha="right", va="center")
        axes[0].annotate("450 nt", xy=((rs+tss)/2, axes[0].get_ylim()[1]), ha="center",
                         va="bottom", fontsize=8, color="navy")
        axes[0].annotate("50 nt", xy=((tss+re_)/2, axes[0].get_ylim()[1]), ha="center",
                         va="bottom", fontsize=8, color="navy")
        fig.suptitle(f"Figure 4{panel} (reproduced) — {gene} ({chrom}:{rs:,}-{re_:,})  promoter ISM",
                     y=0.995, fontsize=12)
        out = F.RD / f"Figure_4{panel}_reproduced.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("saved", out, f"| hits={len(hits)} TFs:", sorted({h['tf'] for h in hits}))
        rows_meta.append(dict(panel=f"4{panel}", gene=gene, chrom=chrom, win_start=rs, win_end=re_,
                              loc_LM=round(loc_lm, 2), loc_ISM=round(loc_ism, 2),
                              loc_Random=round(loc_rnd, 2) if loc_rnd == loc_rnd else "n/a",
                              window_overlaps_gene=ovl, n_DB_hits=len(hits),
                              DB_TFs=";".join(sorted({h["tf"] for h in hits}))))

    # combined ABC image (stack the three per-panel PNGs)
    _combine([F.RD / f"Figure_4{p['panel']}_reproduced.png" for p in F.PROM],
             F.RD / "Figure_4ABC_reproduced.png")
    df = pd.DataFrame(rows_meta); df.to_csv(F.RECHECK / "fig4ABC_metrics.csv", index=False)
    print(df.to_string(index=False))


def _combine(pngs, out):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in pngs if p.exists()]
    if not ims:
        return
    w = max(i.width for i in ims); H = sum(i.height for i in ims)
    canvas = Image.new("RGB", (w, H), "white"); y = 0
    for im in ims:
        canvas.paste(im, (0, y)); y += im.height
    canvas.save(out); print("saved", out)


if __name__ == "__main__":
    main()
