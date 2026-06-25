#!/usr/bin/env python3
"""Figure 4 panel D (recheck) — canonical S. cerevisiae splicing motifs.

Top: the intron schematic (5' donor GTATGT / branch TACTAAC / 3' acceptor YAG,
Schirman et al.). Below, two columns (donor / branch) x two rows:
  - "Database Motifs"               : clean consensus IC logos (GTATGT, TACTAAC).
  - "Shorkie reconstruction (ISM)"  : the per-base Shorkie-ISM saliency at the actual
    donor / branch positions of the SS genes (DTD1/MMS2/HOP2), averaged across the
    intron donor/branch sites (reverse-complemented for - strand) — the motif the
    model reconstructs from in-silico mutagenesis.

(No SS-MoDISco exists on disk, so the reconstruction is taken directly from the
SS-ISM PWM around the splice sites.)

Output: reproduced/Figure_4D_reproduced.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F


def consensus_ic(seq):
    arr = np.zeros((len(seq), 4))
    for j, ch in enumerate(seq):
        arr[j, F._NT.index(ch)] = 2.0
    return arr


def revcomp_sal(v):
    return v[::-1, ::-1]


def collect_motif(kind, width):
    """Average Shorkie-ISM reference-projected saliency at donor or branch sites of the
    SS genes. kind in {'donor','branch'}. Returns (width,4) averaged PWM-like array."""
    stacks = []
    for spec in F.SPLICE:
        gene, part = spec["gene"], spec["part"]
        res = F.ism_saliency("motif_shorkie_RP_TSS", "gene_exp_motif_test_SS", part, 0)
        gf = F.gene_features(gene)
        if res is None or gf is None or not gf["introns"]:
            continue
        v, chrom, start, end = res
        plus = gf["strand"] == "+"
        for (i_s, i_e) in gf["introns"]:
            if kind == "donor":
                # motif starts at the 5' end of the intron (donor)
                gpos = i_s if plus else (i_e - width + 1)
            else:  # branch ~ 30 bp upstream of the 3' acceptor
                acc = i_e if plus else i_s
                gpos = (acc - 30) if plus else (acc + 30 - width + 1)
            ls = gpos - start
            if ls < 0 or ls + width > v.shape[0]:
                continue
            seg = v[ls:ls + width]
            if not plus:
                seg = revcomp_sal(seg)
            stacks.append(seg)
    if not stacks:
        return None
    return np.mean(stacks, axis=0)


def main():
    fig = plt.figure(figsize=(9, 5.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.55, wspace=0.3)

    # --- schematic bar (row 0, spans both cols) ---
    axs = fig.add_subplot(gs[0, :]); axs.axis("off")
    axs.set_xlim(0, 100); axs.set_ylim(0, 1)
    axs.add_patch(plt.Rectangle((6, 0.30), 4, 0.40, fc="#c8771f", ec="none"))   # 5' exon
    axs.add_patch(plt.Rectangle((10, 0.40), 80, 0.20, fc="#8a7fb8", ec="none"))  # intron
    axs.add_patch(plt.Rectangle((90, 0.30), 4, 0.40, fc="#c8771f", ec="none"))   # 3' exon
    axs.text(18, 0.78, "GUAUGU", ha="center", fontsize=11, family="monospace", weight="bold")
    axs.text(55, 0.78, "UACUAAC", ha="center", fontsize=11, family="monospace", weight="bold")
    axs.text(86, 0.78, "YAG", ha="center", fontsize=11, family="monospace", weight="bold")
    axs.text(0, 0.45, "Yeast splicing\nmotifs (Schirman et al.)", ha="left", va="center", fontsize=8)

    # --- column titles ---
    fig.text(0.34, 0.585, "5' splice site (donor site)", ha="center", fontsize=9, weight="bold")
    fig.text(0.77, 0.585, "Branch point", ha="center", fontsize=9, weight="bold")

    # --- Database Motifs row ---
    for col, seq in [(0, "GTATGT"), (1, "TACTAAC")]:
        ax = fig.add_subplot(gs[1, col])
        F.draw_pwm_logo(ax, consensus_ic(seq))
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("Database\nMotifs", fontsize=8, rotation=0, ha="right", va="center")

    # --- Shorkie reconstruction row ---
    donor = collect_motif("donor", 6)
    branch = collect_motif("branch", 7)
    for col, arr in [(0, donor), (1, branch)]:
        ax = fig.add_subplot(gs[2, col])
        if arr is None:
            ax.text(0.5, 0.5, "no SS-ISM", ha="center"); ax.axis("off")
        else:
            F.draw_pwm_logo(ax, arr)
            ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("Shorkie recon.\n(from ISM PWM)", fontsize=8, rotation=0, ha="right", va="center")

    fig.suptitle("Figure 4D (reproduced) — splicing motifs: database vs Shorkie ISM reconstruction",
                 y=0.99, fontsize=11)
    out = F.RD / "Figure_4D_reproduced.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("saved", out)
    print("donor reconstructed:", None if donor is None else donor.shape,
          "| branch reconstructed:", None if branch is None else branch.shape)


if __name__ == "__main__":
    main()
