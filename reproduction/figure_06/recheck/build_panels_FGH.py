#!/usr/bin/env python3
"""Figure 6 panels F (SNV), G (Motif Perturbation), H (Motif Tiling) — published
two-subpanel composites of reference-alternate (Alt - Ref) effects.

Each panel = Shorkie (blue) Alt-Ref logSED difference on the left + DREAM-RNN (green)
Alt-Ref prediction difference on the right, both vs measured expression difference.
Published targets (Shorkie / DREAM Pearson): F 0.539 / 0.866, G 0.819 / 0.983,
H 0.561 / 0.943.

Outputs reproduced/Figure_6{F,G,H}.png and appends rows to recheck/fig6_DEFGH_R.csv.
The on-disk motif_tiling_seqs tree has 13 gene dirs (others 22) — the per-panel print
reports the gene count so any 6H residual can be attributed to the gene set.
"""
import os
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpra_common as mc
from shorkie import config

config.load()
REPRO = config.repo_root() / "reproduction" / "figure_06" / "reproduced"
REPRO.mkdir(parents=True, exist_ok=True)
RECHECK = config.repo_root() / "reproduction" / "figure_06" / "recheck"

PANELS = [
    ("6F", "all_SNVs_seqs", "SNV Sequences", "Rafi et al. SNV Perturbation Sequences\n(dual-sequence)", 0.539, 0.866),
    ("6G", "motif_perturbation", "Motif Perturbation Sequences", "Rafi et al. Motif Perturbation Sequences\n(dual-sequence)", 0.819, 0.983),
    ("6H", "motif_tiling_seqs", "Motif Tiling Sequences", "Rafi et al. Motif Tiling Sequences\n(dual-sequence)", 0.561, 0.943),
]


def _scatter(ax, x, y, color, label, pear, spear, xlabel, ylabel, title):
    ax.scatter(x, y, color=color, s=15, alpha=0.6, label=label)
    s, b = np.polyfit(x, y, 1)
    xr = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xr, s * xr + b, color=mc.COL_REG, lw=2,
            label=f"Pearson: {pear:.3f}, Spearman: {spear:.3f}")
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main():
    gt = mc.load_ground_truth()
    dream = mc.load_dream()
    rows = []
    for panel, seq_type, nice, suptitle, pub_shk, pub_dr in PANELS:
        s_pred, s_gt, s_pear, s_spear, s_n, n_genes = mc.shorkie_dual(seq_type, gt)
        d_pred, d_gt, d_pear, d_spear, d_n = mc.dream_dual(seq_type, gt, dream)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        _scatter(axes[0], s_pred, s_gt, mc.COL_SHORKIE, "Data", s_pear, s_spear,
                 "Shorkie predicted logSED differences (Alt - Ref)",
                 "Average expression levels differences\n(YFP fluorescence, Alt - Ref)",
                 f"{nice}:\nAggregated across all genes, 180 bp")
        _scatter(axes[1], d_pred, d_gt, mc.COL_DREAM, "Data", d_pear, d_spear,
                 "DREAM-RNN model prediction differences (Alt - Ref)",
                 "Average expression levels differences\n(YFP fluorescence, Alt - Ref)",
                 f"{nice}:\nDREAM-RNN Model vs Ground Truth ALT / REF Differences")
        fig.suptitle(f"Figure {panel} (reproduced) — {suptitle}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = REPRO / f"Figure_{panel}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)

        print(f"[{panel}] {seq_type}: genes={n_genes}  "
              f"Shorkie(n={s_n}) Pearson={s_pear:.3f} Spearman={s_spear:.3f} (pub {pub_shk}, Δ{s_pear-pub_shk:+.3f}) | "
              f"DREAM(n={d_n}) Pearson={d_pear:.3f} Spearman={d_spear:.3f} (pub {pub_dr}, Δ{d_pear-pub_dr:+.3f})  -> {out.name}")
        rows.append([panel, seq_type, "Shorkie", f"{s_pear:.4f}", f"{s_spear:.4f}", s_n, pub_shk, f"{s_pear-pub_shk:+.4f}"])
        rows.append([panel, seq_type, "DREAM-RNN", f"{d_pear:.4f}", f"{d_spear:.4f}", d_n, pub_dr, f"{d_pear-pub_dr:+.4f}"])

    out_csv = RECHECK / "fig6_DEFGH_R.csv"
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"\nAppended panels 6F/6G/6H to {out_csv}")


if __name__ == "__main__":
    main()
