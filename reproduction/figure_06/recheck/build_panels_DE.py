#!/usr/bin/env python3
"""Figure 6 panels D (Yeast) and E (Random) — published two-subpanel composites.

Each panel = Shorkie (blue) scatter on the left + DREAM-RNN (green) scatter on the
right, both vs measured MAUDE expression, matching the published layout/labels.
Published targets (Shorkie / DREAM Pearson): D 0.695 / 0.891, E 0.744 / 0.981.

Outputs reproduced/Figure_6D.png, reproduced/Figure_6E.png and appends rows to
recheck/fig6_DEFGH_R.csv.
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
    ("6D", "yeast_seqs", "Yeast Sequence", "Rafi et al. Yeast Sequences\n(single-sequence)", 0.695, 0.891),
    ("6E", "all_random_seqs", "Random Sequence", "Rafi et al. Random Sequences\n(single-sequence)", 0.744, 0.981),
]


def _scatter(ax, x, y, color, label, pear, spear, xlabel, ylabel, title):
    ax.scatter(x, y, color=color, s=15, alpha=0.6, label=label)
    s, b = np.polyfit(x, y, 1)
    xr = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xr, s * xr + b, color=mc.COL_REG, lw=2,
            label=f"Pearson: {pear:.3f}, Spearman: {spear:.3f}")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)
    ax.set_box_aspect(1)  # square subplots, matching the published figure


def main():
    gt = mc.load_ground_truth()
    dream = mc.load_dream()
    rows = []
    for panel, seq_type, nice, suptitle, pub_shk, pub_dr in PANELS:
        s_pred, s_gt, s_pear, s_spear, s_n, n_genes = mc.shorkie_single(seq_type, gt)
        d_pred, d_gt, d_pear, d_spear, d_n = mc.dream_single(seq_type, gt, dream)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        _scatter(axes[0], s_pred, s_gt, mc.COL_SHORKIE, nice, s_pear, s_spear,
                 "Shorkie Predicted logSED", "Average expression levels (YFP fluorescence)",
                 f"{nice}:\nAggregated across all genes, 180 bp")
        _scatter(axes[1], d_pred, d_gt, mc.COL_DREAM, "Data", d_pear, d_spear,
                 "DREAM-RNN model prediction", "Average expression levels (YFP fluorescence)",
                 f"{nice}:\nDREAM-RNN Predictions vs. Experimental Measurements")
        fig.suptitle(suptitle, fontsize=13)
        fig.text(0.01, 0.99, panel[-1], fontsize=20, fontweight="bold", va="top", ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = REPRO / f"Figure_{panel}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)

        print(f"[{panel}] {seq_type}: genes={n_genes}  "
              f"Shorkie(n={s_n}) Pearson={s_pear:.3f} Spearman={s_spear:.3f} (pub {pub_shk}, Δ{s_pear-pub_shk:+.3f}) | "
              f"DREAM(n={d_n}) Pearson={d_pear:.3f} Spearman={d_spear:.3f} (pub {pub_dr}, Δ{d_pear-pub_dr:+.3f})  -> {out.name}")
        rows.append([panel, seq_type, "Shorkie", f"{s_pear:.4f}", f"{s_spear:.4f}", s_n, pub_shk, f"{s_pear-pub_shk:+.4f}"])
        rows.append([panel, seq_type, "DREAM-RNN", f"{d_pear:.4f}", f"{d_spear:.4f}", d_n, pub_dr, f"{d_pear-pub_dr:+.4f}"])

    out_csv = RECHECK / "fig6_DEFGH_R.csv"
    write_header = not out_csv.exists()
    # DE builder writes fresh; FGH appends. Always rewrite the DE rows at the top.
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "seq_type", "model", "pearson_repro", "spearman_repro", "n", "published_pearson", "delta"])
        w.writerows(rows)
    print(f"\nWrote {out_csv} (panels 6D/6E)")


if __name__ == "__main__":
    main()
