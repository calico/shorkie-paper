#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 5b: render panel-2B (iterative PPM reconstruction)
from the GPU iterative-masking output.

Published 2B ("Predicting 15% masked regions for each iteration") shows the per-
position A/C/G/T probabilities the Shorkie LM predicts for a promoter sub-window,
built up over successive 15%-mask iterations. We render that directly from
`reproduced/iterative_smt3/preds_smt3_iterative.npz` (x_pred_iter +
iter_assignment) for a 16 bp window of the SMT3 promoter: an A/C/G/T x position
probability matrix, annotated with the iteration in which each column was
predicted.

Outputs: reproduced/Figure_2B_reproduced.png, recheck/fig2B_ppm.csv.

Run (env yeast_ml): python reproduction/figure_02/recheck/build_2B_matrix.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
ALPH = ["A", "C", "G", "T"]
PROM_START = 1469090     # published 2A/2B SMT3 promoter sub-window start (chrIV)
NPOS = 16                # columns to display


def main():
    iz = np.load(RD / "iterative_smt3" / "preds_smt3_iterative.npz", allow_pickle=True)
    windows = iz["windows"]
    xp = np.asarray(iz["x_pred_iter"], dtype=np.float64)    # (W,L,4)
    assign = np.asarray(iz["iter_assignment"])              # (W,L)

    # use window 0; map the promoter sub-window to its offset
    w = 0
    ws = int(windows[w][1])
    off = PROM_START - ws
    sub = xp[w, off:off + NPOS, :]            # (NPOS,4) predicted probs
    sub = sub / sub.sum(axis=1, keepdims=True)
    it = assign[w, off:off + NPOS]            # iteration index per position

    # matrix A/C/G/T (rows) x position (cols)
    mat = sub.T                                # (4, NPOS)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(4)); ax.set_yticklabels(ALPH)
    ax.set_xticks(range(NPOS))
    ax.set_xticklabels([f"{PROM_START+i}" for i in range(NPOS)], rotation=90, fontsize=6)
    for i in range(4):
        for j in range(NPOS):
            ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.5 else "black")
    # iteration annotation strip on top
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(range(NPOS))
    ax2.set_xticklabels([f"it{int(v)+1}" for v in it], fontsize=6, rotation=90)
    ax.set_title("Figure 2B (reproduced) — iterative 15%-masked PPM reconstruction "
                 f"(SMT3 promoter chrIV:{PROM_START}-{PROM_START+NPOS}); "
                 "top = iteration that predicted each column", fontsize=10)
    ax.set_xlabel("genomic position (bp)")
    fig.colorbar(im, ax=ax, label="predicted probability", shrink=0.8)
    fig.tight_layout()
    out = RD / "Figure_2B_reproduced.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # save the PPM + iteration table
    import csv
    with open(RECHECK / "fig2B_ppm.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["genomic_pos", "iteration", "pA", "pC", "pG", "pT"])
        for j in range(NPOS):
            wcsv.writerow([PROM_START + j, int(it[j]) + 1] + [f"{sub[j,k]:.3f}" for k in range(4)])

    n_iters = int(assign[w].max()) + 1
    print(f"window 0 iterations: {n_iters}; promoter sub-window chrIV:{PROM_START}-{PROM_START+NPOS}")
    print(f"mean max-prob over sub-window: {sub.max(axis=1).mean():.3f}")
    print(f"[OK] -> {out}")
    print(f"[OK] -> {RECHECK/'fig2B_ppm.csv'}")
    # verify: every position was assigned to an iteration (full coverage)
    full = bool((assign[w] >= 0).all())
    print(f"[{'PASS' if full else 'FAIL'}] all positions covered by the iterative masking = {full}")


if __name__ == "__main__":
    main()
