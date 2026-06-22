#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 4 (visual refinement): panel 2A SMT3-promoter DNA
logos, rendered with the EXACT published method.

Published 2A stacks three conservation-weighted DNA letter logos of the SMT3
(YDR510W) promoter — SpeciesLM (external; Tomaz da Silva et al.), Shorkie LM, and
Shorkie LM 15% iterative inference — highlighting poly(dA:dT), Cbf1, Tye7, Reb1.

This reproduces each row with the upstream scripts' own `plot_dna_logo`
(conservation = 2 − entropy; per-position letter heights = p·conservation; colors
A=green, C=blue, G=orange, T=red), the same source data, and the same plot regions:

  SpeciesLM row  — `1_viz_dna_logo_specieslm_fungi.py`: all_prbs.npy, plot[97:207].
                   (external model's cached prediction; we render it, we don't re-run
                   the model.)
  Shorkie LM     — `2_viz_dna_pwm_shorkie_lm.py`: SMT3 (YDR510W) gene-averaged
                   512 bp-upstream PWM, plot[204:500], from preds_smt3_unmasked.npz.
  Shorkie 15% it — same averaged-upstream method on this recheck's GPU iterative
                   reconstruction (reproduced/iterative_smt3/preds_smt3_iterative.npz).

Outputs reproduced/Figure_2A_reproduced.png + recheck/fig2A_consistency.csv.
Run (env yeast_ml): python reproduction/figure_02/recheck/build_2A_logos.py
"""
from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

from shorkie import config

WORK = str(config.path("work_root"))
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

ATG = 1469400          # SMT3 (YDR510W) start codon, + strand
PAD = 512              # 2_viz upstream pad
ALPH = ["A", "C", "G", "T"]
_FP = FontProperties(family="DejaVu Sans", weight="bold")
_LET = {c: TextPath((-0.35, 0), c, size=1, prop=_FP) for c in "ACGT"}
_COL = {"A": "green", "C": "blue", "G": "orange", "T": "red"}  # exact published scheme


def dna_letter_at(letter, x, y, yscale, ax):
    t = (mpl.transforms.Affine2D().scale(1.35, yscale * 1.35)
         + mpl.transforms.Affine2D().translate(x, y) + ax.transData)
    ax.add_artist(PathPatch(_LET[letter], lw=0, fc=_COL[letter], transform=t))


def plot_dna_logo(ax, pwm, plot_start, plot_end):
    """Conservation-weighted letter logo (verbatim from the lm_SMT3_viz scripts)."""
    pwm = np.copy(pwm[plot_start:plot_end, :]).astype(float) + 1e-4
    pwm /= pwm.sum(axis=1, keepdims=True)
    ent = np.zeros_like(pwm)
    ent[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    cons = 2.0 - ent.sum(axis=1)
    for j in range(pwm.shape[0]):
        order = np.argsort(pwm[j])
        for ii in range(4):
            i = order[ii]
            h = pwm[j, i] * cons[j]
            y = 0.0 if ii == 0 else float(np.sum(pwm[j, order[:ii]] * cons[j]))
            dna_letter_at(ALPH[i], j + 0.5, y, h, ax)
    ax.set_xlim(0, plot_end - plot_start)
    ax.set_ylim(0, 2)
    ax.axhline(0.01, color="black", lw=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def upstream_pwm(x, windows):
    acc = np.zeros((PAD, 4)); n = 0
    for w in range(x.shape[0]):
        off = ATG - int(windows[w][1])
        if 0 <= off - PAD and off <= x.shape[1]:
            acc += np.asarray(x[w, off - PAD:off, :], float); n += 1
    return acc / max(n, 1)


def seq_of(pwm):
    return "".join(ALPH[i] for i in pwm.argmax(1))


def annotate_motifs(ax, pwm, plot_start, plot_end):
    """Mark the published 2A motifs (E-box=Cbf1/Tye7, poly(dA:dT), Reb1) on the row."""
    seq = seq_of(pwm[plot_start:plot_end])
    for m in re.finditer("CACGTG", seq):          # Cbf1/Tye7 E-box
        x = m.start() + 3
        ax.axvline(x, color="0.4", ls="--", lw=0.6)
        ax.text(x, 2.02, "Cbf1/Tye7", ha="center", va="bottom", fontsize=7, color="0.2")
    for m in re.finditer("TTACCCG|CGGGTAA", seq):  # Reb1
        x = m.start() + 3
        ax.axvline(x, color="0.4", ls="--", lw=0.6)
        ax.text(x, 2.02, "Reb1", ha="center", va="bottom", fontsize=7, color="0.2")
    runs = list(re.finditer("A{8,}|T{8,}", seq))
    if runs:
        r = max(runs, key=lambda m: m.end() - m.start())
        ax.text(0.5 * (r.start() + r.end()), 2.02, "poly(dA:dT)", ha="center", va="bottom", fontsize=7, color="0.2")


def main():
    iz = np.load(RD / "iterative_smt3" / "preds_smt3_iterative.npz", allow_pickle=True)
    windows = iz["windows"]
    sho_iter_up = upstream_pwm(np.asarray(iz["x_pred_iter"], float), windows)
    uz = np.load(f"{WORK}/experiments/Shorkie_LM_SMT3_viz/inference_smt3_output/preds_smt3_unmasked.npz")
    sho_up = upstream_pwm(np.asarray(uz["x_pred"], float), windows)
    true_up = upstream_pwm(np.asarray(uz["x_true"], float), windows)
    spec = np.load(f"{WORK}/experiments/dependencies_DNALM/all_prbs.npy").astype(float)

    # plot regions (verbatim from the upstream scripts)
    SHO_S, SHO_E = 204, 500            # 2_viz_dna_pwm_shorkie_lm.py
    SPEC_S, SPEC_E = 97, 207           # 1_viz_dna_logo_specieslm_fungi.py

    fig, axes = plt.subplots(3, 1, figsize=(16, 5.4))
    plot_dna_logo(axes[0], spec, SPEC_S, SPEC_E)
    axes[0].set_title("SpeciesLM (Tomaz da Silva et al.) — external, all_prbs[97:207]", fontsize=10, loc="left")
    plot_dna_logo(axes[1], sho_up, SHO_S, SHO_E)
    annotate_motifs(axes[1], sho_up, SHO_S, SHO_E)
    axes[1].set_title("Shorkie LM — SMT3 gene-averaged upstream PWM [204:500]", fontsize=10, loc="left")
    plot_dna_logo(axes[2], sho_iter_up, SHO_S, SHO_E)
    annotate_motifs(axes[2], sho_iter_up, SHO_S, SHO_E)
    axes[2].set_title("Shorkie LM 15% iterative inference — averaged upstream [204:500]", fontsize=10, loc="left")
    axes[2].set_xlabel("SMT3 (YDR510W) promoter (conservation-weighted DNA logo; A green / C blue / G orange / T red)", fontsize=9)
    fig.suptitle("Figure 2A (reproduced) — SMT3 promoter logos: SpeciesLM vs Shorkie LM vs Shorkie 15% iterative", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RD / "Figure_2A_reproduced.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # consistency + motif checks (the reproducible claims)
    def agree(a, b): return float((a.argmax(1) == b.argmax(1)).mean())
    def corr(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    sho_seq = seq_of(sho_up); recon = float((sho_up.argmax(1) == true_up.argmax(1)).mean())
    rows = [
        ("Shorkie_unmasked_vs_iter_pwm_corr", round(corr(sho_up, sho_iter_up), 4)),
        ("Shorkie_unmasked_vs_iter_maxbase_agree", round(agree(sho_up, sho_iter_up), 4)),
        ("Shorkie_unmasked_recon_acc_vs_true", round(recon, 4)),
        ("Cbf1_Ebox(CACGTG)_in_Shorkie_promoter", int(bool(re.search("CACGTG", sho_seq)))),
        ("polydAdT_run>=8_in_Shorkie_promoter", int(bool(re.search("A{8,}|T{8,}", sho_seq)))),
    ]
    import csv
    with open(RECHECK / "fig2A_consistency.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in rows:
            print(f"  {k:<44} {v}"); w.writerow([k, v])
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
