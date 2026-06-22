#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 4: rebuild panel-2A (SMT3 promoter, 3 rows).

Published 2A overlays predicted DNA logos of the SMT3 (YDR510W) promoter from
three sources — SpeciesLM (external; Tomaz da Silva et al.), Shorkie LM, and
Shorkie LM 15% iterative inference — showing all three recover the same promoter
motifs (poly(dA:dT), Cbf1, Tye7, Reb1). SMT3 = chrIV:1,469,400-1,469,705 (+ strand,
ATG at 1,469,400); the promoter is the ~1 kb immediately upstream.

Reproducible rows (in-repo Shorkie LM):
  Shorkie LM           experiments/.../preds_smt3_unmasked.npz   (unmasked inference)
  Shorkie 15% iter.    reproduced/iterative_smt3/preds_smt3_iterative.npz  (this recheck's GPU job)
We extract the 1 kb upstream of the ATG from each of the 4 SMT3 windows and average,
render IC logos, and verify (a) the unmasked and iterative predictions are mutually
consistent and (b) the published promoter motifs (poly(dA:dT), Cbf1 = CACGTG) appear.

SpeciesLM row: EXTERNAL model (johahi/specieslm-fungi-upstream-k1). We display its
released cached prediction (`all_prbs.npy`, 500 bp). Its exact position-wise
alignment to the Shorkie window cannot be reconstructed without re-running the
external model (cross-correlation against the Shorkie prediction finds no clean
offset; the SpeciesLM k-mer tokenisation/coordinates differ) — so it is shown over
the proximal promoter and labelled external/approximate. See DISCREPANCIES.md.

Outputs: reproduced/Figure_2A_reproduced.png, recheck/fig2A_consistency.csv.

Run (env yeast_ml): python reproduction/figure_02/recheck/build_2A_logos.py
"""
from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

WORK = str(config.path("work_root"))
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

ATG = 1469400
PROM = 1000              # full promoter window upstream of ATG
ALPH = ["A", "C", "G", "T"]


def upstream_pwm(x, windows, prom):
    accum = np.zeros((prom, 4), dtype=np.float64)
    n = 0
    for w in range(x.shape[0]):
        ws = int(windows[w][1])
        off = ATG - ws
        if off - prom < 0 or off > x.shape[1]:
            continue
        accum += np.asarray(x[w, off - prom:off, :], dtype=np.float64)
        n += 1
    return accum / max(n, 1)


def ic_matrix(pwm):
    p = pwm + 1e-9
    p = p / p.sum(axis=1, keepdims=True)
    ent = -(p * np.log2(p)).sum(axis=1)
    return (2.0 - ent)[:, None] * p


def draw_logo(ax, ic, title, xspan=None):
    # fast vectorised stacked-bar logo (A/C/G/T coloured; column height = IC bits).
    # At ~1 kb a per-letter logo is both slow and illegible; the colour/height
    # pattern reveals the conserved motifs (poly-dA:dT block, E-box peaks, etc.).
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]  # A C G T
    L = ic.shape[0]
    xs = np.linspace(xspan[0], xspan[1], L) if xspan else np.arange(L)
    width = (xs[1] - xs[0]) if L > 1 else 1
    bottom = np.zeros(L)
    for k in range(4):
        ax.bar(xs, ic[:, k], bottom=bottom, width=width, color=colors[k], linewidth=0, label=ALPH[k])
        bottom += ic[:, k]
    ax.set_xlim(xspan if xspan else (0, L)); ax.set_ylim(0, 2)
    ax.set_ylabel("bits", fontsize=8); ax.set_title(title, fontsize=10, loc="left")
    ax.set_yticks([0, 1, 2])


def seq_of(pwm):
    return "".join(ALPH[i] for i in pwm.argmax(1))


def main():
    iz = np.load(RD / "iterative_smt3" / "preds_smt3_iterative.npz", allow_pickle=True)
    windows = iz["windows"]
    sho_iter = upstream_pwm(np.asarray(iz["x_pred_iter"], dtype=np.float64), windows, PROM)
    uz = np.load(f"{WORK}/experiments/Shorkie_LM_SMT3_viz/inference_smt3_output/preds_smt3_unmasked.npz")
    sho_unmask = upstream_pwm(np.asarray(uz["x_pred"], dtype=np.float64), windows, PROM)
    x_true = upstream_pwm(np.asarray(uz["x_true"], dtype=np.float64), windows, PROM)

    spec = np.load(f"{WORK}/experiments/dependencies_DNALM/all_prbs.npy").astype(np.float64)
    spec = spec / spec.sum(axis=1, keepdims=True)

    gstart = ATG - PROM     # genomic x-axis start
    fig, axes = plt.subplots(3, 1, figsize=(16, 5.4))
    # row 1: SpeciesLM (external) over the proximal promoter (its 500 bp), labelled
    draw_logo(axes[0], ic_matrix(spec), "SpeciesLM (Tomaz da Silva et al.) — external, cached 500 bp (proximal)",
              xspan=(ATG - spec.shape[0], ATG))
    draw_logo(axes[1], ic_matrix(sho_unmask), "Shorkie LM (unmasked)", xspan=(gstart, ATG))
    draw_logo(axes[2], ic_matrix(sho_iter), "Shorkie LM 15% iterative inference", xspan=(gstart, ATG))
    axes[-1].set_xlabel(f"SMT3 (YDR510W) promoter — {PROM} bp upstream of ATG (chrIV:{gstart}-{ATG})", fontsize=9)
    fig.suptitle("Figure 2A (reproduced) — SMT3 promoter logos: SpeciesLM vs Shorkie LM vs Shorkie 15% iterative",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RD / "Figure_2A_reproduced.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # ---- consistency + motif checks (the reproducible claims) ----
    def agree(a, b):
        return float((a.argmax(1) == b.argmax(1)).mean())

    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    sho_seq = seq_of(sho_unmask)
    true_seq = seq_of(x_true)
    has_cbf1 = bool(re.search("CACGTG", sho_seq))        # Cbf1 E-box
    # poly(dA:dT): a run of >=8 A or >=8 T in the predicted promoter
    polyAT = bool(re.search("A{8,}|T{8,}", sho_seq))
    recon_acc = float((sho_unmask.argmax(1) == x_true.argmax(1)).mean())

    rows = [
        ("Shorkie_unmasked vs Shorkie_iter maxbase_agree", round(agree(sho_unmask, sho_iter), 4)),
        ("Shorkie_unmasked vs Shorkie_iter pwm_corr", round(corr(sho_unmask, sho_iter), 4)),
        ("Shorkie_unmasked reconstruction_acc vs true", round(recon_acc, 4)),
        ("Cbf1_Ebox(CACGTG)_in_Shorkie_promoter", int(has_cbf1)),
        ("polydAdT_run>=8_in_Shorkie_promoter", int(polyAT)),
    ]
    import csv
    with open(RECHECK / "fig2A_consistency.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["metric", "value"])
        for k, v in rows:
            print(f"  {k:<50} {v}")
            wr.writerow([k, v])
    print(f"\n[OK] -> {out}")
    print(f"[OK] -> {RECHECK/'fig2A_consistency.csv'}")


if __name__ == "__main__":
    main()
