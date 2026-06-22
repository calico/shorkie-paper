#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 2 (visual refinement): reproduce the panel-2C
TF-MoDISco motif-logo grid (6 datasets x 11 motifs).

The published 2C is a grid of small CWM logos — for each (dataset, motif) cell,
the de-novo TF-MoDISco motif recovered by Shorkie LM, or "—" if not recovered.
This renders that grid from the SAME artifacts the authors used: for each of the 9
TF motifs we pick, in each tier, the modisco pattern with the best TOMTOM match to
that TF (from `report_*/motifs.html`) and render its **CWM logo** (contrib_scores)
from the per-tier modisco `.h5`; "—" where no confident match (qval >= QTHRESH).
The 2 sequence motifs (5' splice donor, branch point) and TATA/TBP are matched by
consensus on the modisco PWMs where identifiable.

Conservation pattern (which cells are filled) is the data-driven grid from
`build_2C_grid.py`; this script makes the *logos* for the filled cells.

Outputs reproduced/Figure_2C_reproduced.png (logo grid) and keeps the q-value
heatmap as recheck/Figure_2C_qval_heatmap.png.

Run (env yeast_ml): python reproduction/figure_02/recheck/build_2C_logos.py
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

from shorkie import config

WORK = str(config.path("work_root"))
ML = f"{WORK}/experiments/motif_LM__unseen_species"
IND = f"{WORK}/experiments/motif_LM"
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
QTHRESH = 0.10
ALPH = list("ACGT")

# tier -> (modisco h5, report motifs.html)
TIERS = [
    ("S. cerevisiae\nR64",
     f"{IND}/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5",
     f"{IND}/saccharomycetales_viz_seq/unet_small_bert_drop/report_w16384_n100000/motifs.html"),
    ("4 S. cerevisiae\nstrains",
     f"{ML}/strains_select_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5",
     f"{ML}/strains_select_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("5 Saccharo-\nmycetales",
     f"{ML}/saccharomycetales_select_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5",
     f"{ML}/saccharomycetales_select_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("4 Ascomycota",
     f"{ML}/ascomycota_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5",
     f"{ML}/ascomycota_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("4 Orbiliales",
     f"{ML}/orbiliales_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5",
     f"{ML}/orbiliales_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
    ("4 Schizosaccharo-\nmycetales",
     f"{ML}/schizosaccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5",
     f"{ML}/schizosaccharomycetales_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"),
]
# 11 published columns. TF motifs use TOMTOM aliases; the 3 sequence motifs use a
# consensus (matched on the modisco PWM) — None alias => consensus-only column.
MOTIFS = [
    ("TBP", ["TBP", "SPT15", "TBP1"], "TATAAA"),
    ("5'SS\ndonor", None, "GTAAGT"),
    ("branch\npoint", None, "TACTAAC"),
    ("Cbf1p", ["CBF1"], None),
    ("Reb1.1", ["REB1"], None),
    ("Snf1.1", ["SNF1"], None),
    ("Mcm1.1", ["MCM1"], None),
    ("Rap1.1", ["RAP1"], None),
    ("Sfp1.2", ["SFP1"], None),
    ("Abf1.1", ["ABF1"], None),
    ("Dot6", ["DOT6"], None),
]


def norm_name(m):
    if not isinstance(m, str):
        return ""
    base = m.split("&")[0].split(".")[0].upper()
    return base.rstrip("P") if "&" in m else base


def best_pattern_for_tf(df, aliases):
    """Return (pattern_name, qval) of the modisco pattern best matching the TF."""
    targets = set(a.rstrip("P") for a in aliases) | set(aliases)
    best = (None, np.nan)
    for _, r in df.iterrows():
        for mi in (0, 1, 2):
            if norm_name(r.get(f"match{mi}")) in targets:
                q = r.get(f"qval{mi}")
                if pd.notna(q) and (np.isnan(best[1]) or q < best[1]):
                    best = (r["pattern"], float(q))
    return best


def best_pattern_for_consensus(h5path, consensus):
    """Scan modisco pos_patterns PWMs for the best match to a consensus (fwd or revcomp).
    Returns (pattern_name, score) with score in [0,1] (mean per-position max-prob match)."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    rc = "".join(comp.get(b, b) for b in reversed(consensus))
    idx = {b: i for i, b in enumerate(ALPH)}
    best = (None, 0.0)
    with h5py.File(h5path, "r") as f:
        for pn in f["pos_patterns"].keys():
            ppm = np.asarray(f["pos_patterns"][pn]["sequence"][:], float)
            ppm = ppm / (ppm.sum(1, keepdims=True) + 1e-9)
            for cons in (consensus, rc):
                k = len(cons)
                if ppm.shape[0] < k:
                    continue
                vec = np.array([idx[b] for b in cons])
                for s in range(ppm.shape[0] - k + 1):
                    sc = float(np.mean([ppm[s + j, vec[j]] for j in range(k)]))
                    if sc > best[1]:
                        best = (f"pos_patterns.{pn}", sc)
    return best


def trim_cwm(cwm, thr=0.3, flank=3):
    sc = np.abs(cwm).sum(1)
    if sc.max() <= 0:
        return cwm
    p = np.where(sc >= sc.max() * thr)[0]
    if len(p) == 0:
        return cwm
    return cwm[max(p.min() - flank, 0):min(p.max() + flank + 1, cwm.shape[0])]


def load_cwm(h5path, pattern_name):
    grp = pattern_name.replace("pos_patterns.", "")
    with h5py.File(h5path, "r") as f:
        return np.asarray(f["pos_patterns"][grp]["contrib_scores"][:], float)


def main():
    nrow, ncol = len(TIERS), len(MOTIFS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.45 * ncol, 1.0 * nrow + 0.6))
    filled = 0
    for ri, (tname, h5path, rep) in enumerate(TIERS):
        df = pd.read_html(rep)[0]
        for ci, (mname, aliases, consensus) in enumerate(MOTIFS):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            pat, ok = None, False
            if aliases is not None:
                pat, q = best_pattern_for_tf(df, aliases)
                ok = pat is not None and pd.notna(q) and q < QTHRESH
            if not ok and consensus is not None:
                pat, sc = best_pattern_for_consensus(h5path, consensus)
                ok = pat is not None and sc >= 0.80
            if ok:
                try:
                    cwm = trim_cwm(load_cwm(h5path, pat))
                    logomaker.Logo(pd.DataFrame(cwm, columns=ALPH), ax=ax,
                                   color_scheme={"A": "green", "C": "blue", "G": "orange", "T": "red"})
                    filled += 1
                except Exception as e:
                    ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14, transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14, color="0.4", transform=ax.transAxes)
            if ri == 0:
                ax.set_title(mname, fontsize=8)
            if ci == 0:
                ax.set_ylabel(tname, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Figure 2C (reproduced) — TF-MoDISco motif conservation grid "
                 "(de-novo CWM logos; '—' = not recovered)", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    out = RD / "Figure_2C_reproduced.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {filled}/{nrow*ncol} cells filled with CWM logos -> {out}")


if __name__ == "__main__":
    main()
