#!/usr/bin/env python3
"""Shared helpers for the Figure 5 recheck builders (MSN2 & MSN4 time-course TF induction).

Faithful to the source notebook recipe (reproduce_figure_05.ipynb) and the source
scripts under scripts/04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/, but:
  - logos are rendered over the FULL 500 bp promoter window (the published A/F view),
    NOT the 90 bp `focus` zoom the notebook used;
  - panel letters follow the PUBLISHED figure: D/I = norm-R boxplots, E/J = TF-Modisco
    motif progression (the notebook had D/E and I/J swapped).

ISM logSED `scores.h5` schema (released + the recomputed ATG42 one are identical):
  logSED (N,500,4,3053) f16, seqs (N,500,170) bool, chr/start/end/strand.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

from shorkie import config
config.load()

REPRO = config.repo_root() / "reproduction" / "figure_05"
RD = REPRO / "reproduced"
RECHECK = REPRO / "recheck"
ISM = Path(config.path("results.ism_scores")) / "motif_shorkie_RP_TSS"
MODISCO_DIFF = ISM / "2_timepoint_analysis" / "modisco_analysis" / "results"
SHEET = (Path(config.path("work_root")) /
         "seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt")

TPS = [0, 5, 10, 15, 30, 45, 60, 90]   # the β-estradiol induction timepoints (minutes)
OFFSET = 1148                          # cleaned_sheet 'index' -> logSED track index (3053-track RNA-seq subset)
ACGT_COLORS = {"A": "green", "C": "blue", "G": "orange", "T": "red"}  # matches the source dna_letter_at()

_sheet = None


def sheet():
    global _sheet
    if _sheet is None:
        _sheet = pd.read_csv(SHEET, sep="\t")
    return _sheet


def tp_tracks(tf):
    """{timepoint: [logSED track indices]} for a TF's induction RNA-seq tracks."""
    sub = sheet()[sheet()["identifier"].str.contains(fr"{tf}_T")].copy()
    sub["tp"] = sub["identifier"].str.extract(fr"{tf}_T(\d+)_").astype(int)
    return {t: [i - OFFSET for i in sub[sub["tp"] == t]["index"].astype(int) if 0 <= i - OFFSET < 3053]
            for t in TPS}


def load_locus(h5path, idx, track_dict):
    """Per-timepoint mean-centered PWM (norm, for distances) + projected-on-reference PWM (proj, for logos).
    Verbatim recipe from the source notebook's load_locus()."""
    with h5py.File(h5path, "r") as h:
        seq1 = h["seqs"][idx, :, :4].astype("float32")          # reference one-hot (L,4)
        chrom = h["chr"][idx].decode(); s = int(h["start"][idx]); e = int(h["end"][idx])
        norm, proj = {}, {}
        for t, tr in track_dict.items():
            if not tr:
                continue
            arr = h["logSED"][idx, :, :, np.array(tr)].mean(axis=-1)  # timepoint-averaged (L,4)
            arr = arr - arr.mean(axis=-1, keepdims=True)             # mean-center across bases
            norm[t] = arr; proj[t] = arr * seq1
    return norm, proj, (chrom, s, e), seq1


def plot_full_logos(proj, win, title, png_name):
    """Full-500 bp per-timepoint ISM logo stack (one row per timepoint), shared symmetric
    y-scale so the induction-driven growth of the motif is visible across timepoints."""
    ts = sorted(proj.keys())
    ymax = max(float(np.abs(proj[t]).max()) for t in ts) or 1.0
    fig, axes = plt.subplots(len(ts), 1, figsize=(26, 1.15 * len(ts)), sharex=True)
    for ax, t in zip(np.atleast_1d(axes), ts):
        df = pd.DataFrame(proj[t], columns=list("ACGT"))
        logomaker.Logo(df, ax=ax, color_scheme=ACGT_COLORS)
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"T{t}", rotation=0, ha="right", va="center", fontsize=11)
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(False)
        ax.axhline(0, color="black", lw=0.6)
    fig.suptitle(f"{title}\n{win[0]}:{win[1]:,}-{win[2]:,}  (full 500 bp promoter window)", y=1.0, fontsize=12)
    fig.tight_layout()
    out = RD / png_name
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")
    return ymax


def distance_heatmap(norm, title, png_name):
    """8x8 pairwise Euclidean-distance heatmap of per-timepoint mean-centered PWMs (viridis)."""
    ts = sorted(norm.keys()); n = len(ts)
    D = np.zeros((n, n))
    for a, ta in enumerate(ts):
        for b, tb in enumerate(ts):
            D[a, b] = np.linalg.norm(norm[ta].flatten() - norm[tb].flatten())
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(D, cmap="viridis"); fig.colorbar(im, label="Euclidean distance")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{t}" for t in ts]); ax.set_yticklabels([f"T{t}" for t in ts])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    out = RD / png_name
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(D, index=[f"T{t}" for t in ts], columns=[f"T{t}" for t in ts]).to_csv(
        RD / png_name.replace(".png", "_matrix.csv"))
    print(f"  saved {out}")
    return D, ts


# ---- TF-Modisco motif helpers (E/J progression) ----
def ic(ppm, bg=np.array([.25, .25, .25, .25]), ps=0.001):
    return np.sum((np.log((ppm + ps) / (1 + ps * 4)) / np.log(2)) * ppm
                  - (np.log(bg) * bg / np.log(2))[None, :], axis=1)


def trim(cwm, thr=0.3, fl=4):
    sc = np.sum(np.abs(cwm), axis=1); t = np.max(sc) * thr; w = np.where(sc >= t)[0]
    return (0, cwm.shape[0]) if len(w) == 0 else (max(w.min() - fl, 0), min(w.max() + fl + 1, cwm.shape[0]))


def consensus(ppm):
    return "".join("ACGT"[i] for i in np.argmax(ppm, axis=1))


def max_run(seq, base):
    best = cur = 0
    for ch in seq:
        cur = cur + 1 if ch == base else 0
        best = max(best, cur)
    return best
