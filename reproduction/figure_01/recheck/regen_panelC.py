#!/usr/bin/env python3
"""Figure 1 deep-recheck — regenerate the reproduced panel-1C dot plots with the
corrected YJM195 strain representative (was YJM1078).

Identical plotting logic to reproduce_figure_01.ipynb cell 7, except the strain
target is GCA_000975585_2 (YJM195) — the genome the published figure actually
shows ("TGT: Saccharomyces cerevisiae / YJM195"). Writes the canonical
reproduced/panelC_mummer/Figure_1C_reproduced.png and prints the aligned-fraction
spectrum used in the verification.

Run (env yeast_ml):
    python reproduction/figure_01/recheck/regen_panelC.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

RD = Path(config.repo_root()) / "reproduction" / "figure_01" / "reproduced"
MUM = RD / "panelC_mummer"

TARGETS = [
    ("R64-1.1 (species)", "GCA_000146045_2", "tab:blue"),
    ("YJM195 (strain)", "GCA_000975585_2", "tab:red"),
    ("N. glabratus CBS138 (order)", "GCA_000002545_2", "tab:purple"),
    ("C. albicans SC5314 (order)", "GCA_000182965_3", "tab:orange"),
    ("N. crassa OR74A (kingdom)", "GCA_000182925_2", "saddlebrown"),
    ("S. pombe 972h (kingdom)", "GCA_000002945_2", "teal"),
]


def load_coords(p):
    rows = []
    for ln in open(p):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 11 and parts[0].lstrip("-").isdigit():
            rows.append(parts)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["S1", "E1", "S2", "E2", "LEN1", "LEN2", "IDY", "LENR", "LENQ", "RT", "QT"])
    for c in ["S1", "E1", "S2", "E2", "LEN1", "LEN2", "IDY", "LENR", "LENQ"]:
        df[c] = pd.to_numeric(df[c])
    return df


def offs(df, tag, ln):
    s = df.groupby(tag)[ln].max().sort_values(ascending=False)
    o = {}
    cum = 0
    for n, sz in s.items():
        o[n] = cum
        cum += sz
    return o, cum


fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
cstats = {}
for ax, (label, acc, color) in zip(axes.flat, TARGETS):
    df = load_coords(MUM / f"{acc}.coords")
    ax.set_title(label, fontsize=10)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "no alignments", ha="center")
        continue
    ro, rt = offs(df, "RT", "LENR")
    qo, qt = offs(df, "QT", "LENQ")
    for _, r in df.iterrows():
        ax.plot([(ro[r.RT] + r.S1) / 1e6, (ro[r.RT] + r.E1) / 1e6],
                [(qo[r.QT] + r.S2) / 1e6, (qo[r.QT] + r.E2) / 1e6],
                "-", color=color, lw=0.6, alpha=0.8)
    ax.set_xlim(0, rt / 1e6)
    ax.set_ylim(0, qt / 1e6)
    ax.set_xlabel("R64 (Mb)", fontsize=8)
    ax.set_ylabel("target (Mb)", fontsize=8)
    ax.tick_params(labelsize=7)
    cstats[label] = df["LEN1"].sum() / df.groupby("RT")["LENR"].max().sum()
fig.suptitle("Figure 1C (reproduced) — MUMmer dot plots: R64 (x) vs representative genome (y)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = MUM / "Figure_1C_reproduced.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print("aligned-fraction vs R64:", {k: round(v, 4) for k, v in cstats.items()})
print("[OK] ->", out)
