#!/usr/bin/env python3
"""Figure 1 data-driven panel builders (B / C) — single source of truth, mirroring
`restyle_panels_DFG.py` (D/F/G). The reproduction notebook cells delegate to these
so the rendering lives in one place (no inline duplication; e.g. panel C's dot-plot
code previously lived in both the notebook and a separate `regen_panelC.py`).

These are faithful ports of the original notebook cells — the figures are unchanged
(approved); this is a refactor, not a re-style.

Panels 1A (architecture) and 1E (preprocessing) are hand-drawn schematics in the
paper; they are skipped (not reproduced), so they have no builder here.

  build_1B  circular NCBI-taxonomy cladogram (ete4); ensures species_tree.nwk first
  build_1C  MUMmer dot plots R64 vs one representative genome per dataset

Run (env yeast_ml):
    python reproduction/figure_01/recheck/build_panels_BC.py
"""
from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

REPRO = Path(config.repo_root()) / "reproduction" / "figure_01"
RD = REPRO / "reproduced"
PANELS = REPRO / "panels"
TREE_DIR = RD / "panelB_tree"
MUM = RD / "panelC_mummer"


def _run_panel(script, *args):
    """Run a panels/*.sh recompute via bash (yeast_ml env active) — same as the
    notebook's run_panel, so the builders are self-sufficient when run standalone."""
    cmd = ["bash", str(PANELS / script), *args]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(config.repo_root()))


# --------------------------------------------------------------------------- #
# Panel B — circular phylogenetic cladogram
# --------------------------------------------------------------------------- #
def build_1B():
    import csv
    from ete4 import Tree
    nwk = TREE_DIR / "species_tree.nwk"
    if not nwk.exists():
        _run_panel("build_tree.sh")

    def taxids_from(p):
        s = set()
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                t = (row.get("Taxon ID") or "").strip()
                if t.isdigit():
                    s.add(t)
        return s

    SL = Path(config.repo_root()) / "data" / "species_lists"
    sacc_ids = taxids_from(SL / "species_saccharomycetales_gtf.cleaned.csv")
    strain_ids = taxids_from(SL / "species_strains_gtf.cleaned.csv")

    t = Tree(open(nwk).read(), parser=1)
    leaves = list(t.leaves())
    N = len(leaves)

    def tid(n):
        return str(n.props.get("taxid") or "")

    def depth(n):
        d = 0
        p = n.up
        while p is not None:
            d += 1
            p = p.up
        return d

    maxd = max(depth(lf) for lf in leaves)
    ang, rad, issacc = {}, {}, {}
    for i, lf in enumerate(leaves):
        ang[lf] = 2 * np.pi * i / N
    for node in t.traverse():
        rad[node] = depth(node) / maxd
    for node in reversed(list(t.traverse())):
        if node.is_leaf:
            issacc[node] = tid(node) in sacc_ids
        else:
            ch = list(node.children)
            ang[node] = float(np.mean([ang[c] for c in ch]))
            issacc[node] = len(ch) > 0 and all(issacc.get(c, False) for c in ch)

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_axis_off()
    sth = [ang[lf] for lf in leaves if issacc[lf]]
    if sth:
        ax.fill_between(np.linspace(min(sth), max(sth), 300), 0, 1.06, color="lightgreen", alpha=0.35, zorder=-10)
    for node in t.traverse():
        if node.is_leaf:
            continue
        rp = rad[node]
        chs = list(node.children)
        ths = sorted(ang[c] for c in chs)
        arc = np.linspace(ths[0], ths[-1], max(2, int(abs(ths[-1] - ths[0]) * 200) + 2))
        sg = issacc[node]
        ax.plot(arc, [rp] * len(arc), color=("green" if sg else "0.45"), lw=(0.9 if sg else 0.4), zorder=1)
        for c in chs:
            cs = issacc.get(c, False)
            ax.plot([ang[c], ang[c]], [rp, rad[c]], color=("green" if cs else "0.45"),
                    lw=(0.9 if cs else 0.4), zorder=1)
    for lf in leaves:
        if tid(lf) == "559292":
            ax.plot([ang[lf]], [rad[lf]], "*", color="red", ms=13, zorder=6)
        elif tid(lf) in strain_ids:
            ax.plot([ang[lf]], [rad[lf]], "o", color="red", ms=3, zorder=5)
    nsh = sum(issacc[lf] for lf in leaves)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Figure 1B (reproduced) — NCBI tree, {N} fungal genomes; "
                 f"165_Saccharomycetales (green, {nsh} tips); R64 (red star)", fontsize=10, pad=18)
    out = TREE_DIR / "Figure_1B_reproduced.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", out, "| Saccharomycetales tips:", nsh)
    return nsh


# --------------------------------------------------------------------------- #
# Panel C — MUMmer dot plots (strain rep YJM195 = GCA_000975585.2)
# --------------------------------------------------------------------------- #
TARGETS_1C = [
    ("R64-1.1 (species)", "GCA_000146045_2", "tab:blue"),
    ("YJM195 (strain)", "GCA_000975585_2", "tab:red"),
    ("N. glabratus CBS138 (order)", "GCA_000002545_2", "tab:purple"),
    ("C. albicans SC5314 (order)", "GCA_000182965_3", "tab:orange"),
    ("N. crassa OR74A (kingdom)", "GCA_000182925_2", "saddlebrown"),
    ("S. pombe 972h (kingdom)", "GCA_000002945_2", "teal"),
]


def _load_coords(p):
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


def _offs(df, tag, ln):
    s = df.groupby(tag)[ln].max().sort_values(ascending=False)
    o, cum = {}, 0
    for n, sz in s.items():
        o[n] = cum
        cum += sz
    return o, cum


def build_1C():
    if not (MUM / "GCA_000146045_2.coords").exists():
        _run_panel("run_mummer.sh")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    cstats = {}
    for ax, (label, acc, color) in zip(axes.flat, TARGETS_1C):
        df = _load_coords(MUM / f"{acc}.coords")
        ax.set_title(label, fontsize=10)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "no alignments", ha="center")
            continue
        ro, rt = _offs(df, "RT", "LENR")
        qo, qt = _offs(df, "QT", "LENQ")
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
    print("aligned-fraction vs R64:", {k: round(v, 3) for k, v in cstats.items()})
    return cstats


if __name__ == "__main__":
    build_1B()
    build_1C()
    print("[OK] built 1B/1C -> reproduced/")
