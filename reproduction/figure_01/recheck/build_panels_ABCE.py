#!/usr/bin/env python3
"""Figure 1 panels A / B / C / E builders — single source of truth, mirroring
`restyle_panels_DFG.py` (D/F/G). The reproduction notebook cells delegate to these
so the rendering lives in one place (no inline duplication; e.g. panel C's dot-plot
code previously lived in both the notebook and a separate `regen_panelC.py`).

These are faithful ports of the original notebook cells — the figures are unchanged
(approved); this is a refactor, not a re-style.

  build_1A  Shorkie-LM trunk block-stack schematic from the released params.json
  build_1B  circular NCBI-taxonomy cladogram (ete4); ensures species_tree.nwk first
  build_1C  MUMmer dot plots R64 vs one representative genome per dataset
  build_1E  data-preprocessing illustration (one-hot + region loss weight) on one R64 window

Run (env yeast_ml):
    python reproduction/figure_01/recheck/build_panels_ABCE.py
"""
from __future__ import annotations
import json
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
# Panel A — Shorkie-LM architecture schematic
# --------------------------------------------------------------------------- #
def build_1A():
    from collections import Counter
    params_file = Path(config.path("models.shorkie_lm")) / "params.json"
    with open(params_file) as f:
        P = json.load(f)
    m = P["model"]
    trunk = m.get("trunk", [])
    block_counts = Counter(b.get("name", "?") for b in trunk)
    print("Architecture (LM params.json):")
    print(f"  seq_length      = {m.get('seq_length')}")
    print(f"  trunk blocks    = {len(trunk)}  ->", dict(block_counts))
    print(f"  transformer towers = {block_counts.get('transformer_tower',0)}")
    print(f"  task/head       = MLM (masked token), use_bert={P['train'].get('use_bert')}, "
          f"mask_rate={P['train'].get('mask_rate')}")

    # simple block-stack diagram
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    ax.axis("off")
    x = 0
    for b in trunk[:18]:
        name = b.get("name", "?")
        ax.add_patch(plt.Rectangle((x, 0), 0.9, 1, facecolor="#bcd", edgecolor="k", lw=0.6))
        ax.text(x + 0.45, 0.5, name.replace("_", "\n"), ha="center", va="center", fontsize=5.5)
        x += 1
    ax.set_xlim(0, max(18, len(trunk)))
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1A (reproduced) — Shorkie-LM trunk block stack "
                 "(multi-res U-Net + transformers; MLM head)", fontsize=10)
    fig.tight_layout()
    out = RD / "Figure_1A_reproduced.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", out)


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


# --------------------------------------------------------------------------- #
# Panel E — data-preprocessing illustration
# --------------------------------------------------------------------------- #
def build_1E():
    import pysam
    SEQ_LEN, STRIDE = 16384, 4096
    fa = config.path("genome.fasta")
    fasta = pysam.Fastafile(str(fa))
    chrom = fasta.references[0]
    clen = fasta.get_reference_length(chrom)
    n_windows = max(0, (clen - SEQ_LEN) // STRIDE + 1)
    print(f"R64 {chrom}: {clen:,} bp -> {n_windows} windows of {SEQ_LEN} bp @ stride {STRIDE}")
    # one-hot a small slice for illustration
    sl = fasta.fetch(chrom, 1000, 1000 + 60).upper()
    mp = {"A": 0, "C": 1, "G": 2, "T": 3}
    oh = np.zeros((4, len(sl)))
    for i, b in enumerate(sl):
        if b in mp:
            oh[mp[b], i] = 1
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.2), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].imshow(oh, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(list("ACGT"))
    axes[0].set_title(f"Figure 1E (reproduced) — one-hot of {chrom}:1000-1060 (input channels 0-3)")
    axes[0].set_xticks([])
    # region loss weight cartoon: repeat=0, gene/intergenic=1
    w = np.ones(len(sl))
    w[20:30] = 0.0  # pretend repeat-masked stretch
    axes[1].fill_between(range(len(sl)), 0, w, step="mid", color="tab:green", alpha=0.5)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_yticks([0, 1])
    axes[1].set_ylabel("loss\nweight", fontsize=8)
    axes[1].set_xlabel("position (bp)")
    axes[1].set_title("region loss weight (repeat-masked positions -> 0; example)", fontsize=9)
    fig.tight_layout()
    out = RD / "Figure_1E_reproduced.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Pipeline: chunk(16384/4096) -> one-hot(4ch)+species(166ch) -> homolog removal "
          "-> 7% repeat threshold -> chr split (test chrXI/XIII/XV, valid chrXII/XIV/XVI)")


if __name__ == "__main__":
    build_1A()
    build_1B()
    build_1C()
    build_1E()
    print("[OK] built 1A/1B/1C/1E -> reproduced/")
