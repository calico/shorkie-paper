#!/usr/bin/env python3
"""Revision experiment 03, step 2 — supplementary figure describing the 3,053 induction
RNA-seq tracks.

Four panels from the step-1 tables, designed so a reader can see the whole design at a
glance rather than reconstructing it from prose:

  A  where the 3,053 tracks come from (the two partitions)
  B  tracks per induced gene, split by partition -- shows the 8 TFs are deep-replicate
     series while the broader panel is mostly one replicate per timepoint
  C  timepoint occupancy -- the two distinct sampling schedules
  D  replicate depth per (gene, timepoint)

Writes ``results/atlas_design.png``. CPU only, seconds.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shorkie import config

PART_COLOR = {"8_TF": "#377eb8", "gene_panel": "#ff7f00"}
PART_LABEL = {"8_TF": "8 TF perturbations (replicate series)",
              "gene_panel": "broader gene panel"}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot the induction RNA-seq atlas design.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    by_gene_csv = out_dir / "atlas_by_gene.csv"
    tp_csv = out_dir / "atlas_timepoints.csv"
    for p in (by_gene_csv, tp_csv):
        if not p.exists():
            sys.exit(f"error: {p} not found -- run 1_atlas_accounting.py first")

    by_gene = pd.read_csv(by_gene_csv)
    occ = pd.read_csv(tp_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    (axA, axB), (axC, axD) = axes

    # --- A: partition composition ---------------------------------------------------
    comp = by_gene.groupby("partition").agg(genes=("gene", "size"), tracks=("tracks", "sum"))
    comp = comp.reindex([p for p in PART_COLOR if p in comp.index])
    bottom = 0
    for part, row in comp.iterrows():
        axA.bar(0, row.tracks, bottom=bottom, width=0.55, color=PART_COLOR[part],
                label=f"{PART_LABEL[part]}\n{int(row.genes)} genes, {int(row.tracks):,} tracks")
        axA.text(0, bottom + row.tracks / 2, f"{int(row.tracks):,}", ha="center", va="center",
                 color="white", fontsize=12, weight="bold")
        bottom += row.tracks
    axA.set_xlim(-0.7, 1.9); axA.set_xticks([])
    axA.set_ylabel("RNA-seq tracks")
    axA.set_title(f"A · {int(comp.tracks.sum()):,} induction RNA-seq tracks "
                  f"from {int(comp.genes.sum())} induced genes", fontsize=11, loc="left")
    axA.legend(fontsize=8, loc="upper right", frameon=False)

    # --- B: tracks per gene ------------------------------------------------------------
    bins = np.arange(0.5, by_gene.tracks.max() + 1.5)
    for part in comp.index:
        axB.hist(by_gene[by_gene.partition == part].tracks, bins=bins,
                 color=PART_COLOR[part], alpha=0.75, label=PART_LABEL[part])
    axB.set_yscale("log")
    axB.set_xlabel("tracks per induced gene"); axB.set_ylabel("genes (log)")
    axB.set_title("B · replicate depth is concentrated in the 8 TFs", fontsize=11, loc="left")
    axB.legend(fontsize=8, frameon=False)

    # --- C: timepoint occupancy ----------------------------------------------------------
    for part in comp.index:
        sub = occ[occ.partition == part].sort_values("timepoint")
        axC.plot(sub.timepoint, sub.tracks, "o-", color=PART_COLOR[part],
                 label=PART_LABEL[part], ms=5)
    axC.set_xlabel("minutes after $\\beta$-estradiol induction")
    axC.set_ylabel("tracks at this timepoint")
    axC.set_title("C · two distinct sampling schedules", fontsize=11, loc="left")
    axC.grid(alpha=0.3); axC.legend(fontsize=8, frameon=False)

    # --- D: replicates per (gene, timepoint) ----------------------------------------------
    for part in comp.index:
        sub = by_gene[by_gene.partition == part]["mean_replicates_per_timepoint"]
        axD.hist(sub, bins=np.arange(0.5, max(2, sub.max()) + 1.0, 0.25),
                 color=PART_COLOR[part], alpha=0.75, label=PART_LABEL[part])
    axD.set_yscale("log")
    axD.set_xlabel("mean replicates per (gene, timepoint)")
    axD.set_ylabel("genes (log)")
    axD.set_title("D · the broader panel is near-singleton per timepoint", fontsize=11, loc="left")
    axD.legend(fontsize=8, frameon=False)

    fig.suptitle("Design of the induction RNA-seq resource generated for this study", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / "atlas_design.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("saved", out_png, flush=True)

if __name__ == "__main__":
    main()
