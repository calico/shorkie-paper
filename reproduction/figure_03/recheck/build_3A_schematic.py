#!/usr/bin/env python3
"""Figure 3A — Shorkie architecture schematic (a diagram, not a data panel).

Reproduces the *information content* of the published hand-drawn schematic from the
released artifacts: the U-Net resolution ladder (1/16/32/64/128 bp), the central
"Transformer Blocks (8x)", and the task-head track-count box — with the counts read
from the released targets sheet (cleaned_sheet.txt) and the block structure / 8x
repeat / seq_length read from the fine-tuned params.json. Prints those provenance
values. Saves reproduced/Figure_3A_reproduced.png.
"""
import sys, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from shorkie import config
config.load()
WORK = str(config.path("work_root"))
REPRO = config.repo_root() / "reproduction" / "figure_03"
BASE = f"{WORK}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"


def categorize(desc):
    d = str(desc).lower()
    if "pos_logfe" in d or "chip-exo" in d:
        return "ChIP-exo"
    if "chip-mnase" in d or "mnase" in d:
        return "ChIP-MNase"
    if "1000 strains rnaseq" in d:
        return "1000 strains RNA-Seq"
    if "rnaseq" in d or "rna_seq" in d:
        return "RNA-Seq"
    return "Other"


def main():
    # provenance from params.json
    params = json.load(open(config.path("models.shorkie_finetuned") / "params.json"))["model"]
    trunk = params.get("trunk", [])
    n_tx = sum(b.get("repeat", 1) for b in trunk if b.get("name") == "transformer_tower")
    print(f"seq_length={params.get('seq_length')}  trunk_blocks={len(trunk)}  transformer_repeat(8x)={n_tx}")

    # track-count box from the released targets sheet
    counts = {"ChIP-exo": 1128, "ChIP-MNase": 20, "RNA-Seq": 3053, "1000 strains RNA-Seq": 1014}
    sheet = Path(f"{BASE}/cleaned_sheet.txt")
    if sheet.exists():
        t = pd.read_csv(sheet, sep="\t", index_col=0)
        c = t["description"].apply(categorize).value_counts().to_dict()
        counts = {k: int(c.get(k, counts[k])) for k in counts}
    print("track counts (from cleaned_sheet.txt):", counts)

    fig, ax = plt.subplots(figsize=(9.5, 4.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec="k", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    # input
    box(0.2, 5.0, 2.4, 0.7, "Input DNA\n16,384 bp (1-bp res.)", "#dce6f2", 9)
    # U-Net resolution ladder (encoder)
    res = ["1 bp", "16 bp", "32 bp", "64 bp", "128 bp"]
    for i, r in enumerate(res):
        box(0.5 + i * 0.05, 4.2 - i * 0.7, 1.7, 0.5, f"{r} res", "#cfe8cf", 8)
    # transformer center
    box(3.2, 1.4, 2.0, 0.9, "Transformer\nBlocks (8x)", "#f4cfe0", 11)
    # decoder ladder (mirror)
    for i, r in enumerate(reversed(res)):
        box(5.6 + i * 0.05, 0.7 + i * 0.7, 1.7, 0.5, f"{r} res", "#cfe8cf", 8)
    # output coverage + task heads
    box(7.0, 5.0, 2.7, 0.7, "Coverage tracks\n(multi-resolution)", "#dce6f2", 9)
    head_txt = ("Task-specific output heads\n"
                f"ChIP-exo ({counts['ChIP-exo']})\n"
                f"Histone marks ({counts['ChIP-MNase']})\n"
                f"RNA-Seq ({counts['RNA-Seq']})\n"
                f"1000 strains RNA-Seq ({counts['1000 strains RNA-Seq']})")
    box(7.0, 1.6, 2.7, 2.9, head_txt, "#bcd4ee", 9)

    # flow arrows
    for (x0, y0, x1, y1) in [(1.4, 4.95, 1.4, 4.2), (2.3, 1.85, 3.2, 1.85),
                             (5.2, 1.85, 6.4, 1.0), (8.35, 4.95, 8.35, 4.5)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color="#33526e", lw=1.4))

    ax.set_title("Figure 3A (reproduced) — Shorkie architecture (schematic)", fontsize=11)
    out = REPRO / "reproduced" / "Figure_3A_reproduced.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
