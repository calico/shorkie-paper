#!/usr/bin/env python3
"""Figure 1 deep-recheck — restyle panels 1F & 1G to match the published render.

The reproduced numbers are already exact (12/12 Δ=0). This script regenerates the
*figures* so the side-by-side against the published panels is visually faithful:

  1F  validation-loss curves: dashed lines per corpus (blue/orange/green/red),
      raw per-epoch loss, faint vertical min-marker at the argmin batch, legend
      "<label>; loss = <min>", x = epoch * 64 (extends to ~320k for 165_Sacc).
  1G  region-specific perplexity: gene (blue) / intergenic (orange) grouped bars,
      zoomed y-axis, title "Region-specific Perplexity (Gene vs Intergenic)".

Published display labels differ from the internal tier keys (kingdom tier is shown
as "1341_Fungus"; the F legend capitalises "R64_Yeast"). Those are applied here.

Writes the canonical reproduced PNGs (so the committed figure matches) AND a copy
under recheck/ for the composites.

Run (env yeast_ml):
    python reproduction/figure_01/recheck/restyle_panels_FG.py
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

REPRO = Path(config.repo_root()) / "reproduction" / "figure_01"
RD = REPRO / "reproduced"
RECHECK = REPRO / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)
LMR = Path(config.path("lm_experiment_root")) / "test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI"

TIERS = ["R64_yeast", "80_Strains", "165_Saccharomycetales", "1342_Fungus"]
COL = {"R64_yeast": "tab:blue", "80_Strains": "tab:orange",
       "165_Saccharomycetales": "tab:green", "1342_Fungus": "tab:red"}
# published display labels
LABEL_F = {"R64_yeast": "R64_Yeast", "80_Strains": "80_Strains",
           "165_Saccharomycetales": "165_Saccharomycetales", "1342_Fungus": "1341_Fungus"}
LABEL_G = {"R64_yeast": "R64_yeast", "80_Strains": "80_Strains",
           "165_Saccharomycetales": "165_Saccharomycetales", "1342_Fungus": "1341_Fungus"}

SUBS = {
    "R64_yeast": "lm_r64_gtf/lm_r64_gtf_unet_small",
    "80_Strains": "lm_strains_gtf/lm_strains_gtf_unet_small",
    "165_Saccharomycetales": "LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small",
    "1342_Fungus": "LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small",
}
PSUBS = dict(SUBS)
PSUBS["165_Saccharomycetales"] += "_bert_drop"
PSUBS["1342_Fungus"] += "_bert_drop"

VLRE = re.compile(r"valid_loss:\s*([0-9.]+)")
BATCHES_PER_EPOCH = 64


def restyle_1F():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    mins = {}
    for t in TIERS:
        vls = np.array([float(VLRE.search(ln).group(1)) for ln in open(LMR / SUBS[t] / "train" / "train.out")
                        if ln.strip().startswith("Epoch") and VLRE.search(ln)])
        x = np.arange(1, len(vls) + 1) * BATCHES_PER_EPOCH
        mins[t] = float(vls.min())
        ax.plot(x, vls, ls="--", color=COL[t], lw=1.1, alpha=0.9,
                label=f"{LABEL_F[t]}; loss = {vls.min():.4f}")
        # faint vertical min-marker at the argmin batch (matches the published guide lines)
        ax.axvline(int(vls.argmin() + 1) * BATCHES_PER_EPOCH, color=COL[t], ls="--", lw=0.8, alpha=0.35)
    ax.set_xlabel("# Training Batches")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Losses")
    ax.set_ylim(0.404, 0.448)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    for out in (RD / "Figure_1F_reproduced.png", RECHECK / "Figure_1F_restyled.png"):
        fig.savefig(out, dpi=140)
    plt.close(fig)
    print("[1F] min valid loss:", {LABEL_F[t]: round(mins[t], 4) for t in TIERS})
    return mins


def restyle_1G():
    gene, inter = {}, {}
    for t in TIERS:
        f = LMR / PSUBS[t] / "test_testset_perplexity_region" / "test_testset_perplexity_region.out"
        rows, inreg = [], False
        for ln in open(f):
            s = ln.strip()
            if s.startswith("Region-specific"):
                inreg = True
                continue
            elif inreg and s:
                rows.append(s)
        body = [r for r in rows if re.match(r"^\d+\s+\w+", r)]
        df = pd.read_csv(StringIO("\n".join(rows[:1] + body)), sep=r"\s+")
        gene[t] = float(df[df.region == "gene"]["perplexity"].iloc[0])
        inter[t] = float(df[df.region == "intergenic"]["perplexity"].iloc[0])
    x = np.arange(len(TIERS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w / 2, [gene[t] for t in TIERS], w, label="gene", color="tab:blue")
    ax.bar(x + w / 2, [inter[t] for t in TIERS], w, label="intergenic", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_G[t] for t in TIERS], fontsize=9)
    ax.set_ylim(3.52, 3.78)
    ax.set_ylabel("Perplexity")
    ax.set_title("Region-specific Perplexity (Gene vs Intergenic)")
    ax.legend()
    fig.tight_layout()
    for out in (RD / "Figure_1G_reproduced.png", RECHECK / "Figure_1G_restyled.png"):
        fig.savefig(out, dpi=140)
    plt.close(fig)
    print("[1G] gene:", {LABEL_G[t]: round(gene[t], 4) for t in TIERS})
    print("[1G] inter:", {LABEL_G[t]: round(inter[t], 4) for t in TIERS})
    return gene, inter


if __name__ == "__main__":
    restyle_1F()
    restyle_1G()
    print("[OK] restyled 1F/1G -> reproduced/ + recheck/")
