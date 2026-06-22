#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 4 (visual refinement): re-render panel 2E from the
cached full-16-chr t-SNE projection with the published class palette + layout.

Loads reproduced/panelE_tsne/{proj.npy,meta.csv} (computed by build_2E_tsne.py over
all 16 chromosomes) — no t-SNE recompute — and renders with the published colours:
the two large classes (Promoter, Protein-coding gene) in blue & green, Intergenic in
orange, tRNA in red, Transposable element in purple. Keeps the silhouette metric.

Run (env yeast_ml): python reproduction/figure_02/recheck/render_2E.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from shorkie import config

F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
cache = RD / "panelE_tsne"

proj = np.load(cache / "proj.npy")
mf = pd.read_csv(cache / "meta.csv")
labels = mf["feature"].values

# published palette: two big classes blue/green, medium orange, tiny red/purple
PAL = {"Promoter": "tab:blue", "Protein-coding gene": "tab:green",
       "Intergenic region": "tab:orange", "tRNA": "tab:red", "Transposable element": "tab:purple"}
ORDER = ["Promoter", "Intergenic region", "Protein-coding gene", "tRNA", "Transposable element"]

sil = float(silhouette_score(proj, labels))
per = pd.Series(silhouette_samples(proj, labels), index=labels).groupby(level=0).mean()

fig, ax = plt.subplots(figsize=(8, 7))
for f in ORDER:
    i = (mf["feature"] == f).values
    if i.any():
        ax.scatter(proj[i, 0], proj[i, 1], s=4, label=f, color=PAL.get(f, "gray"))
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=9, markerscale=3, frameon=True)
ax.set_title(f"Figure 2E (reproduced) — t-SNE of 1st-attn LM embeddings, 16 chr "
             f"(n={len(proj)}, silhouette={sil:.3f})", y=1.0, fontsize=11)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(RD / "Figure_2E_reproduced.png", dpi=140, bbox_inches="tight")
plt.close(fig)

with open(RECHECK / "fig2E_separation.csv", "w") as f:
    f.write("metric,value\n")
    f.write(f"n_points,{len(proj)}\nn_classes,{mf['feature'].nunique()}\nsilhouette_2d,{sil:.4f}\n")
    for cls, v in per.items():
        f.write(f"silhouette_2d[{cls}],{v:.4f}\n")
    for cls, c in mf["feature"].value_counts().items():
        f.write(f"count[{cls}],{c}\n")
print(f"[OK] 2E re-rendered (published palette), n={len(proj)}, silhouette={sil:.3f}")
print("per-class silhouette:", {k: round(v, 3) for k, v in per.items()})
