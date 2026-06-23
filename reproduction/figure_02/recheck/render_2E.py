#!/usr/bin/env python3
"""Figure 2E — re-render from the cached full-16-chr t-SNE projection, faithful to the
original `2_viz_clusters_LM.py` styling (figsize (8,7), s=3, no title, no grid,
spines top/right off, legend top-center ncol=3 fontsize=11, dpi=300), with the
published class palette (Promoter blue / Intergenic orange / Protein-coding green /
tRNA red / Transposable element purple).

Loads reproduced/panelE_tsne/{proj.npy,meta.csv} (computed by build_2E_tsne.py, which
runs t-SNE directly on the full embeddings with NO PCA) — no t-SNE recompute. Keeps
the silhouette metric in recheck/fig2E_separation.csv for the verifier.

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

PAL = {"Promoter": "tab:blue", "Intergenic region": "tab:orange",
       "Protein-coding gene": "tab:green", "tRNA": "tab:red",
       "Transposable element": "tab:purple"}
ORDER = ["Promoter", "Intergenic region", "Protein-coding gene", "tRNA", "Transposable element"]

sil = float(silhouette_score(proj, labels))
per = pd.Series(silhouette_samples(proj, labels), index=labels).groupby(level=0).mean()

fig, ax = plt.subplots(figsize=(8, 7))
for f in ORDER:
    i = (mf["feature"] == f).values
    if i.any():
        ax.scatter(proj[i, 0], proj[i, 1], s=3, alpha=1, label=f, color=PAL.get(f, "gray"))
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(bbox_to_anchor=(0.5, 1.16), loc="upper center", ncol=3, fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(RD / "Figure_2E_reproduced.png", dpi=300)
plt.close(fig)

with open(RECHECK / "fig2E_separation.csv", "w") as f:
    f.write("metric,value\n")
    f.write(f"n_points,{len(proj)}\nn_classes,{mf['feature'].nunique()}\nsilhouette_2d,{sil:.4f}\n")
    for cls, v in per.items():
        f.write(f"silhouette_2d[{cls}],{v:.4f}\n")
    for cls, c in mf["feature"].value_counts().items():
        f.write(f"count[{cls}],{c}\n")
print(f"[OK] 2E re-rendered (faithful original style), n={len(proj)}, silhouette={sil:.3f}")
print("per-class silhouette:", {k: round(v, 3) for k, v in per.items()})
