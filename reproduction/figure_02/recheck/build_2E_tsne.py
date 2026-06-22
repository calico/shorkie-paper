#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 3: panel-2E t-SNE on ALL 16 chromosomes + a
quantified cluster-separation metric.

The committed reproduction capped the projection at 2500 points. The published 2E
uses the first-self-attention-layer Shorkie-LM embeddings of intervals across all
sixteen S. cerevisiae chromosomes, classified into 5 genomic-element groups
(Protein-coding gene, Intergenic region, tRNA, Transposable element, Promoter =
500 bp upstream of the start codon). This recheck uses all qualifying points (up to
TSNE_CAP, default 20000 -> effectively all) and reports the silhouette score of the
2-D projection by genomic-element class, to make "the classes cluster" a number.

Outputs: reproduced/panelE_tsne/{proj.npy,meta.csv}, reproduced/Figure_2E_reproduced.png,
recheck/fig2E_separation.csv.

Run (env yeast_ml; CPU, a few minutes) e.g. via tmux:
  reproduction/common/run_in_tmux.sh fig2e /tmp/fig2e.log \
    "/home/kchao10/miniconda3/envs/yeast_ml/bin/python3 reproduction/figure_02/recheck/build_2E_tsne.py"
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import time, glob
import numpy as np, pandas as pd, h5py
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import pyranges as pr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shorkie import config

CAP = int(os.environ.get("TSNE_CAP", "20000"))
N_ITER = int(os.environ.get("TSNE_NITER", "1000"))
SEED = 42
t0 = time.time()
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"; RECHECK = F2 / "recheck"; RECHECK.mkdir(parents=True, exist_ok=True)

umap_dir = config.path("results.umap")
files = sorted(glob.glob(str(umap_dir / "embeddings_LM_sequence" / "embeddings_chr*.h5")))
emb_data, meta = {}, []
for f in files:
    with h5py.File(f, "r") as h:
        if "metadata" in h:
            meta.append(np.array(h["metadata"]))
        for ds in h.keys():
            if ds.startswith("embeddings_"):
                emb_data.setdefault(ds, []).append(np.array(h[ds]))
print(f"[{time.time()-t0:.0f}s] loaded {len(files)} chromosome h5 files", flush=True)
meta_arr = np.concatenate(meta, axis=0)
ms = np.char.decode(meta_arr.astype(np.bytes_), "utf-8")
mdf = pd.DataFrame(ms, columns=["chrom", "start", "end", "strand", "feature", "gene_id"])
mdf["orig"] = mdf["feature"]
g = pr.read_gtf(str(config.path("genome.gtf"))).as_df()
mdf = mdf.merge(g[["gene_id", "gene_biotype"]].drop_duplicates(), on="gene_id", how="left")
isg = (mdf["feature"] == "gene") & mdf["gene_biotype"].notna()
mdf.loc[isg, "feature"] = mdf.loc[isg, "gene_biotype"]
inc = ["gene", "intergenic", "promoter", "tRNA", "transposable_element"]
exc = ["snoRNA", "ncRNA", "pseudogene", "snRNA", "rRNA"]
mask = mdf["orig"].isin(inc) & ~mdf["feature"].isin(exc)
nice = {"protein_coding": "Protein-coding gene", "intergenic": "Intergenic region",
        "tRNA": "tRNA", "transposable_element": "Transposable element", "promoter": "Promoter"}
ds = next(iter(emb_data))
allemb = np.concatenate(emb_data[ds], axis=0)
emb = allemb[mask.values]
mf = mdf.loc[mask].copy(); mf["feature"] = mf["feature"].map(lambda x: nice.get(x, x))
print(f"[{time.time()-t0:.0f}s] qualifying intervals: {emb.shape[0]} (16 chr); "
      f"classes: {dict(mf['feature'].value_counts())}", flush=True)

X = PCA(n_components=50, random_state=SEED).fit_transform(emb)
n_use = min(CAP, X.shape[0])
ridx = np.random.RandomState(SEED).choice(X.shape[0], n_use, replace=False)
mfs = mf.iloc[ridx].reset_index(drop=True)
print(f"[{time.time()-t0:.0f}s] PCA done; t-SNE on {n_use} pts (n_iter={N_ITER})", flush=True)
t1 = time.time()
proj = TSNE(n_components=2, random_state=SEED, init="pca", n_iter=N_ITER).fit_transform(X[ridx])
print(f"[{time.time()-t0:.0f}s] t-SNE done in {time.time()-t1:.0f}s", flush=True)

# separation metrics
labels = mfs["feature"].values
sil_2d = float(silhouette_score(proj, labels))
sil_pca = float(silhouette_score(X[ridx], labels))
per = pd.Series(silhouette_samples(proj, labels), index=labels).groupby(level=0).mean()

od = RD / "panelE_tsne"; od.mkdir(parents=True, exist_ok=True)
np.save(od / "proj.npy", proj)
mfs[["feature"]].to_csv(od / "meta.csv", index=False)

# class order + palette to match the published (Promoter, Intergenic, Protein-coding, tRNA, TE)
order = ["Promoter", "Intergenic region", "Protein-coding gene", "tRNA", "Transposable element"]
# published palette: the two large classes (Promoter/Protein-coding) blue & green,
# Intergenic orange (medium), tRNA red (small), Transposable element purple (tiny)
pal = {"Promoter": "tab:blue", "Protein-coding gene": "tab:green",
       "Intergenic region": "tab:orange", "tRNA": "tab:red", "Transposable element": "tab:purple"}
fig, ax = plt.subplots(figsize=(8, 7))
for f in order:
    i = (mfs["feature"] == f).values
    if i.any():
        ax.scatter(proj[i, 0], proj[i, 1], s=4, label=f, color=pal.get(f, "gray"))
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=9, markerscale=3)
ax.set_title(f"Figure 2E (reproduced) — t-SNE of 1st-attn LM embeddings, 16 chr "
             f"(n={n_use}, silhouette={sil_2d:.3f})", y=1.0, fontsize=11)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(RD / "Figure_2E_reproduced.png", dpi=140, bbox_inches="tight")

with open(RECHECK / "fig2E_separation.csv", "w") as fcsv:
    fcsv.write("metric,value\n")
    fcsv.write(f"n_points,{n_use}\n")
    fcsv.write(f"n_classes,{mfs['feature'].nunique()}\n")
    fcsv.write(f"silhouette_2d,{sil_2d:.4f}\n")
    fcsv.write(f"silhouette_pca50,{sil_pca:.4f}\n")
    for cls, v in per.items():
        fcsv.write(f"silhouette_2d[{cls}],{v:.4f}\n")
    for cls, c in mfs["feature"].value_counts().items():
        fcsv.write(f"count[{cls}],{c}\n")
print(f"[{time.time()-t0:.0f}s] classes={mfs['feature'].nunique()} silhouette_2d={sil_2d:.3f} "
      f"silhouette_pca={sil_pca:.3f}; per-class:\n{per}", flush=True)
print("[OK] saved proj/meta/png + recheck/fig2E_separation.csv")
