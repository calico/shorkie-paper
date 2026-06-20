#!/usr/bin/env python3
"""Figure 2E — precompute the t-SNE of LM embeddings (heavy/slow on shared nodes).

Loads the per-chromosome LM embeddings, refines feature labels via the GTF, reduces
with PCA-50, caps the point count, runs t-SNE, and caches the 2-D projection
(`panelE_tsne/proj.npy` + `meta.csv`) plus a rendered `Figure_2E_reproduced.png`.
The notebook then just LOADS this cache so it executes quickly. Run via tmux:

  reproduction/common/run_in_tmux.sh fig2e_tsne /tmp/fig2e.log \
    "conda run -n yeast_ml python reproduction/figure_02/panels/precompute_tsne.py"
"""
import os
# keep BLAS/OpenMP from oversubscribing on a shared node
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import time, glob
import numpy as np, pandas as pd, h5py
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pyranges as pr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shorkie import config

CAP = int(os.environ.get("TSNE_CAP", "2500"))
N_ITER = int(os.environ.get("TSNE_NITER", "300"))
t0 = time.time()
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
print(f"[{time.time()-t0:.0f}s] loaded {len(files)} h5 files", flush=True)
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
print(f"[{time.time()-t0:.0f}s] embedding {emb.shape}; reducing with PCA-50", flush=True)
X = PCA(n_components=50, random_state=42).fit_transform(emb)
print(f"[{time.time()-t0:.0f}s] PCA done; t-SNE on capped {min(CAP, X.shape[0])} pts", flush=True)
ridx = np.random.RandomState(42).choice(X.shape[0], min(CAP, X.shape[0]), replace=False)
mfs = mf.iloc[ridx].reset_index(drop=True)
t1 = time.time()
proj = TSNE(n_components=2, random_state=42, init="pca", n_iter=N_ITER).fit_transform(X[ridx])
print(f"[{time.time()-t0:.0f}s] t-SNE done in {time.time()-t1:.0f}s", flush=True)
od = Path(config.repo_root()) / "reproduction" / "figure_02" / "reproduced" / "panelE_tsne"
od.mkdir(parents=True, exist_ok=True)
np.save(od / "proj.npy", proj)
mfs[["feature"]].to_csv(od / "meta.csv", index=False)
pal = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
feats = list(mfs["feature"].unique()); cm = {f: pal[i % 5] for i, f in enumerate(feats)}
fig, ax = plt.subplots(figsize=(8, 7))
for f in feats:
    i = (mfs["feature"] == f).values
    ax.scatter(proj[i, 0], proj[i, 1], s=4, label=f, color=cm[f])
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=9, markerscale=3)
ax.set_title("Figure 2E (reproduced) — t-SNE of LM embeddings by genomic element", y=1.0)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(Path(config.repo_root()) / "reproduction" / "figure_02" / "reproduced" / "Figure_2E_reproduced.png",
            dpi=140, bbox_inches="tight")
print(f"[{time.time()-t0:.0f}s] saved cache+png; features: {dict(mfs['feature'].value_counts())}", flush=True)
