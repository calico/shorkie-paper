#!/usr/bin/env python3
"""Figure 2E — t-SNE of the 1st-self-attention-layer Shorkie-LM embeddings, faithful
to the original `umap_cluster_promoter/2_viz_clusters_LM.py`.

The published 2E runs t-SNE DIRECTLY on the full embedding vectors (NO PCA
pre-reduction) over all qualifying intervals across the sixteen S. cerevisiae
chromosomes, classified into 5 genomic-element groups (Protein-coding gene,
Intergenic region, tRNA, Transposable element, Promoter = 500 bp upstream of the
start codon). The exact original call is:

    TSNE(n_components=2, random_state=42, verbose=1).fit_transform(emb)

i.e. scikit-learn defaults for perplexity (30), n_iter (1000) and init. This script
reproduces that exactly (the earlier version inserted a PCA-50 step + init="pca",
which changed the layout). It caches the projection and reports the silhouette score
of the 2-D projection by class (used by the verifier, not shown on the plot).

Outputs: reproduced/panelE_tsne/{proj.npy,meta.csv}, reproduced/Figure_2E_reproduced.png,
recheck/fig2E_separation.csv.

Run (env yeast_ml; CPU, a few minutes — t-SNE on full-dim embeddings) e.g. via tmux:
  reproduction/common/run_in_tmux.sh fig2e /tmp/fig2e.log \
    "/home/kchao10/miniconda3/envs/yeast_ml/bin/python3 reproduction/figure_02/recheck/build_2E_tsne.py"
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import time, glob
import numpy as np, pandas as pd, h5py
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import pyranges as pr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shorkie import config

SEED = 42
t0 = time.time()
F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"; RECHECK = F2 / "recheck"; RECHECK.mkdir(parents=True, exist_ok=True)

# published palette + plot order (faithful to 2_viz_clusters_LM.py's feature ordering)
PAL = {"Promoter": "tab:blue", "Intergenic region": "tab:orange",
       "Protein-coding gene": "tab:green", "tRNA": "tab:red",
       "Transposable element": "tab:purple"}
ORDER = ["Promoter", "Intergenic region", "Protein-coding gene", "tRNA", "Transposable element"]


def render(proj, mf, out_png):
    """Faithful render: figsize (8,7), s=3, no title, no grid, spines top/right off."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for f in ORDER:
        i = (mf["feature"] == f).values
        if i.any():
            ax.scatter(proj[i, 0], proj[i, 1], s=3, alpha=1, label=f, color=PAL.get(f, "gray"))
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(0.5, 1.16), loc="upper center", ncol=3, fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
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
    mf = mf.reset_index(drop=True)
    print(f"[{time.time()-t0:.0f}s] qualifying intervals: {emb.shape[0]} (16 chr); dim={emb.shape[1]}; "
          f"classes: {dict(mf['feature'].value_counts())}", flush=True)

    # FAITHFUL: t-SNE directly on full embeddings, no PCA, all points, sklearn defaults
    print(f"[{time.time()-t0:.0f}s] running t-SNE (no PCA) on {emb.shape[0]} pts ...", flush=True)
    t1 = time.time()
    proj = TSNE(n_components=2, random_state=SEED, verbose=1).fit_transform(emb)
    print(f"[{time.time()-t0:.0f}s] t-SNE done in {time.time()-t1:.0f}s", flush=True)

    labels = mf["feature"].values
    sil_2d = float(silhouette_score(proj, labels))
    per = pd.Series(silhouette_samples(proj, labels), index=labels).groupby(level=0).mean()

    od = RD / "panelE_tsne"; od.mkdir(parents=True, exist_ok=True)
    np.save(od / "proj.npy", proj)
    mf[["feature"]].to_csv(od / "meta.csv", index=False)
    render(proj, mf, RD / "Figure_2E_reproduced.png")

    with open(RECHECK / "fig2E_separation.csv", "w") as fcsv:
        fcsv.write("metric,value\n")
        fcsv.write(f"n_points,{len(proj)}\n")
        fcsv.write(f"n_classes,{mf['feature'].nunique()}\n")
        fcsv.write(f"silhouette_2d,{sil_2d:.4f}\n")
        for cls, v in per.items():
            fcsv.write(f"silhouette_2d[{cls}],{v:.4f}\n")
        for cls, c in mf["feature"].value_counts().items():
            fcsv.write(f"count[{cls}],{c}\n")
    print(f"[{time.time()-t0:.0f}s] classes={mf['feature'].nunique()} silhouette_2d={sil_2d:.3f}; "
          f"per-class:\n{per}", flush=True)
    print("[OK] saved proj/meta/png + recheck/fig2E_separation.csv")


if __name__ == "__main__":
    main()
