#!/usr/bin/env python

import os
import glob
import numpy as np
import h5py
import pandas as pd
from optparse import OptionParser
import warnings

# For dimensionality reduction using t-SNE
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import pyranges as pr

def main():
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("--embedding_pattern", dest="embedding_pattern", default="embeddings*.h5",
                      help="Glob pattern to match HDF5 files with stored embeddings.")
    parser.add_option("--gtf_file", dest="gtf_file", default="annotations.gtf",
                      help="Path to GTF file to retrieve gene_biotype info.")
    parser.add_option("--n_components", dest="n_components", default=2, type=int,
                      help="Number of components for t-SNE.")
    parser.add_option("--out_prefix", dest="out_prefix", default="embedding_viz",
                      help="Prefix for output plots.")
    parser.add_option("--annotate_points", dest="annotate_points", default=0, type=int,
                      help="Number of points to annotate on the plots (0 disables).")
    parser.add_option("--groups", dest="groups", default="",
                      help="Comma-separated list of feature groups to include. Empty => include all.")
    parser.add_option("--exclude_groups", dest="exclude_groups",
                      default="snoRNA,ncRNA,pseudogene,snRNA,rRNA",
                      help="Comma-separated list of feature groups to drop entirely.")
    (opts, args) = parser.parse_args()

    ###################################################
    # 1) Find and read HDF5 files matching the pattern
    ###################################################
    files = glob.glob(opts.embedding_pattern)
    if not files:
        print("No files found matching pattern:", opts.embedding_pattern)
        return
    print("Found files:\n", "\n ".join(files))

    embeddings_data = {}
    meta_list = []

    for file in files:
        try:
            with h5py.File(file, "r") as h5f:
                if "metadata" in h5f:
                    meta_list.append(np.array(h5f["metadata"]))
                else:
                    warnings.warn(f"'metadata' not found in file {file}.")
                for ds in h5f.keys():
                    if ds.startswith("embeddings_"):
                        arr = np.array(h5f[ds])
                        embeddings_data.setdefault(ds, []).append(arr)
        except OSError as e:
            warnings.warn(f"Could not open {file}: {e}")

    ###################################################
    # 2) Build combined metadata DataFrame
    ###################################################
    if meta_list:
        meta_arr = np.concatenate(meta_list, axis=0)
        meta_str = np.char.decode(meta_arr.astype(np.bytes_), 'utf-8')
        meta_df = pd.DataFrame(meta_str, columns=["chrom","start","end","strand","feature","gene_id"])
        meta_df[["start","end"]] = meta_df[["start","end"]].apply(pd.to_numeric, errors="coerce")
    else:
        meta_df = pd.DataFrame([], columns=["chrom","start","end","strand","feature","gene_id"])
    meta_df.reset_index(drop=True, inplace=True)
    meta_df["original_feature"] = meta_df["feature"]

    ###################################################
    # 3) Read GTF, extract gene_biotype, and merge
    ###################################################
    if not meta_df.empty and os.path.exists(opts.gtf_file):
        print("Reading GTF for gene_biotype …")
        gtf_df = pr.read_gtf(opts.gtf_file).as_df()
        if "gene_biotype" in gtf_df.columns:
            gene_info = gtf_df[["gene_id","gene_biotype"]].drop_duplicates()
            merged = meta_df.merge(gene_info, on="gene_id", how="left")
            mask_g = (merged["feature"]=="gene") & merged["gene_biotype"].notna()
            merged.loc[mask_g, "feature"] = merged.loc[mask_g, "gene_biotype"]
            meta_df = merged
        else:
            warnings.warn("No 'gene_biotype' in GTF; skipping merge.")
    else:
        print("Skipping GTF merge.")

    ###################################################
    # 4) Filter by include/exclude groups
    ###################################################
    # include
    if opts.groups.strip():
        inc = [g.strip() for g in opts.groups.split(',')]
        mask = meta_df["original_feature"].isin(inc)
    else:
        mask = np.ones(len(meta_df), dtype=bool)
    # exclude
    if opts.exclude_groups.strip():
        exc = [g.strip() for g in opts.exclude_groups.split(',')]
        mask &= ~meta_df["feature"].isin(exc)
    meta_df_filt = meta_df.loc[mask].copy()
    print(f"After filter: {len(meta_df_filt)} rows; features = {meta_df_filt['feature'].unique()}")

    ###################################################
    # 5) Map features to nice names
    ###################################################
    nice = {
        "protein_coding":"Protein-coding gene",
        "intergenic":"Intergenic region",
        "tRNA":"tRNA",
        "transposable_element":"Transposable element",
        "promoter":"Promoter",
        # others will pass through
    }
    meta_df_filt["feature"] = meta_df_filt["feature"].map(lambda x: nice.get(x, x))

    ###################################################
    # 6) Prepare output dir
    ###################################################
    out_dir = os.path.dirname(opts.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    ###################################################
    # 7) t-SNE + plotting with fixed palette
    ###################################################
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for ds_name, arrs in embeddings_data.items():
        all_emb = np.concatenate(arrs, axis=0)
        if all_emb.shape[0] != len(meta_df):
            warnings.warn(f"Size mismatch for {ds_name}; skipping.")
            continue
        emb = all_emb[mask]
        if emb.size == 0:
            continue

        print(f"Running t-SNE on {ds_name} ({emb.shape})…")
        tsne = TSNE(n_components=opts.n_components, random_state=42, verbose=1)
        proj = tsne.fit_transform(emb)

        feats = meta_df_filt["feature"].unique()
        # assign each feature one of the five colors (wrap if >5)
        color_map = {f: palette[i % len(palette)] for i,f in enumerate(feats)}

        plt.figure(figsize=(8,7))
        for f in feats:
            idx = meta_df_filt["feature"] == f
            plt.scatter(proj[idx,0], proj[idx,1],
                        s=3, alpha=1,
                        label=f,
                        color=color_map[f])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(bbox_to_anchor=(0.5,1.16), loc='upper center', ncol=3, fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{opts.out_prefix}_{ds_name}_features.png", dpi=300)
        plt.show()

        if opts.annotate_points > 0:
            np.random.seed(42)
            pts = np.random.choice(len(proj), size=opts.annotate_points, replace=False)
            plt.figure(figsize=(8,6))
            plt.scatter(proj[:,0], proj[:,1], s=3, alpha=0.6, c='gray')
            for i in pts:
                x,y = proj[i]
                lbl = meta_df_filt.iloc[i]["gene_id"]
                plt.text(x, y, lbl, fontsize=6)
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            plt.savefig(f"{opts.out_prefix}_{ds_name}_annotated.png", dpi=300)
            plt.show()

    print("Done — check your plots!")

if __name__ == "__main__":
    main()
