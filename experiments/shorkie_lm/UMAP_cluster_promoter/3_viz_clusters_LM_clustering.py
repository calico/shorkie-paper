#!/usr/bin/env python

import os
import glob
import numpy as np
import h5py
import pandas as pd
from optparse import OptionParser
import warnings

# For dimensionality reduction using t-SNE and clustering via K-Means
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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
                      help="Prefix for output plots and files.")
    parser.add_option("--annotate_points", dest="annotate_points", default=0, type=int,
                      help="Number of points to annotate on the plots (0 disables).")
    parser.add_option("--groups", dest="groups", default="gene,intergenic",
                      help="Comma-separated list of feature groups to plot (e.g. 'gene,intergenic'). Empty => use all.")
    parser.add_option("--n_clusters", dest="n_clusters", default=5, type=int,
                      help="Number of clusters for K-Means on the t-SNE embeddings.")
    (opts, args) = parser.parse_args()

    # 1) Find and read HDF5 files matching the pattern
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
                    meta_array = np.array(h5f["metadata"])
                    meta_list.append(meta_array)
                else:
                    warnings.warn(f"'metadata' not found in file {file}. Skipping metadata.")
                for ds_name in h5f.keys():
                    if ds_name.startswith("embeddings_"):
                        arr = np.array(h5f[ds_name])
                        embeddings_data.setdefault(ds_name, []).append(arr)
        except OSError as e:
            warnings.warn(f"Could not open file '{file}': {e}. Skipping.")
            continue

    # 2) Build combined metadata DataFrame
    if meta_list:
        meta_array = np.concatenate(meta_list, axis=0)
        meta_str = np.char.decode(meta_array.astype(np.bytes_), 'utf-8')
        meta_df = pd.DataFrame(meta_str, columns=["chrom","start","end","strand","feature","gene_id"])
        meta_df["start"] = pd.to_numeric(meta_df["start"], errors="coerce")
        meta_df["end"] = pd.to_numeric(meta_df["end"], errors="coerce")
        meta_df = meta_df.reset_index(drop=True)
    else:
        print("No metadata found in any file. Proceeding without metadata.")
        meta_df = pd.DataFrame([], columns=["chrom","start","end","strand","feature","gene_id"])

    meta_df["original_feature"] = meta_df["feature"]

    # 3) Read GTF and merge gene_biotype
    if not meta_df.empty and os.path.exists(opts.gtf_file):
        gtf_pr = pr.read_gtf(opts.gtf_file)
        gtf_df = gtf_pr.as_df()
        if "gene_biotype" in gtf_df.columns:
            gene_info = gtf_df[["gene_id","gene_biotype"]].drop_duplicates()
            merged = meta_df.merge(gene_info, on="gene_id", how="left")
            mask = (merged["feature"]=="gene") & merged["gene_biotype"].notnull()
            merged.loc[mask,"feature"] = merged.loc[mask,"gene_biotype"]
            meta_df = merged
        else:
            warnings.warn("No 'gene_biotype' in GTF. Skipping merge.")

    # 4) Filter groups & remap labels
    if opts.groups.strip():
        groups = [g.strip() for g in opts.groups.split(',')]
        filt = meta_df["original_feature"].isin(groups)
    else:
        filt = np.ones(len(meta_df), dtype=bool)

    meta_df = meta_df.loc[filt].reset_index(drop=True)
    mapping = {
        "protein_coding":"Protein-coding gene",
        "intergenic":"Intergenic region",
        "snoRNA":"snoRNA","tRNA":"tRNA",
        "transposable_element":"Transposable element",
        "ncRNA":"Non-coding RNA","pseudogene":"Pseudogene",
        "promoter":"Promoter"
    }
    meta_df["feature"] = meta_df["feature"].map(lambda x: mapping.get(x,x))

    # Ensure output dir
    out_dir = os.path.dirname(opts.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 5) Loop datasets: t-SNE, clustering, plotting, saving
    for ds_name, arrs in embeddings_data.items():
        emb_all = np.concatenate(arrs, axis=0)
        if emb_all.shape[0] != len(filt):
            warnings.warn(f"Mismatch rows: {emb_all.shape[0]} vs {len(filt)}. Skipping.")
            continue
        emb = emb_all[filt]
        if emb.shape[0]==0:
            continue

        # t-SNE
        tsne = TSNE(n_components=opts.n_components, random_state=42, verbose=1)
        emb2d = tsne.fit_transform(emb)

        # K-Means clustering
        kmeans = KMeans(n_clusters=opts.n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(emb2d)
        meta_df['cluster'] = labels

        # Save full cluster map with biotype
        csv_out = f"{opts.out_prefix}_{ds_name}_clusters.csv"
        cols_to_save = ['gene_id', 'gene_biotype', 'cluster'] if 'gene_biotype' in meta_df.columns else ['gene_id', 'cluster']
        meta_df[cols_to_save].to_csv(csv_out, index=False)

        # Save each cluster's gene list with biotype
        for c in range(opts.n_clusters):
            cluster_df = meta_df.loc[meta_df['cluster']==c, cols_to_save]
            txt_path = f"{opts.out_prefix}_{ds_name}_cluster_{c}_genes.tsv"
            cluster_df.to_csv(txt_path, sep='\t', index=False)

        # Plot clusters by color
        plt.figure(figsize=(8,6))
        for c in range(opts.n_clusters):
            idx = (labels==c)
            plt.scatter(
                emb2d[idx,0], emb2d[idx,1],
                s=5, alpha=0.7, label=f"Cluster {c}"
            )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(bbox_to_anchor=(0.5,1.15), loc='upper center', ncol=min(opts.n_clusters,4))
        plt.tight_layout()
        plt.savefig(f"{opts.out_prefix}_{ds_name}_kmeans_clusters.png", dpi=300)
        plt.show()

        # Original feature-colored plot
        plt.figure(figsize=(8.2,7))
        feats = meta_df['feature'].unique()
        cmap = plt.cm.get_cmap("tab20", len(feats))
        for i,feat in enumerate(feats):
            sel = meta_df['feature']==feat
            col = "#572D0C" if feat=="Transposable element" else cmap(i)
            plt.scatter(emb2d[sel,0], emb2d[sel,1], s=3, alpha=0.7, label=feat, color=col)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(bbox_to_anchor=(0.5,1.16), loc='upper center', ncol=3, fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{opts.out_prefix}_{ds_name}_features.png", dpi=300)
        plt.show()

        # Optional annotation
        if opts.annotate_points>0:
            np.random.seed(42)
            idxs = np.random.choice(len(emb2d), size=opts.annotate_points, replace=False)
            plt.figure(figsize=(8,6))
            plt.scatter(emb2d[:,0], emb2d[:,1], s=3, c='gray', alpha=0.6)
            for i in idxs:
                x,y = emb2d[i]
                label = meta_df.iloc[i]['gene_id']
                plt.text(x,y,label,fontsize=6)
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            plt.savefig(f"{opts.out_prefix}_{ds_name}_annotated.png", dpi=300)
            plt.show()

    print("All done! Check output files.")

if __name__ == "__main__":
    main()
