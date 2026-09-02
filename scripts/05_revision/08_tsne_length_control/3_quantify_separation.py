#!/usr/bin/env python3
"""Revision experiment 08, step 3 — quantify how much Figure 2E class separation survives
once element length is controlled.

Replaces "the projection looks structured" with a number, for each of the four embedding
schemes from step 2 and against the length-only baseline from step 1:

  * 5-fold-CV k-NN classification accuracy and macro-F1 in the FULL embedding space (not
    the 2-D projection -- t-SNE coordinates are not a faithful metric space, and the
    published silhouette was computed on them);
  * silhouette score of the 2-D t-SNE projection, so the published number stays comparable;
  * adjusted Rand index of k-means (k=5) against the true classes.

The interpretation is a comparison, not an absolute: an embedding that beats the
length-only baseline is carrying information beyond length. One that does not, is not --
whichever way it falls, and it should be reported that way.

Writes ``results/separation_metrics.csv`` and ``results/tsne_panel.png``.
CPU only, a few minutes (t-SNE dominates).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shorkie import config

SCHEMES = ["published", "masked_pool", "length_matched", "residualised"]
SCHEME_LABEL = {
    "published": "published\n(full-window mean pool)",
    "masked_pool": "masked pooling\n(real positions only)",
    "length_matched": "length-matched\n(fixed 500 bp)",
    "residualised": "length-residualised",
}
PALETTE = {"Promoter": "tab:blue", "Intergenic region": "tab:orange",
           "Protein-coding gene": "tab:green", "Transposable element": "tab:red",
           "tRNA": "tab:purple"}
ORDER = ["Promoter", "Intergenic region", "Protein-coding gene",
         "Transposable element", "tRNA"]

def parse_args():
    parser = argparse.ArgumentParser(description="Quantify Figure 2E class separation.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def evaluate(X, y, seed):
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if len(np.unique(y)) < 2 or len(X) < 50:
        return None
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    proj = TSNE(n_components=2, random_state=seed, init="pca",
                perplexity=min(30, max(5, len(X) // 4))).fit_transform(
                    StandardScaler().fit_transform(X))
    sil = silhouette_score(proj, y)
    km = KMeans(n_clusters=len(np.unique(y)), n_init=10, random_state=seed).fit_predict(proj)
    return dict(n=len(X), knn_accuracy=float(acc.mean()), knn_accuracy_sd=float(acc.std()),
                knn_macro_f1=float(f1.mean()), silhouette_2d=float(sil),
                kmeans_ari=float(adjusted_rand_score(y, km))), proj, y

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    baseline_csv = out_dir / "length_only_baseline.csv"
    baseline = (pd.read_csv(baseline_csv).iloc[0].to_dict()
                if baseline_csv.exists() else None)

    rows, projections = [], {}
    for scheme in SCHEMES:
        path = out_dir / f"embeddings_{scheme}.npz"
        if not path.exists():
            print(f"SKIPPED: {path} not found -- run 2_embed_variants.py first",
                  file=sys.stderr)
            continue
        z = np.load(path, allow_pickle=True)
        out = evaluate(np.asarray(z["embedding"], dtype=float),
                       np.asarray(z["feature"]), args.seed)
        if out is None:
            print(f"SKIPPED: {scheme} (too few usable rows)", file=sys.stderr)
            continue
        metrics, proj, y = out
        projections[scheme] = (proj, y)
        rows.append(dict(scheme=scheme, **{k: round(v, 4) for k, v in metrics.items()}))

    if not rows:
        sys.exit("error: no embeddings to evaluate -- run 2_embed_variants.py first")

    df = pd.DataFrame(rows)
    if baseline:
        df = pd.concat([df, pd.DataFrame([dict(
            scheme="length only (1-D baseline)", n=None,
            knn_accuracy=baseline.get("cv_accuracy"),
            knn_accuracy_sd=baseline.get("cv_accuracy_sd"),
            knn_macro_f1=baseline.get("cv_macro_f1"),
            silhouette_2d=None, kmeans_ari=None)])], ignore_index=True)
    df.to_csv(out_dir / "separation_metrics.csv", index=False)
    print(df.to_string(index=False))

    if baseline and "published" in set(df.scheme):
        pub = df[df.scheme == "published"].iloc[0]
        print(f"\npublished embedding kNN accuracy {pub.knn_accuracy:.3f} vs "
              f"length-only baseline {baseline['cv_accuracy']:.3f}")
        for scheme in ("masked_pool", "length_matched", "residualised"):
            hit = df[df.scheme == scheme]
            if len(hit):
                print(f"  {scheme:15s} {hit.iloc[0].knn_accuracy:.3f}")
        print("An embedding above the baseline carries information beyond length; one at "
              "or below it does not.")

    if projections:
        n = len(projections)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
        for ax, (scheme, (proj, y)) in zip(axes[0], projections.items()):
            for f in ORDER:
                m = y == f
                if m.any():
                    ax.scatter(proj[m, 0], proj[m, 1], s=6, alpha=0.8,
                               color=PALETTE.get(f, "gray"), label=f)
            hit = df[df.scheme == scheme]
            sub = (f"kNN {hit.iloc[0].knn_accuracy:.2f} · sil {hit.iloc[0].silhouette_2d:.2f}"
                   if len(hit) else "")
            ax.set_title(f"{SCHEME_LABEL.get(scheme, scheme)}\n{sub}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0][0].legend(fontsize=7, loc="upper left", markerscale=2)
        fig.suptitle("Figure 2E class separation under length-controlled embedding schemes",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_dir / "tsne_panel.png", dpi=150)
        plt.close(fig)
        print(f"\nwrote {out_dir/'tsne_panel.png'}")
    print(f"wrote {out_dir/'separation_metrics.csv'}")

if __name__ == "__main__":
    main()
