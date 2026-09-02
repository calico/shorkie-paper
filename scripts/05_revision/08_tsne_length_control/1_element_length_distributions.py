#!/usr/bin/env python3
"""Revision experiment 08, step 1 — measure the element-length confound in the Figure 2E
t-SNE.

The concern: the five element classes have very different length distributions, but they are
extracted and zero-padded to a common input length and then mean-pooled across the full
padded sequence. The fraction of real sequence entering each mean embedding therefore differs
substantially by class, which could produce the observed separation independently of any
learned regulatory feature.

The mechanism is visible directly in the code. In
``scripts/04_analysis/shorkie_lm/umap_cluster_promoter/1_predict_seqs_LM.py`` each interval
is centre-padded with ``N`` to 16,384 bp (lines 117-119) and the layer activations are then
mean-pooled across the FULL padded axis (line 173). So every embedding is scaled by roughly
(real length / 16,384), and that factor is a property of the class, not of its regulatory
content.

This step establishes the size of the problem before any GPU work: the per-class length
distribution, the fraction of the padded window that is real sequence, and -- the decisive
number -- how well the five classes can be told apart from **length alone**. If a one-
dimensional length feature already separates them, then t-SNE separation of a 16,384-long
mean-pooled embedding is not evidence of learned structure on its own.

Writes ``results/element_lengths.csv``, ``results/length_only_baseline.csv`` and
``results/element_lengths.png``. CPU only, seconds; needs only the GTF.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score

from shorkie import config

SEQ_LEN = 16384
PROMOTER_BP = 500          # 500 bp immediately upstream of each start codon (Methods)
# gene_biotype -> the five display classes of Figure 2E (umap_cluster_promoter/2_viz_clusters_LM.py)
BIOTYPE_CLASS = {
    "protein_coding": "Protein-coding gene",
    "tRNA": "tRNA",
    "transposable_element": "Transposable element",
}
PALETTE = {"Promoter": "tab:blue", "Intergenic region": "tab:orange",
           "Protein-coding gene": "tab:green", "Transposable element": "tab:red",
           "tRNA": "tab:purple"}
ORDER = ["Promoter", "Intergenic region", "Protein-coding gene",
         "Transposable element", "tRNA"]
# Silhouette scores of the published 2-D projection, from
# reproduction/figure_02/recheck/fig2E_separation.csv -- reported alongside so the
# length ordering and the separation ordering can be compared directly.
PUBLISHED_SILHOUETTE = {
    "tRNA": 0.8499, "Transposable element": 0.6359, "Promoter": 0.1884,
    "Protein-coding gene": 0.0490, "Intergenic region": -0.0946,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure the element-length confound behind Figure 2E.")
    parser.add_argument("--gtf", default=None, help="R64 GTF [default: config genome.gtf]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_genes(gtf_path):
    """Gene records with biotype, plus per-chromosome extents for the intergenic complement."""
    genes, chrom_end = [], defaultdict(int)
    for line in open(gtf_path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        chrom_end[f[0]] = max(chrom_end[f[0]], int(f[4]))
        if f[2] != "gene":
            continue
        m = re.search(r'gene_biotype "([^"]+)"', f[8])
        genes.append(dict(chrom=f[0], start=int(f[3]) - 1, end=int(f[4]), strand=f[6],
                          biotype=m.group(1) if m else "unknown"))
    return pd.DataFrame(genes), dict(chrom_end)

def intergenic_intervals(genes, chrom_end):
    """Complement of the gene spans, per chromosome — the Methods' third interval class."""
    out = []
    for chrom, g in genes.groupby("chrom"):
        spans = sorted(zip(g.start, g.end))
        cursor = 0
        merged = []
        for s, e in spans:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        for s, e in merged:
            if s > cursor:
                out.append(dict(chrom=chrom, start=cursor, end=s))
            cursor = max(cursor, e)
        if cursor < chrom_end.get(chrom, 0):
            out.append(dict(chrom=chrom, start=cursor, end=chrom_end[chrom]))
    return pd.DataFrame(out)

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    gtf = Path(args.gtf) if args.gtf else config.path("genome.gtf")
    if gtf is None or not Path(gtf).exists():
        sys.exit(f"error: --gtf not resolved ({gtf}). Fetch the genome with "
                 "`bash data/download.sh --genome -u PROJECT` or pass --gtf.")

    genes, chrom_end = load_genes(gtf)
    print(f"gtf   : {gtf}\ngenes : {len(genes)}", flush=True)

    rows = []
    for _, g in genes.iterrows():
        cls = BIOTYPE_CLASS.get(g.biotype)
        if cls:
            rows.append(dict(feature=cls, length=g.end - g.start))
    # Promoters are a fixed 500 bp by construction — that is itself part of the confound.
    n_prom = int((genes.biotype == "protein_coding").sum())
    rows += [dict(feature="Promoter", length=PROMOTER_BP)] * n_prom
    for _, r in intergenic_intervals(genes, chrom_end).iterrows():
        rows.append(dict(feature="Intergenic region", length=r.end - r.start))

    df = pd.DataFrame(rows)
    df = df[df.length > 0]
    df["real_fraction"] = (df.length.clip(upper=SEQ_LEN) / SEQ_LEN)

    summ = (df.groupby("feature")
.agg(n=("length", "size"), median_bp=("length", "median"),
                   mean_bp=("length", "mean"), p10_bp=("length", lambda s: s.quantile(0.10)),
                   p90_bp=("length", lambda s: s.quantile(0.90)),
                   median_real_fraction=("real_fraction", "median"))
.reindex([c for c in ORDER if c in set(df.feature)]))
    summ["median_pct_of_window"] = (100 * summ.median_real_fraction).round(2)
    summ["published_silhouette_2d"] = [PUBLISHED_SILHOUETTE.get(c) for c in summ.index]
    summ = summ.round(3)
    summ.to_csv(out_dir / "element_lengths.csv")
    print("\n" + summ.to_string())

    # --- the decisive control: how separable are the classes from LENGTH ALONE? -------
    rng = np.random.default_rng(args.seed)
    per_class = min(df.feature.value_counts().min(), 2000)
    balanced = (df.groupby("feature", group_keys=False)
.apply(lambda g: g.sample(n=per_class, random_state=args.seed)))
    X = np.log10(balanced[["length"]].to_numpy() + 1)
    y = balanced.feature.to_numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    chance = 1.0 / balanced.feature.nunique()

    base = pd.DataFrame([dict(
        feature_set="log10(element length) only", n_per_class=per_class,
        n_classes=int(balanced.feature.nunique()),
        cv_accuracy=round(float(acc.mean()), 4), cv_accuracy_sd=round(float(acc.std()), 4),
        cv_macro_f1=round(float(f1.mean()), 4), chance_accuracy=round(chance, 4))])
    base.to_csv(out_dir / "length_only_baseline.csv", index=False)
    print(f"\nLength-only 5-class baseline: accuracy {acc.mean():.3f} +/- {acc.std():.3f}, "
          f"macro-F1 {f1.mean():.3f} (chance {chance:.3f})")
    print(f"  balanced at {per_class} intervals per class (capped by the smallest class), "
          f"{per_class * balanced.feature.nunique()} total, 5-fold CV")
    print("This is the bar the embedding has to clear before its separation can be "
          "attributed to learned sequence features.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    present = [c for c in ORDER if c in set(df.feature)]
    ax1.boxplot([df[df.feature == c].length for c in present], labels=present,
                showfliers=False, patch_artist=True,
                boxprops=dict(alpha=0.7), medianprops=dict(color="black"))
    for patch, c in zip(ax1.patches, present):
        patch.set_facecolor(PALETTE[c])
    ax1.set_yscale("log"); ax1.set_ylabel("element length (bp, log)")
    ax1.axhline(SEQ_LEN, color="0.3", ls="--", lw=1)
    ax1.text(0.02, SEQ_LEN * 1.15, "model input window (16,384 bp)", fontsize=8,
             transform=ax1.get_yaxis_transform())
    ax1.set_title("A · the five classes have very different lengths",
                  fontsize=11, loc="left")
    ax1.tick_params(axis="x", rotation=25)

    sil = [PUBLISHED_SILHOUETTE.get(c, np.nan) for c in present]
    med = [df[df.feature == c].length.median() for c in present]
    ax2.scatter(med, sil, s=90, c=[PALETTE[c] for c in present], edgecolor="white")
    for c, m, s in zip(present, med, sil):
        ax2.annotate(c, (m, s), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("median element length (bp, log)")
    ax2.set_ylabel("published 2-D silhouette (Fig 2E)")
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_title("B · the best-separated classes are the length extremes",
                  fontsize=11, loc="left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "element_lengths.png", dpi=160)
    plt.close(fig)
    print(f"\nwrote {out_dir/'element_lengths.csv'}")
    print(f"wrote {out_dir/'length_only_baseline.csv'}")
    print(f"wrote {out_dir/'element_lengths.png'}")

if __name__ == "__main__":
    main()
