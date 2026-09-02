#!/usr/bin/env python3
"""Revision experiment 07, step 2 — quantify whether restricting the ISM bin sum to the
target gene changes the conclusions.

The question has two halves and this step answers both.

**Static saliency.** Per locus, the per-position reference-projected saliency is compared
between bin scopes by Pearson and Spearman correlation and by the overlap of the strongest
positions -- because what the figures show is *which* positions matter, not their absolute
magnitude, and a sequence-logo panel is decided by the top positions alone.

**The time course.** The comment ends "...and affect the comparison across induction time
points", which is the Figure-5C 8x8 pairwise Euclidean-distance heatmap between per-timepoint
ISM logos. Correlating static saliency would not test that at all, so this step rebuilds that
matrix under each bin scope, using the published recipe (mean-centre each timepoint's PWM
across the four bases, then take the Frobenius distance between timepoints -- verbatim
``reproduction/figure_05/recheck/fig05_lib.distance_heatmap``), and reports how similar the
resulting matrices are. That correlation is the number the whole concern turns on: if
the distance structure is preserved, the time-course conclusions stand regardless of scope.

**Contamination.** Also reported: the fraction of reference predicted coverage in the window
that falls outside the target gene -- how much room neighbours had to contribute at all.

Writes ``results/bin_scope_comparison.csv``, ``results/timecourse_distance_comparison.csv``
and two figures. CPU only, seconds.
"""
import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from shorkie import config

SCOPES = ["all_bins", "gene_body", "tss_window"]
NT = ["A", "C", "G", "T"]
TOP_N = 25

def parse_args():
    parser = argparse.ArgumentParser(description="Compare ISM bin scopes.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def mean_centred(grid, ti):
    """One timepoint's (L,4) logSED, mean-centred across the four bases (Methods Eq 19)."""
    g = np.asarray(grid, dtype=float)
    g = g[:, :, ti] if g.ndim == 3 else g
    return g - np.nanmean(g, axis=1, keepdims=True)

def projected(grid, ref_bases, ti=0):
    """Per-position reference-projected saliency (Methods Eq 20)."""
    centred = mean_centred(grid, ti)
    out = np.zeros(len(ref_bases))
    for i, b in enumerate(ref_bases):
        if b in NT:
            out[i] = centred[i, NT.index(b)]
    return out

def distance_matrix(grid, n_tp):
    """Figure-5C recipe: Frobenius distance between per-timepoint mean-centred PWMs."""
    norms = [mean_centred(grid, ti) for ti in range(n_tp)]
    D = np.zeros((n_tp, n_tp))
    for a in range(n_tp):
        for b in range(n_tp):
            D[a, b] = np.linalg.norm(np.nan_to_num(norms[a] - norms[b]).flatten())
    return D

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    ism_dir = out_dir / "ism"
    files = sorted(ism_dir.glob("*.npz")) if ism_dir.exists() else []
    if not files:
        sys.exit(f"error: no ISM npz under {ism_dir} -- run 1_gene_restricted_ism.py first")

    static_rows, tc_rows, per_locus, tc_panels = [], [], {}, []
    for f in files:
        z = np.load(f, allow_pickle=True)
        ref = [str(b) for b in z["ref_bases"]]
        timepoints = [int(t) for t in z["timepoints"]] if "timepoints" in z else [0]
        grids = {s: z[f"grid_{s}"] for s in SCOPES if f"grid_{s}" in z}
        contamination = float(z["coverage_fraction_outside_gene"])
        n_bins, n_gene = int(z["n_bins"]), int(len(z["gene_bin_idx"]))

        # --- static saliency agreement (at the first timepoint) -----------------------
        sal = {s: projected(g, ref, 0) for s, g in grids.items()}
        per_locus[f.stem] = sal
        for a, b in itertools.combinations(sal, 2):
            va, vb = sal[a], sal[b]
            m = np.isfinite(va) & np.isfinite(vb)
            if m.sum() < 10:
                continue
            top_a = set(np.argsort(-np.abs(va))[:TOP_N])
            top_b = set(np.argsort(-np.abs(vb))[:TOP_N])
            static_rows.append(dict(
                locus=f.stem, scope_a=a, scope_b=b, n_positions=int(m.sum()),
                gene_bins=n_gene, total_bins=n_bins,
                coverage_outside_gene=round(contamination, 4),
                pearson=round(float(pearsonr(va[m], vb[m])[0]), 4),
                spearman=round(float(spearmanr(va[m], vb[m])[0]), 4),
                top25_jaccard=round(len(top_a & top_b) / len(top_a | top_b), 3)))

        # --- time-course distance structure -------------------------------------------
        if len(timepoints) < 2:
            continue
        mats = {s: distance_matrix(g, len(timepoints)) for s, g in grids.items()}
        tc_panels.append((f.stem, timepoints, mats))
        iu = np.triu_indices(len(timepoints), k=1)
        for a, b in itertools.combinations(mats, 2):
            da, db = mats[a][iu], mats[b][iu]
            ok = np.isfinite(da) & np.isfinite(db)
            if ok.sum() < 3:
                continue
            tc_rows.append(dict(
                locus=f.stem, tf=str(z["tf"]) if "tf" in z else "",
                scope_a=a, scope_b=b, n_timepoints=len(timepoints),
                n_pairs=int(ok.sum()),
                pearson_of_distances=round(float(pearsonr(da[ok], db[ok])[0]), 4),
                spearman_of_distances=round(float(spearmanr(da[ok], db[ok])[0]), 4),
                mean_distance_a=round(float(np.mean(da[ok])), 4),
                mean_distance_b=round(float(np.mean(db[ok])), 4)))

    static = pd.DataFrame(static_rows)
    if static.empty:
        sys.exit("error: nothing to compare")
    static.to_csv(out_dir / "bin_scope_comparison.csv", index=False)
    print("=== static saliency agreement between bin scopes ===")
    print(static.to_string(index=False))
    key = static[(static.scope_a == "all_bins") & (static.scope_b == "gene_body")]
    if len(key):
        print(f"\nall_bins vs gene_body: median Pearson {key.pearson.median():.3f}, "
              f"median top-{TOP_N} Jaccard {key.top25_jaccard.median():.3f}, "
              f"median coverage outside the gene {100*key.coverage_outside_gene.median():.1f}%")

    if tc_rows:
        tc = pd.DataFrame(tc_rows)
        tc.to_csv(out_dir / "timecourse_distance_comparison.csv", index=False)
        print("\n=== Figure-5C time-course distance structure, per bin scope ===")
        print(tc.to_string(index=False))
        k = tc[(tc.scope_a == "all_bins") & (tc.scope_b == "gene_body")]
        if len(k):
            print(f"\nDistance-matrix agreement (all_bins vs gene_body): "
                  f"median Pearson {k.pearson_of_distances.median():.3f}. "
                  "High agreement means that neighbouring-gene coverage "
                  "affecting the time-course comparison does not change its conclusions.")
    else:
        print("\nNOTE: no locus carried more than one timepoint, so the time-course half of "
              "the time-course half was not tested. Re-run step 1 on a locus with a `tf`.",
              file=sys.stderr)

    # --- figures -------------------------------------------------------------------
    n = len(per_locus)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), squeeze=False)
    for ax, (name, sal) in zip(axes[:, 0], per_locus.items()):
        for s, colour in zip(SCOPES, ["#377eb8", "#e41a1c", "#4daf4a"]):
            if s in sal:
                ax.plot(sal[s], lw=1.0, color=colour, label=s, alpha=0.85)
        ax.set_title(f"{name} — per-position ISM saliency by output-bin scope",
                     fontsize=10, loc="left")
        ax.set_xlabel("position in promoter window (bp)")
        ax.set_ylabel("saliency")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "bin_scope_comparison.png", dpi=150)
    plt.close(fig)

    if tc_panels:
        rows = len(tc_panels)
        cols = max(len(m) for _, _, m in tc_panels)
        fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.4 * rows), squeeze=False)
        for r, (name, tps, mats) in enumerate(tc_panels):
            for c, scope in enumerate(SCOPES):
                ax = axes[r][c]
                if scope not in mats:
                    ax.axis("off"); continue
                im = ax.imshow(mats[scope], cmap="viridis")
                ax.set_xticks(range(len(tps))); ax.set_yticks(range(len(tps)))
                ax.set_xticklabels([f"T{t}" for t in tps], fontsize=6, rotation=90)
                ax.set_yticklabels([f"T{t}" for t in tps], fontsize=6)
                ax.set_title(f"{name} · {scope}", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Figure-5C time-course distance structure under each output-bin scope",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / "timecourse_distance_matrices.png", dpi=150)
        plt.close(fig)
        print(f"\nwrote {out_dir/'timecourse_distance_matrices.png'}")
    print(f"wrote {out_dir/'bin_scope_comparison.csv'}")

if __name__ == "__main__":
    main()
