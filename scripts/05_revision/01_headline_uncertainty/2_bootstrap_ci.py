#!/usr/bin/env python3
"""Revision experiment 01, step 2 — confidence intervals on the Figure 3 headline
metrics, defined using the eight cross-validation test folds.

Consumes ``results/fold_metrics.csv`` from step 1 and, for every published panel
statistic, reports four things:

  1. ``point``          the estimate computed EXACTLY as the published figure code
                        computes it (aggregate across folds per track/gene first,
                        then aggregate across tracks/genes).
  2. ``per-fold``       the same aggregation applied WITHIN each fold -> eight
                        values -> mean, SD and a 95% t interval on n=8. This is
                        the interval this experiment exists to provide.
  3. ``bootstrap``      95% percentile intervals from resampling units (tracks or
                        genes) with replacement, and from a hierarchical resample
                        of folds-then-units.
  4. ``paired delta``   Shorkie minus each random-init baseline, computed per fold,
                        with a two-sided Wilcoxon signed-rank test (n=8; the
                        smallest attainable p is 0.0078).

The panel recipes are ported from ``reproduction/figure_03/recheck/build_3C_violin.py``
and ``build_3DEFG_scatter.py`` -- including their two filters: gene-level group means
are taken over positive values only, and Figure 3G first drops the bottom 10% of genes
by ``coverage_norm``.

Writes ``results/headline_ci.csv``, ``results/paired_fold_deltas.csv`` and
``results/verify_revision_01.csv``.

CPU only, ~1 min at the default 2000 bootstrap replicates.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from shorkie import config

sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa: E402

# Published panel recipes. `agg_unit` collapses the fold axis for one track/gene;
# `agg_group` collapses across tracks/genes. Both come from the figure-3 builders.
PANELS = {
    "3C": dict(level="bin", metric="pearsonr", agg_unit="median", agg_group="median",
               pos_only=False, cov_filter=False,
               track_types=["RNA-Seq", "1000 strains RNA-Seq", "ChIP-MNase", "ChIP-exo"],
               title="bin-level Pearson's R (median)"),
    "3D": dict(level="bin", metric="pearsonr", agg_unit="mean", agg_group="mean",
               pos_only=False, cov_filter=False,
               track_types=["RNA-Seq", "1000 strains RNA-Seq"],
               title="bin-level Pearson's R (mean)"),
    "3E": dict(level="gene_track", metric="pearsonr", agg_unit="mean", agg_group="mean",
               pos_only=True, cov_filter=False,
               track_types=["RNA-Seq", "1000-RNA-seq"],
               title="gene-level Pearson's R"),
    "3F": dict(level="gene_track", metric="pearsonr_norm", agg_unit="mean", agg_group="mean",
               pos_only=True, cov_filter=False,
               track_types=["RNA-Seq", "1000-RNA-seq"],
               title="gene-level Pearson's R (quantile-normalised)"),
    "3G": dict(level="gene", metric="pearsonr_gene", agg_unit="mean", agg_group="mean",
               pos_only=True, cov_filter=True,
               track_types=["RNA-Seq", "1000-RNA-seq"],
               title="within-gene Pearson's R"),
}

MODELS = ["Shorkie", "Shorkie_Random_Init", "Shorkie_Random_Init_untuned"]

# Published values the manuscript / Figure 3 report, used as PASS/FAIL anchors.
PUBLISHED = {
    ("3C", "RNA-Seq", "Shorkie"): 0.78,
    ("3C", "RNA-Seq", "Shorkie_Random_Init_untuned"): 0.67,
    ("3E", "RNA-Seq", "Shorkie"): 0.88,
    ("3E", "RNA-Seq", "Shorkie_Random_Init_untuned"): 0.74,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for the Figure 3 headline metrics.")
    parser.add_argument("--in_csv", default=None, help="fold_metrics.csv from step 1")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip the published-anchor checks (for re-use on models that "
                             "are not expected to reproduce the published numbers)")
    return parser.parse_args()

def _agg(values, how):
    if len(values) == 0:
        return np.nan
    return float(np.median(values)) if how == "median" else float(np.mean(values))

def build_matrix(df, spec):
    """Return (values, weights, units) where values is a units x folds float array."""
    piv = df.pivot_table(index="unit", columns="fold", values="value", aggfunc="mean")
    piv = piv.dropna(how="all")
    weights = None
    if spec["cov_filter"]:
        w = df.pivot_table(index="unit", columns="fold", values="weight", aggfunc="mean")
        weights = w.reindex(piv.index).mean(axis=1).to_numpy(dtype=float)
    return piv.to_numpy(dtype=float), weights, piv.index.to_numpy()

def group_stat(unit_values, weights, spec, cov_threshold=None):
    """Collapse a per-unit vector to the published group statistic."""
    v = np.asarray(unit_values, dtype=float)
    keep = np.isfinite(v)
    if spec["cov_filter"] and weights is not None and cov_threshold is not None:
        keep &= np.isfinite(weights) & (weights > cov_threshold)
    v = v[keep]
    if spec["pos_only"]:
        v = v[v > 0]
    return _agg(v, spec["agg_group"])

def collapse_folds(mat, how):
    """Collapse the fold axis of a units x folds matrix, ignoring missing folds.

    A bootstrap resample can draw a unit that is missing from every resampled fold,
    leaving an all-NaN row; numpy warns on that and returns NaN, which is the answer
    we want, so the warning is suppressed rather than the row special-cased.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(mat, axis=1) if how == "median" else np.nanmean(mat, axis=1)

def analyse(df, spec, rng, n_boot):
    mat, weights, _ = build_matrix(df, spec)
    if mat.size == 0:
        return None
    cov_threshold = None
    if spec["cov_filter"] and weights is not None and np.isfinite(weights).any():
        cov_threshold = float(np.nanpercentile(weights[np.isfinite(weights)], 10))

    # 1. published recipe: collapse folds per unit, then across units
    point = group_stat(collapse_folds(mat, spec["agg_unit"]), weights, spec, cov_threshold)

    # 2. per-fold statistic
    per_fold = np.array([
        group_stat(mat[:, j], weights, spec, cov_threshold) for j in range(mat.shape[1])
    ], dtype=float)
    pf = per_fold[np.isfinite(per_fold)]
    n = len(pf)
    if n > 1:
        se = float(np.std(pf, ddof=1) / np.sqrt(n))
        half = float(stats.t.ppf(0.975, n - 1) * se)
        fold_lo, fold_hi = float(pf.mean()) - half, float(pf.mean()) + half
    else:
        se = half = np.nan
        fold_lo = fold_hi = np.nan

    # 3. bootstraps
    n_units, n_folds = mat.shape
    boot_units, boot_hier = [], []
    for _ in range(n_boot):
        ui = rng.integers(0, n_units, n_units)
        w_u = weights[ui] if weights is not None else None
        boot_units.append(group_stat(collapse_folds(mat[ui], spec["agg_unit"]), w_u, spec, cov_threshold))
        fi = rng.integers(0, n_folds, n_folds)
        boot_hier.append(group_stat(collapse_folds(mat[np.ix_(ui, fi)], spec["agg_unit"]),
                                    w_u, spec, cov_threshold))
    bu = np.array(boot_units, dtype=float); bu = bu[np.isfinite(bu)]
    bh = np.array(boot_hier, dtype=float); bh = bh[np.isfinite(bh)]

    return dict(
        point=point, n_units=n_units, n_folds=n,
        per_fold=pf, fold_mean=float(pf.mean()) if n else np.nan,
        fold_sd=float(np.std(pf, ddof=1)) if n > 1 else np.nan,
        fold_se=se, fold_ci_lo=fold_lo, fold_ci_hi=fold_hi,
        unit_ci_lo=float(np.percentile(bu, 2.5)) if bu.size else np.nan,
        unit_ci_hi=float(np.percentile(bu, 97.5)) if bu.size else np.nan,
        hier_ci_lo=float(np.percentile(bh, 2.5)) if bh.size else np.nan,
        hier_ci_hi=float(np.percentile(bh, 97.5)) if bh.size else np.nan,
    )

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_csv = Path(args.in_csv) if args.in_csv else out_dir / "fold_metrics.csv"
    if not in_csv.exists():
        sys.exit(f"error: {in_csv} not found -- run 1_collect_fold_metrics.py first")

    rng = np.random.default_rng(args.seed)
    big = pd.read_csv(in_csv)
    print(f"read {in_csv}  ({len(big):,} rows)", flush=True)

    rows, fold_rows = [], []
    for panel, spec in PANELS.items():
        for track_type in spec["track_types"]:
            store = {}
            for model in MODELS:
                sub = big[(big.model == model) & (big.level == spec["level"])
                          & (big.metric == spec["metric"]) & (big.track_type == track_type)]
                if sub.empty:
                    continue
                res = analyse(sub, spec, rng, args.n_boot)
                if res is None:
                    continue
                store[model] = res
                rows.append(dict(
                    panel=panel, statistic=spec["title"], track_type=track_type, model=model,
                    n_units=res["n_units"], n_folds=res["n_folds"],
                    point=round(res["point"], 4),
                    fold_mean=round(res["fold_mean"], 4), fold_sd=round(res["fold_sd"], 4),
                    fold_ci95_lo=round(res["fold_ci_lo"], 4), fold_ci95_hi=round(res["fold_ci_hi"], 4),
                    unit_boot_ci95_lo=round(res["unit_ci_lo"], 4),
                    unit_boot_ci95_hi=round(res["unit_ci_hi"], 4),
                    hier_boot_ci95_lo=round(res["hier_ci_lo"], 4),
                    hier_boot_ci95_hi=round(res["hier_ci_hi"], 4),
                ))
                for j, v in enumerate(res["per_fold"]):
                    fold_rows.append(dict(panel=panel, track_type=track_type, model=model,
                                          fold=j, value=round(float(v), 4)))

            # paired per-fold contrasts against each baseline
            if "Shorkie" not in store:
                continue
            a = store["Shorkie"]["per_fold"]
            for baseline in ("Shorkie_Random_Init", "Shorkie_Random_Init_untuned"):
                if baseline not in store:
                    continue
                b = store[baseline]["per_fold"]
                m = min(len(a), len(b))
                d = a[:m] - b[:m]
                if m < 2:
                    continue
                se = float(np.std(d, ddof=1) / np.sqrt(m))
                half = float(stats.t.ppf(0.975, m - 1) * se)
                try:
                    p = float(stats.wilcoxon(a[:m], b[:m]).pvalue)
                except ValueError:            # all differences zero
                    p = 1.0
                rows.append(dict(
                    panel=panel, statistic=f"DELTA Shorkie - {baseline}", track_type=track_type,
                    model=f"Shorkie_minus_{baseline}", n_units=store["Shorkie"]["n_units"],
                    n_folds=m, point=round(float(d.mean()), 4),
                    fold_mean=round(float(d.mean()), 4), fold_sd=round(float(np.std(d, ddof=1)), 4),
                    fold_ci95_lo=round(float(d.mean()) - half, 4),
                    fold_ci95_hi=round(float(d.mean()) + half, 4),
                    unit_boot_ci95_lo=np.nan, unit_boot_ci95_hi=np.nan,
                    hier_boot_ci95_lo=np.nan, hier_boot_ci95_hi=np.nan,
                    wilcoxon_p=round(p, 5),
                ))

    df = pd.DataFrame(rows)
    ci_csv = out_dir / "headline_ci.csv"
    df.to_csv(ci_csv, index=False)
    print(f"wrote {ci_csv}", flush=True)

    fold_csv = out_dir / "paired_fold_deltas.csv"
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False)
    print(f"wrote {fold_csv}", flush=True)

    if args.no_verify:
        print("(--no_verify: skipping the published-anchor checks)")
        return

    # Anchor the recomputation against the published values.
    checks = []
    for (panel, tt, model), reported in PUBLISHED.items():
        hit = df[(df.panel == panel) & (df.track_type == tt) & (df.model == model)]
        repro = float(hit.iloc[0]["point"]) if len(hit) else float("nan")
        checks.append(Check(panel=f"{panel}[{tt},{model}]", metric="published point estimate",
                            reported=reported, reproduced=repro, rtol=0.02))
    verify_csv = out_dir / "verify_revision_01.csv"
    write_verdicts(checks, verify_csv)
    print(f"wrote {verify_csv}")
    print(summary(checks))

    show = df[(df.panel == "3C") & (df.track_type == "RNA-Seq")]
    print("\nFigure 3C, RNA-Seq tracks:")
    print(show[["model", "n_units", "n_folds", "point", "fold_mean", "fold_ci95_lo",
                "fold_ci95_hi", "hier_boot_ci95_lo", "hier_boot_ci95_hi",
                "wilcoxon_p" if "wilcoxon_p" in show.columns else "point"]].to_string(index=False))

if __name__ == "__main__":
    main()
