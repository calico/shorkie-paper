#!/usr/bin/env python3
"""Revision experiment 02, step 3 — quantify how much ISM attribution each model places on
each TF motif, and contrast the two.

The claim under review -- "Shorkie's ISM maps preserved regulatory motif signatures...
whereas Shorkie_Random_Init failed to recover key motifs" -- is qualitative, and the
there are known counter-examples (RRPE). This step replaces it with a per-motif effect
size and interval, computed so that the counter-examples are visible rather than averaged
away.

Measurement, per motif occurrence found in step 2:

    recovery = mean( standardised |saliency| INSIDE the motif )
             - mean( standardised |saliency| in its FLANKS )

Two design choices make the two models comparable:

  * Saliency is standardised WITHIN each window before anything else (step 1). The models'
    raw ISM magnitudes differ by ~2.5x overall and by more than 10x in places, so a
    comparison of raw attribution would measure scale rather than motif recovery.
  * The flank reference excludes positions covered by ANY motif hit, so a neighbouring
    site cannot inflate the background and mask a real enrichment.

Because both models are scored at the SAME occurrences, the contrast is paired -- reported
as the median paired difference with a bootstrap interval, a two-sided Wilcoxon signed-rank
test, and the matched-pairs rank-biserial effect size, Holm-corrected across motifs.

Writes ``results/motif_recovery.csv`` (per motif, per model and the contrast) and
``results/motif_recovery_occurrences.csv`` (per occurrence, for re-analysis).

CPU only, under a minute.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from shorkie import config

MODELS = ["Shorkie", "Shorkie_Random_Init"]

def parse_args():
    parser = argparse.ArgumentParser(description="Per-motif ISM recovery scores and contrasts.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--flank", type=int, default=50,
                        help="Flank width on each side of a motif, in bp")
    parser.add_argument("--min_flank_positions", type=int, default=20,
                        help="Minimum usable flank positions for an occurrence to count")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def boot_ci(values, rng, n_boot, stat=np.median):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan, np.nan
    idx = rng.integers(0, v.size, (n_boot, v.size))
    draws = stat(v[idx], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation: the Wilcoxon effect size, in [-1, 1]."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())

def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        out[i] = min(running, 1.0)
    return out

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    cache = out_dir / "saliency_cache.npz"
    hits_path = out_dir / "motif_hits.tsv"
    for p in (cache, hits_path):
        if not p.exists():
            sys.exit(f"error: {p} not found -- run steps 1 and 2 first")

    data = np.load(cache, allow_pickle=True)
    z = {m: data[f"absz_{m}"] for m in MODELS}
    n_windows, L = z[MODELS[0]].shape
    hits = pd.read_csv(hits_path, sep="\t")
    print(f"windows {n_windows} x {L} bp; {len(hits):,} hits over "
          f"{hits.motif.nunique()} motifs", flush=True)

    # Positions covered by any motif hit, per window: excluded from every flank so a
    # neighbouring site cannot contaminate the background.
    covered = np.zeros((n_windows, L), dtype=bool)
    for w, s, e in zip(hits.window, hits.start, hits.end):
        covered[w, s:e] = True

    rng = np.random.default_rng(args.seed)
    occ_rows = []
    for r in hits.itertuples(index=False):
        w, s, e = int(r.window), int(r.start), int(r.end)
        inside = np.zeros(L, dtype=bool)
        inside[s:e] = True
        flank = np.zeros(L, dtype=bool)
        flank[max(0, s - args.flank):s] = True
        flank[e:min(L, e + args.flank)] = True
        flank &= ~covered[w]
        if flank.sum() < args.min_flank_positions:
            continue
        rec = dict(motif=r.motif, window=w, start=s, end=e,
                   strand=r.strand, score=r.score, flank_n=int(flank.sum()))
        for m in MODELS:
            rec[f"recovery_{m}"] = float(z[m][w][inside].mean() - z[m][w][flank].mean())
        rec["contrast"] = rec[f"recovery_{MODELS[0]}"] - rec[f"recovery_{MODELS[1]}"]
        occ_rows.append(rec)

    occ = pd.DataFrame(occ_rows)
    if occ.empty:
        sys.exit("error: no occurrences had a usable flank")
    occ.to_csv(out_dir / "motif_recovery_occurrences.csv", index=False)

    rows = []
    for motif, g in occ.groupby("motif"):
        row = dict(motif=motif, n_sites=len(g))
        for m in MODELS:
            v = g[f"recovery_{m}"].to_numpy()
            lo, hi = boot_ci(v, rng, args.n_boot)
            row[f"median_{m}"] = round(float(np.median(v)), 4)
            row[f"ci_lo_{m}"] = round(lo, 4)
            row[f"ci_hi_{m}"] = round(hi, 4)
        d = g["contrast"].to_numpy()
        lo, hi = boot_ci(d, rng, args.n_boot)
        row["median_contrast"] = round(float(np.median(d)), 4)
        row["contrast_ci_lo"] = round(lo, 4)
        row["contrast_ci_hi"] = round(hi, 4)
        row["rank_biserial"] = round(rank_biserial(d), 3)
        try:
            row["wilcoxon_p"] = float(stats.wilcoxon(
                g[f"recovery_{MODELS[0]}"], g[f"recovery_{MODELS[1]}"]).pvalue)
        except ValueError:
            row["wilcoxon_p"] = np.nan
        row["favours"] = ("Shorkie" if row["median_contrast"] > 0
                          else "Shorkie_Random_Init")
        rows.append(row)

    res = pd.DataFrame(rows)
    res["wilcoxon_p_holm"] = holm(res.wilcoxon_p.to_numpy())
    res["significant"] = res.wilcoxon_p_holm < 0.05
    res = res.sort_values("median_contrast", ascending=False)
    res.to_csv(out_dir / "motif_recovery.csv", index=False)

    n_sig = int(res.significant.sum())
    n_shk = int((res.significant & (res.favours == "Shorkie")).sum())
    n_rnd = int((res.significant & (res.favours == "Shorkie_Random_Init")).sum())
    print(f"\nmotifs tested            : {len(res)}")
    print(f"significant after Holm   : {n_sig}")
    print(f"  favouring Shorkie      : {n_shk}")
    print(f"  favouring Random_Init  : {n_rnd}")

    named = [m for m in res.motif if m.startswith("cons_") or m in
             ("RAP1", "REB1", "SPT15", "CBF1", "ABF1", "MCM1", "FHL1", "SFP1",
              "UME6", "DOT6", "STB3", "TBF1", "MSN2", "MSN4", "SWI4", "RPN4")]
    show = res[res.motif.isin(named)]
    print("\nMotifs named in the paper (positive contrast = Shorkie attributes more):")
    print(show[["motif", "n_sites", "median_Shorkie", "median_Shorkie_Random_Init",
                "median_contrast", "contrast_ci_lo", "contrast_ci_hi",
                "rank_biserial", "wilcoxon_p_holm", "favours"]].to_string(index=False))
    print(f"\nwrote {out_dir/'motif_recovery.csv'}")
    print(f"wrote {out_dir/'motif_recovery_occurrences.csv'}")

if __name__ == "__main__":
    main()
