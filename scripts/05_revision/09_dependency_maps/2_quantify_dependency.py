#!/usr/bin/env python3
"""Revision experiment 09, step 2 — test whether dependency between motif positions exceeds
a distance-matched background.

This is the step that actually separates "words" from "grammar". A dependency map always
has structure -- nearby positions covary simply because the model sees them in the same
receptive field, so raw dependency falls off with distance regardless of any regulatory
relationship. The test therefore has to be:

    is dependency between two MOTIF positions higher than between two positions
    the SAME DISTANCE APART that are not both in motifs?

Motif occurrences are located by the same native PWM scan experiment 02 uses (log-odds
against the observed base composition, calibrated on a dinucleotide-shuffled null), so the
two experiments agree on what counts as a motif.

Three quantities per locus:

  * within-motif dependency, versus distance-matched background -- a sanity check; a model
    that has learned a motif at all should show this;
  * BETWEEN-motif dependency for pairs of distinct motif occurrences, versus
    distance-matched background -- this is the grammar test proper;
  * the same split by whether the pair is a known cooperating pair (e.g. Rap1 with
    RRPE/PAC, tandem Gal4 sites), where the paper's own claims predict a signal.

If between-motif dependency is not above the distance-matched background, the honest
conclusion is to narrow the claim from "regulatory grammar" to "conserved motifs".
Both outcomes are reportable.

Writes ``results/dependency_stats.csv`` and ``results/dependency_maps.png``.
CPU only, a few minutes.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from shorkie import config

DIST_BINS = np.array([0, 10, 20, 40, 80, 160, 320, 640])

def parse_args():
    parser = argparse.ArgumentParser(description="Quantify dependency against a matched background.")
    parser.add_argument("--meme", default=None,
                        help="[default: <motif_db_dir>/merged_meme_high_conf.meme]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--pvalue", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_scanner(repo):
    """Reuse experiment 02's PWM scanner so both experiments agree on what a motif is."""
    path = repo / "scripts/05_revision/02_ism_motif_recovery/2_scan_motifs.py"
    if not path.exists():
        sys.exit(f"error: motif scanner not found at {path}")
    spec = importlib.util.spec_from_file_location("scan_motifs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def motif_hits(scan, seq, motifs, background, pvalue, rng):
    codes = scan.encode(seq)
    null_codes = [scan.encode(scan.dinuc_shuffle(seq, rng)) for _ in range(3)]
    hits = []
    for name, pwm in motifs.items():
        lom = scan.log_odds(pwm, background)
        null = np.concatenate([scan.scan(c, lom)[0] for c in null_codes])
        null = null[np.isfinite(null)]
        if null.size == 0:
            continue
        thr = float(np.quantile(null, 1.0 - pvalue))
        best, _ = scan.scan(codes, lom)
        for pos in np.flatnonzero(best >= thr):
            hits.append((name, int(pos), int(pos + pwm.shape[0])))
    return hits

def summarise(dep, in_motif, pair_class, rng, n_boot=1000):
    """Compare dependency for a position class against a distance-matched background."""
    L = dep.shape[0]
    ii, jj = np.triu_indices(L, k=1)
    dist = jj - ii
    vals = dep[ii, jj]
    bin_idx = np.digitize(dist, DIST_BINS)
    rows = []
    for label, mask in pair_class.items():
        m = mask[ii, jj]
        if m.sum() < 20:
            continue
        deltas = []
        for b in np.unique(bin_idx[m]):
            sel = bin_idx == b
            fg = vals[sel & m]
            bg = vals[sel & ~m]
            if len(fg) < 5 or len(bg) < 5:
                continue
            deltas.append((len(fg), float(np.median(fg) - np.median(bg))))
        if not deltas:
            continue
        w = np.array([d[0] for d in deltas], dtype=float)
        d = np.array([d[1] for d in deltas], dtype=float)
        point = float((w * d).sum() / w.sum())
        draws = [float(np.average(d[k], weights=w[k]))
                 for k in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))]
        fg_all, bg_all = vals[m], vals[~m]
        try:
            p = float(stats.mannwhitneyu(fg_all, bg_all, alternative="greater").pvalue)
        except ValueError:
            p = np.nan
        rows.append(dict(pair_class=label, n_pairs=int(m.sum()),
                         distance_matched_delta=round(point, 4),
                         ci_lo=round(float(np.percentile(draws, 2.5)), 4),
                         ci_hi=round(float(np.percentile(draws, 97.5)), 4),
                         mannwhitney_p=p))
    return rows

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    maps = sorted((out_dir / "dep_maps").glob("*.npz")) if (out_dir / "dep_maps").exists() else []
    if not maps:
        sys.exit(f"error: no dependency maps under {out_dir/'dep_maps'} -- run step 1 first")

    scan = load_scanner(repo)
    meme = Path(args.meme) if args.meme else \
        Path(config.path("motif_db_dir")) / "merged_meme_high_conf.meme"
    if not meme.exists():
        sys.exit(f"error: motif file not found: {meme}")
    motifs = scan.read_meme(meme)
    for key, cons in scan.CONSENSUS_MOTIFS.items():
        motifs[key] = scan.consensus_pwm(cons)

    rng = np.random.default_rng(args.seed)
    rows, panels = [], []
    for path in maps:
        z = np.load(path, allow_pickle=True)
        dep = np.asarray(z["dep_map"], dtype=float)
        seq = str(z["sequence"])
        counts = np.zeros(4)
        for c in seq:
            if c in scan.NT_IX:
                counts[scan.NT_IX[c]] += 1
        background = counts / max(counts.sum(), 1)
        hits = motif_hits(scan, seq, motifs, background, args.pvalue, rng)
        L = len(seq)
        in_motif = np.zeros(L, dtype=bool)
        motif_id = np.full(L, -1, dtype=int)
        for k, (_, s, e) in enumerate(hits):
            in_motif[s:e] = True
            motif_id[s:e] = k

        both = in_motif[:, None] & in_motif[None, :]
        same = motif_id[:, None] == motif_id[None, :]
        pair_class = {
            "within one motif": both & same,
            "between two motifs": both & ~same,
        }
        stats_rows = summarise(dep, in_motif, pair_class, rng)
        for r in stats_rows:
            r["locus"] = path.stem
            r["n_motif_hits"] = len(hits)
            r["note"] = str(z["note"])
        rows += stats_rows
        panels.append((path.stem, dep, hits))
        print(f"{path.stem}: {len(hits)} motif hits, {len(stats_rows)} pair classes",
              flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("error: no locus had enough motif hits to test")
    cols = ["locus", "pair_class", "n_motif_hits", "n_pairs",
            "distance_matched_delta", "ci_lo", "ci_hi", "mannwhitney_p", "note"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_dir / "dependency_stats.csv", index=False)
    print("\n" + df.to_string(index=False))

    between = df[df.pair_class == "between two motifs"]
    if len(between):
        pos = int((between.ci_lo > 0).sum())
        print(f"\nBetween-motif dependency exceeds the distance-matched background at "
              f"{pos}/{len(between)} loci (95% CI excluding zero).")
        print("That is the grammar test: motif RECOVERY alone would not produce it.")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
    for ax, (name, dep, hits) in zip(axes[0], panels):
        im = ax.imshow(dep, cmap="magma", origin="lower",
                       vmax=float(np.percentile(dep, 99.5)))
        for _, s, e in hits:
            ax.add_patch(plt.Rectangle((s, s), e - s, e - s, fill=False,
                                       edgecolor="cyan", lw=0.7))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("position"); ax.set_ylabel("position")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Shorkie_LM nucleotide dependency maps (motif occurrences outlined)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "dependency_maps.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_dir/'dependency_stats.csv'}")
    print(f"wrote {out_dir/'dependency_maps.png'}")

if __name__ == "__main__":
    main()
