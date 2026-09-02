#!/usr/bin/env python3
"""Revision experiment 02, step 1 — extract per-position ISM saliency for Shorkie and
Shorkie_Random_Init over the promoter windows they were both run on.

The saliency recipe is the published one (Methods Equations 18-20), ported from
``reproduction/figure_04/recheck/fig4_common.py::ism_saliency``:

    pwm  = logSED[idx, :, :, T0].mean(-1)          # average over the 384 T0 RNA-seq tracks
    pwm -= pwm.mean(-1, keepdims=True)             # zero-mean across the four bases  (Eq 19)
    logo = (pwm * reference_one_hot).sum(-1)       # project onto the reference base   (Eq 20)

Only ``(subset, part)`` combinations present for BOTH models are used, and the script
asserts that the two models' windows carry identical coordinates before pairing them --
the comparison is paired per window, so a coordinate mismatch would silently invalidate it.

One thing this step establishes up front, and that the rest of the experiment depends on:
the two models' raw saliency magnitudes differ by more than an order of magnitude
(Shorkie's mean |logo| is ~0.002, Shorkie_Random_Init's ~0.028). Any comparison of raw
attribution values would therefore measure scale, not motif recovery, so step 3 works on
per-window standardised saliency.

Writes ``results/saliency_cache.npz`` (per-window logo, |logo| z-scores, sequence and
coordinates for both models) and ``results/windows.csv``.

CPU only, a few minutes and ~1 GB of transient memory. Reads the cached ``scores.h5``
under ``results.ism_scores``; no model weights, no GPU.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from shorkie import config

MODELS = {"Shorkie": "motif_shorkie_RP_TSS", "Shorkie_Random_Init": "motif_random_init_RP_TSS"}
# Promoter-window subsets to consider; only those present for both models are used.
SUBSETS = ["gene_exp_motif_test_RP", "gene_exp_motif_test_TSS",
           "gene_exp_motif_test_TSS_select", "gene_exp_motif_test_RRB_targets"]
TRACK_OFFSET = 1148          # RNA-seq tracks start here in the 5,215-track sheet
N_RNASEQ = 3053
NT = np.array(list("ACGT"))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract paired ISM saliency for Shorkie and Shorkie_Random_Init.")
    parser.add_argument("--ism_root", default=None,
                        help="ISM score root [default: config results.ism_scores]")
    parser.add_argument("--targets_file", default=None,
                        help="Targets sheet [default: config datasets.targets_sheet]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--max_windows", type=int, default=None,
                        help="Cap the number of paired windows (for a quick smoke run)")
    return parser.parse_args()

def t0_track_indices(targets_file):
    """The 384 pre-induction (T0) RNA-seq tracks the published ISM logos average over."""
    tgt = pd.read_csv(targets_file, sep="\t")
    idx = [int(i) - TRACK_OFFSET for i in
           tgt[tgt["identifier"].str.contains("_T0_")]["index"].astype(int)]
    return np.array([t for t in idx if 0 <= t < N_RNASEQ], dtype=int)

def _readable(h5_path):
    """A handful of the cached scores.h5 are truncated; exclude them up front rather
    than failing mid-run."""
    try:
        with h5py.File(h5_path, "r") as h:
            _ = h["chr"].shape
        return True
    except (OSError, KeyError) as e:
        print(f"SKIPPED: unreadable {h5_path} ({str(e).splitlines()[0][:60]})",
              file=sys.stderr)
        return False

def discover_parts(ism_root, subset):
    """Parts present AND readable for both models, as a sorted list of part names."""
    per_model = []
    for tree in MODELS.values():
        base = ism_root / tree / subset / "f0c0"
        if not base.exists():
            return []
        per_model.append({p.name for p in base.iterdir()
                          if p.is_dir() and (p / "scores.h5").exists()
                          and _readable(p / "scores.h5")})
    common = set.intersection(*per_model) if per_model else set()
    return sorted(common, key=lambda s: int(s.replace("part", "")))

def read_part(h5_path, t0, want_seq):
    """Read one part into {(chrom, start, end, strand): (logo, seq_or_None)}.

    Reads are fail-soft per window: a few of the cached files are truncated and raise
    only once the data block is touched, so a bad window is dropped rather than losing
    the whole part.
    """
    out = {}
    try:
        h = h5py.File(h5_path, "r")
    except OSError as e:
        print(f"SKIPPED: cannot open {h5_path} ({str(e).splitlines()[0][:60]})",
              file=sys.stderr)
        return out
    with h:
        try:
            n = h["chr"].shape[0]
        except (OSError, KeyError):
            return out
        for i in range(n):
            try:
                pwm = h["logSED"][i, :, :, t0].astype(np.float32)
                oh = h["seqs"][i][:, :4].astype(np.float32)
                key = (h["chr"][i].decode(), int(h["start"][i]),
                       int(h["end"][i]), h["strand"][i].decode())
            except (OSError, KeyError, ValueError) as e:
                print(f"SKIPPED: {h5_path}[{i}] ({str(e).splitlines()[0][:50]})",
                      file=sys.stderr)
                continue
            # h5py fancy indexing on the last axis can move it to the front; normalise.
            if pwm.shape[0] == len(t0):
                pwm = pwm.transpose(1, 2, 0)
            pwm = pwm.mean(axis=-1)
            pwm -= pwm.mean(axis=-1, keepdims=True)
            logo = (pwm * oh).sum(axis=-1)
            out[key] = (logo, "".join(NT[oh.argmax(1)]) if want_seq else None)
    return out

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ism_root = Path(args.ism_root) if args.ism_root else config.path("results.ism_scores")
    targets_file = Path(args.targets_file) if args.targets_file \
        else config.path("datasets.targets_sheet")
    if not ism_root or not ism_root.exists():
        sys.exit(f"error: ISM score root not found: {ism_root}")
    t0 = t0_track_indices(targets_file)
    print(f"ism_root : {ism_root}\nT0 tracks: {len(t0)}", flush=True)

    logos = {m: [] for m in MODELS}
    meta, seqs = [], []
    stop = False
    for subset in SUBSETS:
        parts = discover_parts(ism_root, subset)
        if not parts:
            print(f"  {subset}: no parts shared by both models -- skipped", flush=True)
            continue
        print(f"  {subset}: {len(parts)} shared parts", flush=True)
        for part in parts:
            per_model = {}
            for label, tree in MODELS.items():
                path = ism_root / tree / subset / "f0c0" / part / "scores.h5"
                per_model[label] = read_part(path, t0, want_seq=(label == "Shorkie"))
            # Pair on genomic coordinate rather than position: the two models' parts can
            # lose different windows to truncation, so index alignment is not safe.
            shared = sorted(set(per_model["Shorkie"]) & set(per_model["Shorkie_Random_Init"]))
            if not shared:
                print(f"SKIPPED: {subset}/{part} has no windows shared by both models",
                      file=sys.stderr)
                continue
            for key in shared:
                ch, st, en, sd = key
                logos["Shorkie"].append(per_model["Shorkie"][key][0].astype(np.float32))
                logos["Shorkie_Random_Init"].append(
                    per_model["Shorkie_Random_Init"][key][0].astype(np.float32))
                seqs.append(per_model["Shorkie"][key][1])
                meta.append(dict(subset=subset, part=part, chrom=ch, start=st,
                                 end=en, strand=sd, length=len(seqs[-1])))
                if args.max_windows and len(meta) >= args.max_windows:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if not meta:
        sys.exit("error: no paired windows found")

    md = pd.DataFrame(meta)
    md.to_csv(out_dir / "windows.csv", index=False)

    arrays = {"sequences": np.array(seqs, dtype=object)}
    stats = []
    for label in MODELS:
        stack = np.stack(logos[label])                      # (n_windows, L)
        absal = np.abs(stack)
        # Per-window standardisation: the two models' raw magnitudes differ by >10x, so
        # only standardised saliency is comparable between them.
        mu = absal.mean(axis=1, keepdims=True)
        sd = absal.std(axis=1, keepdims=True)
        z = np.divide(absal - mu, sd, out=np.zeros_like(absal), where=sd > 0)
        arrays[f"logo_{label}"] = stack
        arrays[f"absz_{label}"] = z.astype(np.float32)
        stats.append(dict(model=label, windows=stack.shape[0], length=stack.shape[1],
                          mean_abs_logo=float(absal.mean()),
                          max_abs_logo=float(absal.max())))
    np.savez_compressed(out_dir / "saliency_cache.npz", **arrays)

    sdf = pd.DataFrame(stats)
    print("\n" + sdf.to_string(index=False))
    ratio = sdf.mean_abs_logo.max() / sdf.mean_abs_logo.min()
    print(f"\nraw magnitude ratio between models: {ratio:.1f}x "
          f"-- comparisons must use the standardised saliency")
    print(f"\npaired windows: {len(md)} across {md.subset.nunique()} subsets")
    print(md.groupby("subset").size().to_string())
    print(f"\nwrote {out_dir/'saliency_cache.npz'}\nwrote {out_dir/'windows.csv'}")

if __name__ == "__main__":
    main()
