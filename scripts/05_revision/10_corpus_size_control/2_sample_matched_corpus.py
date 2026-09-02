#!/usr/bin/env python3
"""Revision experiment 10, step 2 — draw the 165-genome fungal control corpus
.

Two design decisions that a one-sentence description of this control leaves open, both of
which matter for what it actually tests:

**Which 165.** Sampling from the whole fungal corpus would re-include the ~170
Saccharomycetales genomes it already contains, so the "broad" corpus would partly be the
narrow one and the contrast would be blunted. This script samples from the **1,191
non-Saccharomycetales** genomes only, so the two corpora differ in phylogenetic scope
rather than in membership.

**How to spread them.** A uniform random draw of 165 from 1,191 would be dominated by the
best-represented orders (Eurotiales, Hypocreales) and would not really be "the broader
fungal kingdom". The default is a **stratified** draw, allocating genomes across taxonomic
orders in proportion to their representation with at least one per order where possible, so
breadth is preserved. ``--strategy random`` gives the literal uniform draw for comparison.

Genome-size matching is offered too (``--match_genome_size``): fungal genomes average 2.4x
larger than Saccharomycetales ones, so an unmatched draw confounds scope with genome size
on top of everything else. It is off by default because the control is defined by
WINDOW-count matching, which step 3 does exactly after the corpus is built.

Writes the tier species list in the same schema as ``data/species_lists/*.cleaned.csv`` so
the existing corpus-build pipeline consumes it unchanged, plus a comparison table.

CPU only, seconds. Reads only committed files.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shorkie import config

EXCLUDE_ORDER = "Saccharomycetales"
TARGET_N = 165
REFERENCE_LIST = "species_saccharomycetales_gtf.cleaned.csv"
SOURCE_LIST = "species_fungi_1385_gtf.cleaned.csv"
OUT_NAME = "species_fungi165_matched_gtf.cleaned.csv"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a 165-genome non-Saccharomycetales fungal control corpus.")
    parser.add_argument("--n", type=int, default=TARGET_N, help="Genomes to draw")
    parser.add_argument("--strategy", choices=["stratified", "random"], default="stratified",
                        help="stratified spreads the draw across taxonomic orders")
    parser.add_argument("--match_genome_size", action="store_true",
                        help="Additionally match the genome-size distribution of the "
                             "165_Saccharomycetales tier by nearest-size pairing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--write_species_list", action="store_true",
                        help="Also write the list into data/species_lists/ so the corpus "
                             "build finds it without a path override")
    return parser.parse_args()

def stratified_draw(df, n, rng):
    """Allocate n genomes across taxonomic orders, largest-remainder, >=1 per order."""
    orders = df.Classification.value_counts()
    quota = {o: 1 for o in orders.index[:n]}                 # seed one per order
    remaining = n - len(quota)
    if remaining > 0:
        weights = orders.reindex(quota).astype(float)
        share = weights / weights.sum() * remaining
        base = np.floor(share).astype(int)
        for o, k in base.items():
            quota[o] += int(k)
        leftover = remaining - int(base.sum())
        if leftover > 0:
            for o in (share - base).sort_values(ascending=False).index[:leftover]:
                quota[o] += 1
    picks = []
    for order, k in quota.items():
        pool = df[df.Classification == order]
        k = min(k, len(pool))
        if k:
            picks.append(pool.sample(n=k, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks) if picks else df.head(0)
    if len(out) > n:
        out = out.sample(n=n, random_state=int(rng.integers(1 << 31)))
    elif len(out) < n:                                        # top up from the remainder
        rest = df[~df.index.isin(out.index)]
        out = pd.concat([out, rest.sample(n=min(n - len(out), len(rest)),
                                          random_state=int(rng.integers(1 << 31)))])
    return out

def size_matched_draw(df, reference, n, rng):
    """Greedy nearest-genome-size pairing against the reference tier's size distribution."""
    targets = reference.total_length.sample(
        n=min(n, len(reference)), random_state=int(rng.integers(1 << 31))).to_numpy()
    pool = df.copy()
    picks = []
    for t in targets:
        if pool.empty:
            break
        j = (pool.total_length - t).abs().idxmin()
        picks.append(pool.loc[j])
        pool = pool.drop(index=j)
    return pd.DataFrame(picks)

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    lists = repo / "data" / "species_lists"

    source = pd.read_csv(lists / SOURCE_LIST)
    reference = pd.read_csv(lists / REFERENCE_LIST)
    pool = source[source.Classification != EXCLUDE_ORDER].reset_index(drop=True)
    print(f"source corpus  : {len(source)} genomes", flush=True)
    print(f"excluding {EXCLUDE_ORDER}: {len(pool)} available", flush=True)
    if len(pool) < args.n:
        sys.exit(f"error: only {len(pool)} non-{EXCLUDE_ORDER} genomes available, need {args.n}")

    rng = np.random.default_rng(args.seed)
    if args.match_genome_size:
        picked = size_matched_draw(pool, reference, args.n, rng)
    elif args.strategy == "stratified":
        picked = stratified_draw(pool, args.n, rng)
    else:
        picked = pool.sample(n=args.n, random_state=args.seed)
    # Re-index in the same schema as the committed lists (leading unnamed index column),
    # so the corpus-build pipeline reads this file exactly like the four existing tiers.
    picked = picked.sort_values("Name").reset_index(drop=True)
    picked = picked.drop(columns=[c for c in picked.columns if c.startswith("Unnamed")])
    picked.insert(0, "Unnamed: 0", range(len(picked)))

    out_csv = out_dir / OUT_NAME
    picked.to_csv(out_csv, index=False)
    if args.write_species_list:
        picked.to_csv(lists / OUT_NAME, index=False)
        print(f"also wrote {lists / OUT_NAME}", flush=True)

    comp = pd.DataFrame([
        dict(corpus="165_Saccharomycetales (reference)", genomes=len(reference),
             total_gb=round(reference.total_length.sum() / 1e9, 3),
             mean_genome_mb=round(reference.total_length.mean() / 1e6, 2),
             pct_chromosome_level=round(
                 100 * (reference.assembly_level == "chromosome").mean(), 1),
             taxonomic_orders=reference.Classification.nunique()),
        dict(corpus=f"165 fungal control ({args.strategy}"
                    f"{', size-matched' if args.match_genome_size else ''})",
             genomes=len(picked),
             total_gb=round(picked.total_length.sum() / 1e9, 3),
             mean_genome_mb=round(picked.total_length.mean() / 1e6, 2),
             pct_chromosome_level=round(
                 100 * (picked.assembly_level == "chromosome").mean(), 1),
             taxonomic_orders=picked.Classification.nunique()),
    ])
    comp.to_csv(out_dir / "matched_corpus_comparison.csv", index=False)
    print("\n" + comp.to_string(index=False))
    print(f"\ntop orders drawn: "
          f"{dict(picked.Classification.value_counts().head(8))}")
    print(f"\nwrote {out_csv}")
    print("Next: 3_build_corpus.sh builds TFRecords for this tier and subsamples the "
          "training windows to match 165_Saccharomycetales exactly.")

if __name__ == "__main__":
    main()
