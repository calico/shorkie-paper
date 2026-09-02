#!/usr/bin/env python3
"""Revision experiment 10, step 3b — subsample the control corpus's training windows to
match 165_Saccharomycetales exactly.

This is the "match the total number of training windows" half of the control,
and it has to happen between corpus stage 3 (which writes ``sequences_train.bed``) and
stage 4 (which turns the BEDs into TFRecords) -- once the TFRecords exist the count is
baked in.

Two properties the subsample preserves, because getting either wrong would reintroduce a
confound the control is meant to remove:

  * the **per-genome share** of windows, so a handful of large fungal assemblies cannot
    dominate the reduced corpus;
  * the **valid and test splits untouched** -- they are drawn from S. cerevisiae R64 only
    and are identical across all tiers by construction, so they must stay identical here or
    perplexity is not comparable to the published Figure 1G.

Writes the reduced ``sequences_train.bed`` in place (keeping a ``.full`` backup) and a
report of what was dropped.

CPU only, seconds.
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from shorkie import config

TARGET_TRAIN_SEQS = 385551          # 165_Saccharomycetales, from data/manifest.json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Subsample training windows to a target count, preserving genome shares.")
    parser.add_argument("--bed", required=True, help="sequences_train.bed to reduce")
    parser.add_argument("--target", type=int, default=TARGET_TRAIN_SEQS,
                        help="Target number of training windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--genome_field", type=int, default=4,
                        help="0-based column holding the source-genome id "
                             "(sequences.bed: chrom start end split genome)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    return parser.parse_args()

def main():
    args = parse_args()
    config.load()
    bed = Path(args.bed)
    if not bed.exists():
        sys.exit(f"error: {bed} not found -- run corpus stages 1-3 first")

    lines = [l for l in bed.read_text().splitlines() if l.strip()]
    total = len(lines)
    print(f"{bed}: {total:,} training windows", flush=True)
    if total <= args.target:
        print(f"already at or below the target ({args.target:,}); nothing to do "
              f"-- the control cannot be window-matched by subsampling alone, so report "
              f"the achieved count instead of silently proceeding")
        return

    by_genome = defaultdict(list)
    for i, line in enumerate(lines):
        parts = line.split()
        key = parts[args.genome_field] if len(parts) > args.genome_field else "unknown"
        by_genome[key].append(i)

    rng = np.random.default_rng(args.seed)
    frac = args.target / total
    keep = []
    for key, idx in by_genome.items():
        k = int(round(len(idx) * frac))
        if k:
            keep.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    keep = sorted(set(keep))
    # largest-remainder top-up / trim so the count is exact
    if len(keep) < args.target:
        rest = sorted(set(range(total)) - set(keep))
        keep = sorted(keep + list(rng.choice(rest, size=args.target - len(keep),
                                             replace=False)))
    elif len(keep) > args.target:
        keep = sorted(rng.choice(keep, size=args.target, replace=False))

    print(f"keeping {len(keep):,} of {total:,} windows "
          f"({100*len(keep)/total:.1f}%) across {len(by_genome)} genomes")
    if args.dry_run:
        print("(dry run — not written)")
        return

    backup = bed.with_suffix(bed.suffix + ".full")
    if not backup.exists():
        shutil.copy2(bed, backup)
        print(f"backed up the full BED to {backup}")
    bed.write_text("\n".join(lines[i] for i in keep) + "\n")
    print(f"wrote {bed} with {len(keep):,} windows")

    # Keep statistics.json in step: train_seqs must equal the BED line count, or the
    # training loop and the released tier metadata disagree about the corpus size.
    stats = bed.parent / "statistics.json"
    if stats.exists():
        try:
            meta = json.loads(stats.read_text())
        except ValueError as e:
            print(f"SKIPPED: could not parse {stats} ({e})", file=sys.stderr)
        else:
            backup = stats.with_suffix(".json.full")
            if not backup.exists():
                shutil.copy2(stats, backup)
            before = meta.get("train_seqs")
            meta["train_seqs"] = len(keep)
            stats.write_text(json.dumps(meta, indent=1) + "\n")
            print(f"updated {stats}: train_seqs {before} -> {len(keep):,}")
    else:
        print(f"NOTE: no statistics.json beside {bed}; train_seqs not updated",
              file=sys.stderr)
    print("Now run corpus stage 4 to write the TFRecords.")

if __name__ == "__main__":
    main()
