#!/usr/bin/env python3
"""Revision experiment 04, step 1 — recompute the coding (72%) and repetitive (7.39%)
genome fractions quoted in the Results.

The down-weighting itself IS in the Methods (manuscript L582-584) and in the committed
``scripts/02_train/shorkie_lm/params.json`` -- step 2 documents that. What is missing is
where the two percentages come from, which is what this script pins down.

  * repetitive -- the soft-masked (lowercase) fraction of the RepeatMasker + DUST
    soft-masked R64 assembly the LM corpus build produces
    (``<r64 tier>/fasta/*.cleaned.fasta.masked.dust.softmask``).
  * coding     -- ambiguous in the paper, so every plausible definition is computed and
    reported side by side rather than one being asserted: exon vs CDS, all gene biotypes
    vs the protein-coding subset, with and without the 2 bp edge "chew" that the training
    exon mask applies (``shorkie.data.bed_helper.get_exon_mask``, chew_bp=2).

Both are reported against two denominators: the whole 12,071,326 bp R64 assembly, and the
union of the 16,384 bp windows the language model actually trains and evaluates on (the
loss weighting only ever sees those).

Outputs ``results/annotation_fractions.csv`` and ``results/verify_revision_04.csv``.
CPU only, ~20 s.
"""
import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

from shorkie import config

sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa: E402

PUBLISHED_CODING = 72.0
PUBLISHED_REPEAT = 7.39

# The training exon mask chews 2 bp off each internal exon edge.
CHEW_BP = 2

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute the coding and repetitive fractions of S. cerevisiae R64."
    )
    parser.add_argument("--r64_tier_dir", default=None,
                        help="R64 corpus tier dir holding fasta/ and gtf/ "
                             "[default: <datasets.lm_corpus_split_root>/data_r64_gtf]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def _norm(chrom):
    """The FASTA/window BEDs use chrI..chrXVI; the GTF uses I..XVI."""
    return chrom[3:] if chrom.startswith("chr") else chrom

def merge(by):
    out = {}
    for c, v in by.items():
        v = sorted(v)
        merged, (cs, ce) = [], v[0]
        for s, e in v[1:]:
            if s > ce:
                merged.append((cs, ce)); cs, ce = s, e
            else:
                ce = max(ce, e)
        merged.append((cs, ce))
        out[c] = merged
    return out

def load_gtf(path, feature, chew=0):
    by = collections.defaultdict(list)
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != feature:
            continue
        s, e = int(f[3]) - 1 + chew, int(f[4]) - chew
        if e > s:
            by[_norm(f[0])].append((s, e))
    return merge(by) if by else {}

def load_bed(path):
    by = collections.defaultdict(list)
    for line in open(path):
        f = line.split()
        if len(f) >= 3:
            by[_norm(f[0])].append((int(f[1]), int(f[2])))
    return merge(by) if by else {}

def softmask_intervals(fasta_path):
    """Lowercase runs of a soft-masked FASTA, as merged per-contig intervals."""
    by = collections.defaultdict(list)
    chrom, pos, run_start = None, 0, None
    for line in open(fasta_path):
        if line.startswith(">"):
            if run_start is not None:
                by[_norm(chrom)].append((run_start, pos))
            chrom = line[1:].split()[0]
            pos, run_start = 0, None
            continue
        for ch in line.strip():
            if ch.islower():
                if run_start is None:
                    run_start = pos
            elif run_start is not None:
                by[_norm(chrom)].append((run_start, pos)); run_start = None
            pos += 1
    if run_start is not None:
        by[_norm(chrom)].append((run_start, pos))
    return merge(by) if by else {}

def total_bp(ivs):
    return sum(e - s for v in ivs.values() for s, e in v)

def intersect_bp(a, b):
    """Total bp of a restricted to b (both merged interval dicts)."""
    tot = 0
    for c, av in a.items():
        bv = b.get(c)
        if not bv:
            continue
        i = j = 0
        while i < len(av) and j < len(bv):
            s = max(av[i][0], bv[j][0]); e = min(av[i][1], bv[j][1])
            if e > s:
                tot += e - s
            if av[i][1] < bv[j][1]:
                i += 1
            else:
                j += 1
    return tot

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tier = Path(args.r64_tier_dir) if args.r64_tier_dir else \
        config.path("datasets.lm_corpus_split_root") / "data_r64_gtf"
    if not tier.exists():
        sys.exit(f"error: R64 corpus tier not found: {tier}\n"
                 "       set --r64_tier_dir or datasets.lm_corpus_split_root in config/paths.yaml")

    fasta_dir, gtf_dir = tier / "fasta", tier / "gtf"
    softmask_fa = fasta_dir / "GCA_000146045_2.cleaned.fasta.masked.dust.softmask"
    gtf_all = gtf_dir / "GCA_000146045_2.59.gtf"
    gtf_pc = gtf_dir / "GCA_000146045_2.protein_coding.59.gtf"
    windows_bed = tier / "sequences.bed"
    for p in (softmask_fa, gtf_all, windows_bed):
        if not p.exists():
            sys.exit(f"error: required input missing: {p}")

    print(f"R64 tier : {tier}", flush=True)

    # Denominators.
    fai = fasta_dir / "GCA_000146045_2.cleaned.fasta.fai"
    genome_bp = sum(int(l.split("\t")[1]) for l in open(fai)) if fai.exists() else None
    windows = load_bed(windows_bed)
    windows_bp = total_bp(windows)
    print(f"genome   : {genome_bp:,} bp", flush=True)
    print(f"LM windows (union): {windows_bp:,} bp over {len(windows)} contigs", flush=True)

    repeats = softmask_intervals(softmask_fa)
    definitions = {
        "repetitive: soft-masked (RepeatMasker + DUST)": repeats,
        "coding: exon, all gene biotypes": load_gtf(gtf_all, "exon"),
        f"coding: exon, all biotypes, {CHEW_BP} bp chew": load_gtf(gtf_all, "exon", CHEW_BP),
        "coding: CDS": load_gtf(gtf_all, "CDS"),
        f"coding: CDS, {CHEW_BP} bp chew": load_gtf(gtf_all, "CDS", CHEW_BP),
    }
    if gtf_pc.exists():
        definitions["coding: exon, protein_coding genes only"] = load_gtf(gtf_pc, "exon")
        definitions[f"coding: exon, protein_coding, {CHEW_BP} bp chew"] = \
            load_gtf(gtf_pc, "exon", CHEW_BP)

    rows = []
    for name, ivs in definitions.items():
        if not ivs:
            print(f"SKIPPED: no intervals for '{name}'", file=sys.stderr)
            continue
        gbp = total_bp(ivs)
        wbp = intersect_bp(ivs, windows)
        rows.append(dict(
            definition=name,
            bp_genome=gbp,
            pct_of_genome=round(100 * gbp / genome_bp, 3) if genome_bp else None,
            bp_in_lm_windows=wbp,
            pct_of_lm_windows=round(100 * wbp / windows_bp, 3) if windows_bp else None,
        ))
    df = pd.DataFrame(rows)
    published = df.definition.str.startswith("coding").map({True: PUBLISHED_CODING,
                                                            False: PUBLISHED_REPEAT})
    df["published_value"] = published
    df["abs_diff_genome"] = (df.pct_of_genome - published).abs().round(3)
    df.to_csv(out_dir / "annotation_fractions.csv", index=False)

    rep = df[df.definition.str.startswith("repetitive")].iloc[0]
    coding = df[df.definition.str.startswith("coding")].sort_values("abs_diff_genome").iloc[0]

    checks = [
        Check(panel="loss weights", metric="repetitive fraction of R64 (%)",
              reported=PUBLISHED_REPEAT, reproduced=float(rep.pct_of_genome), rtol=0.01),
        Check(panel="loss weights",
              metric=f"coding fraction of R64 (%), closest definition: {coding.definition}",
              reported=PUBLISHED_CODING, reproduced=float(coding.pct_of_genome), rtol=0.02),
    ]
    write_verdicts(checks, out_dir / "verify_revision_04.csv")
    print("\n" + df.to_string(index=False))
    print()
    print(summary(checks))
    print(f"\nwrote {out_dir/'annotation_fractions.csv'}")

if __name__ == "__main__":
    main()
