#!/usr/bin/env python3
"""Revision experiment 06, step 0 — regenerate the per-gene insertion-context TSVs the
MPRA scorer needs.

The published Figure 6 run passed ``--ctx <gene>.tsv`` to ``hound_MPRA_folds.py``, one file
per reporter gene, listing the genomic contexts an MPRA construct is spliced into. Those
files were working intermediates and are no longer on disk (``experiments/MPRA/genes/`` and
``experiments/SUM_data_process/MPRA/genes/`` are both empty), and no committed script
regenerated them -- so the MPRA pipeline could not be re-run at all, for any model, until
this step existed.

The originals survive one directory deeper than the published run scripts point at, under
``experiments/SUM_data_process/MPRA/MPRA_promoter_seqs/genes/{pos,neg}/``, alongside the
``generate_tsv.py`` that made them. This script reproduces that recipe exactly:

  * eleven insertion sites, 100 to 200 bp from the TSS in 10 bp steps (the minimum keeps a
    110 bp insert clear of the TSS: half the insert, 55 bp, plus a 45 bp gap);
  * midpoint at ``TSS - offset`` on the plus strand, ``TSS + offset`` on the minus strand;
  * the row interval is the 110 bp insert footprint, ``[midpoint - 55, midpoint + 55)``.

``hound_MPRA.py`` only reads ``chrom``/``start``/``end`` and takes the midpoint
(``midp = (start + end) // 2``), so the footprint matters for provenance rather than for
scoring. The one change from the original is that TSS coordinates are read from the GTF
instead of a hand-maintained list of 22 literals; ``--validate_against`` checks the output
is byte-identical to the originals.

Writes ``results/ctx/{pos,neg}/<SYMBOL>.tsv`` plus ``results/ctx/context_index.csv``.
CPU only, seconds.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from shorkie import config

# Reporter genes and their systematic ORFs, taken from the published figure-6 loaders
# (reproduction/figure_06/recheck/mpra_common.py). POS_GENES are on the plus strand,
# NEG_GENES on the minus strand.
POS_GENES = {
    "GPM3": "YOL056W", "SLI1": "YGR212W", "VPS52": "YDR484W",
    "YMR160W": "YMR160W", "MRPS28": "YDR337W", "YCT1": "YLL055W",
    "RDL2": "YOR286W", "PHS1": "YJL097W", "RTC3": "YHR087W", "MSN4": "YKL062W",
}
NEG_GENES = {
    "COA4": "YLR218C", "ERI1": "YPL096C-A", "RSM25": "YIL093C",
    "ERD1": "YDR414C", "MRM2": "YGL136C", "SNT2": "YGL131C",
    "CSI2": "YOL007C", "RPE1": "YJL121C", "PKC1": "YBL105C",
    "AIM11": "YER093C-A", "MAE1": "YKL029C", "MRPL1": "YDR116C",
}
# The eleven insertion sites, in bp from the TSS, and the 110 bp insert footprint --
# both verbatim from the original generate_tsv.py.
INSERTION_LENGTH = 110
HALF = INSERTION_LENGTH // 2                      # 55
SITES = list(range(HALF + 45, HALF + 45 + 10 * 11, 10))   # 100, 110,..., 200

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the per-gene MPRA insertion-context TSVs from the R64 GTF.")
    parser.add_argument("--gtf", default=None, help="R64 GTF [default: config genome.gtf]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--chrom_prefix", default="chr",
                        help="Prefix to give GTF chromosome names so they match the FASTA "
                             "(the GTF uses I..XVI, the FASTA chrI..chrXVI)")
    parser.add_argument("--validate_against", default=None,
                        help="Directory holding the original {pos,neg}/<SYMBOL>.tsv files; "
                             "if given, every regenerated file is compared byte for byte")
    return parser.parse_args()

def gene_coordinates(gtf_path, wanted):
    """{orf: (chrom, start0, end, strand)} for the requested systematic ORF names."""
    found = {}
    for line in open(gtf_path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        m = re.search(r'gene_id "([^"]+)"', f[8])
        if not m or m.group(1) not in wanted:
            continue
        found[m.group(1)] = (f[0], int(f[3]) - 1, int(f[4]), f[6])
    return found

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    ctx_dir = out_dir / "ctx"
    gtf = Path(args.gtf) if args.gtf else config.path("genome.gtf")
    if gtf is None or not Path(gtf).exists():
        sys.exit(f"error: --gtf not resolved ({gtf}). Fetch the genome with "
                 "`bash data/download.sh --genome -u PROJECT` or pass --gtf.")

    wanted = {**POS_GENES, **NEG_GENES}
    coords = gene_coordinates(gtf, set(wanted.values()))
    print(f"gtf   : {gtf}", flush=True)
    print(f"genes : {len(coords)}/{len(wanted)} reporter ORFs found", flush=True)

    rows = []
    for symbol, orf in wanted.items():
        if orf not in coords:
            print(f"SKIPPED: {symbol} ({orf}) not present in the GTF", file=sys.stderr)
            continue
        chrom, start, end, strand = coords[orf]
        # The original used 1-based GTF coordinates: TSS is the GTF `start` field on the
        # plus strand and the `end` field on the minus strand.
        tss = (start + 1) if strand == "+" else end
        tag = "pos" if symbol in POS_GENES else "neg"
        if (strand == "+") != (tag == "pos"):
            print(f"NOTE: {symbol} GTF strand {strand} disagrees with the published "
                  f"{tag} grouping; using the GTF strand", file=sys.stderr)
        chrom_out = f"{args.chrom_prefix}{chrom}" if not chrom.startswith(args.chrom_prefix) \
            else chrom
        recs = []
        for site in SITES:
            midp = tss - site if strand == "+" else tss + site
            recs.append(dict(chrom=chrom_out, start=midp - HALF, end=midp + HALF,
                             site_bp_from_tss=site, gene=symbol, orf=orf, strand=strand))
        gene_dir = ctx_dir / tag
        gene_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(recs)
        # hound_MPRA.py reads chrom/start/end; the rest is provenance for humans.
        df[["chrom", "start", "end"]].to_csv(gene_dir / f"{symbol}.tsv",
                                             sep="\t", index=False)
        rows += recs

    if not rows:
        sys.exit("error: no reporter genes resolved from the GTF")

    if args.validate_against:
        ref_root = Path(args.validate_against)
        same = diff = missing = 0
        for symbol in wanted:
            tag = "pos" if symbol in POS_GENES else "neg"
            ours, theirs = ctx_dir / tag / f"{symbol}.tsv", ref_root / tag / f"{symbol}.tsv"
            if not theirs.exists() or not ours.exists():
                missing += 1
                continue
            if ours.read_text() == theirs.read_text():
                same += 1
            else:
                diff += 1
                print(f"DIFFERS: {symbol}", file=sys.stderr)
        print(f"\nvalidation vs {ref_root}: {same} identical, {diff} differing, "
              f"{missing} missing")
    index = pd.DataFrame(rows)
    index.to_csv(ctx_dir / "context_index.csv", index=False)
    print(f"\nwrote {len(index)} contexts for {index.gene.nunique()} genes "
          f"({len(SITES)} insertion sites each)")
    print(index.groupby("strand").gene.nunique().to_string())
    print(f"\nwrote {ctx_dir}/{{pos,neg}}/<SYMBOL>.tsv")
    print(f"wrote {ctx_dir/'context_index.csv'}")

if __name__ == "__main__":
    main()
