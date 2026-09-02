#!/usr/bin/env python3
"""Revision experiment 10, step 1 — quantify every factor confounded with phylogenetic
scope across the four pretraining corpora.

The evolutionary "sweet spot" at 165 Saccharomycetales genomes is confounded with at least
three other things: corpus size, annotation quality, and optimisation difficulty. The peak
could be driven partly by the amount of data rather than by phylogenetic scope, and the paper
acknowledges the confounds without measuring them.

Steps 2-4 build and train the corpus-matched control. This step is the part that needs no
GPU at all: it puts numbers on each named confound, from the committed
per-tier species lists and the release manifest. Even without the new model, this converts
"potentially confounded by corpus size and annotation quality" from an acknowledgement into
a table.

Reported per tier: genome count, total assembly bp, genome-size distribution, assembly-level
mix (chromosome- vs scaffold-level, a direct proxy for annotation quality), contig counts,
taxonomic breadth, and the train/valid/test window counts the models actually saw.

Writes ``results/corpus_characteristics.csv``, ``results/corpus_taxonomy.csv`` and
``results/verify_revision_10.csv``. CPU only, seconds; reads only committed files.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from shorkie import config

sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa: E402

# Tier label -> (species-list filename, manifest key)
TIERS = [
    ("R64", "species_r64_gtf.cleaned.csv", "R64"),
    ("80_strains", "species_strains_gtf.cleaned.csv", "80_strains"),
    ("165_Saccharomycetales", "species_saccharomycetales_gtf.cleaned.csv",
     "165_Saccharomycetales"),
    ("1341_Fungus", "species_fungi_1385_gtf.cleaned.csv", "1341_Fungus"),
]
PRETRAINING_TIER = "165_Saccharomycetales"
# The LM corpus build tiles each assembly with 16,384 bp windows at a 4,096 bp stride
# (shorkie.data.bed_helper.generate_beds defaults, and scripts/01_data_build/lm_corpus/).
# Windows therefore overlap 4x, so unique sampled bp is train_seqs * STRIDE, NOT
# train_seqs * seq_length -- the latter counts every base up to four times.
LM_WINDOW_STRIDE = 4096

def parse_args():
    parser = argparse.ArgumentParser(
        description="Characterise the four LM pretraining corpora.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((repo / "data" / "manifest.json").read_text())
    tiers_meta = manifest["datasets"]["lm_corpus"]["tiers"]

    rows, tax_rows = [], []
    for label, filename, manifest_key in TIERS:
        path = repo / "data" / "species_lists" / filename
        if not path.exists():
            print(f"SKIPPED: species list missing: {path}", file=sys.stderr)
            continue
        d = pd.read_csv(path)
        stats = tiers_meta.get(manifest_key, {}).get("statistics", {})
        levels = d.assembly_level.value_counts().to_dict() if "assembly_level" in d else {}
        n_chrom_level = int(levels.get("chromosome", 0))
        rows.append(dict(
            tier=label,
            genomes=len(d),
            total_bp=int(d.total_length.sum()),
            total_gb=round(d.total_length.sum() / 1e9, 3),
            mean_genome_mb=round(d.total_length.mean() / 1e6, 2),
            median_genome_mb=round(d.total_length.median() / 1e6, 2),
            max_genome_mb=round(d.total_length.max() / 1e6, 2),
            chromosome_level=n_chrom_level,
            scaffold_level=int(levels.get("scaffold", 0)),
            pct_chromosome_level=round(100 * n_chrom_level / len(d), 1),
            median_contigs=int(d.n_chroms.median()) if "n_chroms" in d else None,
            taxonomic_orders=int(d.Classification.nunique()) if "Classification" in d else None,
            train_seqs=stats.get("train_seqs"),
            valid_seqs=stats.get("valid_seqs"),
            test_seqs=stats.get("test_seqs"),
            seq_length=stats.get("seq_length"),
        ))
        if "Classification" in d:
            for order, n in d.Classification.value_counts().items():
                tax_rows.append(dict(tier=label, order=order, genomes=int(n)))

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("error: no species lists found under data/species_lists/")
    df["train_window_bp"] = df.train_seqs * df.seq_length          # with 4x overlap
    df["unique_train_bp"] = df.train_seqs * LM_WINDOW_STRIDE       # non-redundant
    df["train_windows_per_genome"] = (df.train_seqs / df.genomes).round(1)
    # How much of each raw assembly survives repeat/homology/paralog filtering into
    # training windows. The tiers differ enormously here, which is itself a confound the
    # "annotation quality" concern is pointing at.
    df["pct_assembly_sampled"] = (100 * df.unique_train_bp / df.total_bp).round(2)
    df.to_csv(out_dir / "corpus_characteristics.csv", index=False)
    pd.DataFrame(tax_rows).to_csv(out_dir / "corpus_taxonomy.csv", index=False)

    print(df[["tier", "genomes", "total_gb", "mean_genome_mb", "pct_chromosome_level",
              "taxonomic_orders", "train_seqs", "train_windows_per_genome",
              "pct_assembly_sampled"]].to_string(index=False))

    sac = df[df.tier == PRETRAINING_TIER]
    fun = df[df.tier == "1341_Fungus"]
    if len(sac) and len(fun):
        s, f = sac.iloc[0], fun.iloc[0]
        print(f"\nThe comparison in question:")
        print(f"  training windows   {PRETRAINING_TIER}: {int(s.train_seqs):,}   "
              f"1341_Fungus: {int(f.train_seqs):,}   "
              f"(fungal corpus has {f.train_seqs/s.train_seqs:.2f}x MORE)")
        print(f"  raw assembly       {s.total_gb} Gb vs {f.total_gb} Gb   "
              f"({f.total_bp/s.total_bp:.1f}x more)")
        print(f"  mean genome size   {s.mean_genome_mb} Mb vs {f.mean_genome_mb} Mb   "
              f"({f.mean_genome_mb/s.mean_genome_mb:.1f}x larger)")
        print(f"  chromosome-level   {s.pct_chromosome_level}% vs {f.pct_chromosome_level}%")
        print(f"  assembly sampled   {s.pct_assembly_sampled}% vs {f.pct_assembly_sampled}% "
              f"of raw bp survives filtering into training windows")
        print(f"  windows per genome {s.train_windows_per_genome} vs "
              f"{f.train_windows_per_genome}")
        print("\n  So the broader corpus has MORE training windows, not fewer -- raw data")
        print("  volume alone does not explain why it underperforms. But its genomes are")
        print("  2.4x larger and far more fragmented, and only a small fraction of each")
        print("  assembly survives repeat/homology filtering, so each genome contributes")
        print("  far less usable sequence. Step 2 builds the matched control that separates")
        print("  phylogenetic scope from these.")

    checks = [
        Check(panel="corpora", metric=f"{label} genome count",
              reported=expected, reproduced=int(df[df.tier == label].genomes.iloc[0]),
              rtol=0.0, atol=0.0)
        for label, expected in [("R64", 1), ("80_strains", 80),
                                ("165_Saccharomycetales", 165), ("1341_Fungus", 1361)]
        if label in set(df.tier)
    ]
    checks += [
        Check(panel="corpora", metric=f"{label} training windows",
              reported=expected, reproduced=int(df[df.tier == label].train_seqs.iloc[0]),
              rtol=0.0, atol=0.0)
        for label, expected in [("R64", 1201), ("80_strains", 102315),
                                ("165_Saccharomycetales", 385551), ("1341_Fungus", 625355)]
        if label in set(df.tier)
    ]
    write_verdicts(checks, out_dir / "verify_revision_10.csv")
    print()
    print(summary(checks))
    print(f"\nwrote {out_dir/'corpus_characteristics.csv'}")

if __name__ == "__main__":
    main()
