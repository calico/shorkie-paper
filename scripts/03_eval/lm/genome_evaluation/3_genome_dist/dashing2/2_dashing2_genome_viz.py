#!/usr/bin/env python3
"""Figure 1D (alternative metric) — visualise dashing2 similarity scores from R64 to
every <data_type> genome as a sorted bar chart (one PNG per data_type). Reads the
per-target .txt files written by 1_dashing2_genome.sh. Paths resolve through
config/paths.yaml using the same layout as ../_genome_dist_env.sh (see ../README.md).

    python 2_dashing2_genome_viz.py <data_type>     # e.g. strains_gtf
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config

LCS = str(config.path("datasets.lm_corpus_split_root"))
RESULTS_ROOT = os.environ.get("GD_OUTPUT_ROOT") or str(config.path("corpus_build_results_root"))
REF_FASTA = f"{LCS}/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"
REF_BASE = os.path.basename(REF_FASTA).replace(".cleaned.fasta", "")


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_dashing2_similarity(path):
    """A dashing2 `--cmpout` row carries the similarity score in the 3rd field."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if len(parts) > 2 and _is_float(parts[2]):
                    return float(parts[2])
    return None


def main(data_type):
    fasta_dir = f"{LCS}/data_{data_type}/fasta"
    output_dir = f"{RESULTS_ROOT}/ensembl_fungi_59/{data_type}/genome_dist/dashing2"

    targets = [f.replace(".cleaned.fasta", "") for f in sorted(os.listdir(fasta_dir))
               if f.endswith(".cleaned.fasta")]
    data = {}
    for t in targets:
        s = parse_dashing2_similarity(os.path.join(output_dir, f"{data_type}_{REF_BASE}_{t}.txt"))
        if s is not None:
            data[t] = s

    df = pd.DataFrame(sorted(data.items(), key=lambda kv: kv[1], reverse=True), columns=["File", "Value"])
    print(df)

    plt.figure(figsize=(30, 6))
    plt.bar(df["File"], df["Value"], color="skyblue")
    plt.xlabel("Target Genomes")
    plt.ylabel("Dashing2 Similarity Score")
    plt.title("Dashing2 Similarity Score for Yeast Reference Genome (R64) and Target Genomes")
    plt.xticks(rotation=90)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{data_type}_dashing2_viz.png")
    plt.savefig(out, dpi=300)
    print(f"[OK] {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 2_dashing2_genome_viz.py <data_type>   (e.g. strains_gtf)")
        sys.exit(1)
    main(sys.argv[1])
