#!/usr/bin/env python3

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

############################
# Yeast Chromosome Sizes   #
############################
# Adjust for your specific yeast reference version
YEAST_CHR_SIZES = {
    "chrI":   230218,
    "chrII":  813184,
    "chrIII": 316620,
    "chrIV":  1531933,
    "chrV":   576874,
    "chrVI":  270161,
    "chrVII": 1090940,
    "chrVIII": 562643,
    "chrIX":  439888,
    "chrX":   745751,
    "chrXI":  666816,
    "chrXII": 1078177,
    "chrXIII": 924431,
    "chrXIV": 784333,
    "chrXV":  1091291,
    "chrXVI": 948066
}

def append_chr_to_roman(chromosome):
    roman_mapping = {
        'I': 'chrI', 'II': 'chrII', 'III': 'chrIII', 'IV': 'chrIV',
        'V': 'chrV', 'VI': 'chrVI', 'VII': 'chrVII', 'VIII': 'chrVIII',
        'IX': 'chrIX', 'X': 'chrX', 'XI': 'chrXI', 'XII': 'chrXII',
        'XIII': 'chrXIII', 'XIV': 'chrXIV', 'XV': 'chrXV', 'XVI': 'chrXVI',
    }
    return roman_mapping.get(chromosome, chromosome)

# Define an explicit ordering of chromosomes
CHR_ORDER = [
    "chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI",
    "chrVII", "chrVIII", "chrIX", "chrX", "chrXI", "chrXII",
    "chrXIII", "chrXIV", "chrXV", "chrXVI"
]

############################
# Read FIMO Output (TSV)   #
############################
def read_fimo_results(fimo_file, label=None):
    """
    Reads a FIMO output TSV and returns a DataFrame with columns:
      ['motif_id', 'sequence_name', 'start', 'stop',
       'strand', 'score', 'p-value', 'q-value', 'matched_sequence']
    Also adds 'source_label' to identify which file it came from.
    """
    df = pd.read_csv(
        fimo_file,
        sep='\t',
        comment='#',
        header=None,
        names=["motif_id", "motif_alt_id", "sequence_name",
               "start", "stop", "strand", "score", "p-value",
               "q-value", "matched_sequence"]
    )
    # Remove empty or comment lines
    df = df.dropna(subset=["sequence_name", "start", "stop"])
    df["source_label"] = label if label else os.path.basename(fimo_file)
    return df

############################
# Read GTF for Genes       #
############################
def read_gtf(gtf_file, feature_type="gene"):
    """
    Reads a GTF file into a Pandas DataFrame.
    Keeps only rows where 'feature' == feature_type (by default 'gene').
    Returns columns:
      ['seqname', 'start', 'end', 'strand', 'feature', 'gene_name']
    """
    col_names = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ]
    df = pd.read_csv(gtf_file, sep='\t', comment='#', names=col_names)
    
    # Filter by feature type (e.g. 'gene', 'exon', 'CDS', etc.)
    df = df[df["feature"] == feature_type]
    
    # Optionally parse out gene_name or gene_id from attribute column
    # GTF attributes are often in the format: key "value"; key2 "value2"; etc.
    df["gene_name"] = df["attribute"].apply(parse_gene_name)
    # print(f"Loaded {len(df)} {feature_type} entries from {gtf_file}")
    return df

def parse_gene_name(attr_string):
    """
    Simple parser to extract 'gene_name' or 'gene_id' from the GTF attribute string.
    Falls back if not found.
    Example attribute: gene_id "YAL069W"; gene_version "1"; gene_name "CAF16";
    """
    # A more robust approach might use a regex, or parse with gtfparse, but here's a quick hack:
    fields = attr_string.strip().split(";")
    # For example: ['gene_id "YAL069W"', ' gene_version "1"', ' gene_name "CAF16"', '']
    for field in fields:
        field = field.strip()
        if field.startswith("gene_name"):
            # field might look like: gene_name "CAF16"
            parts = field.split()
            if len(parts) >= 2:
                return parts[1].replace('"', '')
        elif field.startswith("gene_id"):
            # fallback if no gene_name
            parts = field.split()
            if len(parts) >= 2:
                return parts[1].replace('"', '')
    return "unknown_gene"

############################
# Plotting Functions       #
############################
def plot_yeast_ideogram_with_genes(motif_df, gene_df, out_png="yeast_ideogram.png"):
    """
    Plots:
      - Wider chromosome bars
      - Gene regions as horizontal rectangles
      - Motif hits as points (or small bars) on each chromosome
    Saves the plot to out_png.
    """
    fig, ax = plt.subplots(figsize=(24, 12))  # wider figure

    # Assign each chromosome a y-level
    y_positions = {chr_name: i for i, chr_name in enumerate(CHR_ORDER)}

    # =============== #
    # Draw Chromosome #
    # =============== #
    # Instead of thin hlines, we'll use a "barh" for each chromosome
    # so the chromosome is visually thicker.
    bar_height = 0.6  # how thick each chromosome bar is
    for chr_name in CHR_ORDER:
        if chr_name not in YEAST_CHR_SIZES:
            continue
        chr_len = YEAST_CHR_SIZES[chr_name]
        y_val = y_positions[chr_name]
        
        ax.barh(
            y=y_val,
            width=chr_len,
            left=0,
            height=bar_height,
            color='lightgray',
            edgecolor='black'
        )

    # ========= #
    # Plot Genes #
    # ========= #
    # We'll draw each gene as a slightly thinner bar on top of the chromosome.
    # gene_df columns: ['seqname', 'start', 'end', 'strand', 'feature', 'gene_name']
    for idx, row in gene_df.iterrows():
        chrom = row["seqname"]
        chrom = append_chr_to_roman(chrom)
        if chrom not in y_positions:
            continue  # skip if not in our known chromosome list

        y_val = y_positions[chrom]
        start = row["start"]
        end = row["end"]
        # We can color-code by strand or gene name. Here, let's do a single color:
        # If you want to color by strand, for example:
        color = "blue" if row["strand"] == "+" else "green"

        gene_width = end - start
        # We'll center the gene bar on the same y-level, but narrower (bar_height * 0.4)
        gene_bar_height = bar_height * 0.4
        # shift a bit up or down based on strand:
        y_offset = 0.1 if row["strand"] == "+" else -0.1
        ax.barh(
            y=y_val + y_offset,
            width=gene_width,
            left=start,
            height=gene_bar_height,
            color=color,
            alpha=0.7
        )

    # ============== #
    # Plot Motif Hits #
    # ============== #
    # We'll plot each motif as a small circle at the midpoint of the match.
    for idx, row in motif_df.iterrows():
        chrom = row["sequence_name"]
        if chrom not in y_positions:
            continue

        y_val = y_positions[chrom]
        start = int(row["start"])
        stop = int(row["stop"])
        midpoint = (start + stop) / 2

        # Choose color by "source_label" to distinguish among the 3 FIMO files
        label = row["source_label"]
        color_map = {
            "fimo_out1.tsv": "red",
            "fimo_out2.tsv": "purple",
            "fimo_out3.tsv": "orange"
        }
        color = color_map.get(label, "black")  # default black if not in map

        ax.plot(midpoint, y_val, marker='o', markersize=0.2, color=color, alpha=0.8)

    # =========== #
    # Formatting  #
    # =========== #
    ax.set_yticks([y_positions[ch] for ch in CHR_ORDER])
    ax.set_yticklabels(CHR_ORDER)
    ax.set_xlabel("Genomic coordinate (bp)")
    ax.set_ylabel("Chromosome")
    ax.set_title("Motif Hits & Gene Regions on Yeast Chromosomes")

    # Expand x-limits to a bit beyond the longest chromosome
    max_chr_len = max(YEAST_CHR_SIZES.values())
    ax.set_xlim(-0.01 * max_chr_len, 1.01 * max_chr_len)
    # Use tight_layout for neatness
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")

def main():
    """
    Usage:
      python plot_yeast_ideogram_with_genes.py \
             fimo_out1.tsv fimo_out2.tsv fimo_out3.tsv \
             annotation.gtf \
             output.png
    """
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <fimo1.tsv> <fimo2.tsv> <fimo3.tsv> <annotation.gtf> <output.png>")
        sys.exit(1)

    fimo_files = sys.argv[1:4]
    gtf_file = sys.argv[4]
    out_png = sys.argv[5]

    # Read the FIMO results
    dfs = []
    for fimo_file in fimo_files:
        label = os.path.basename(fimo_file)
        df_fimo = read_fimo_results(fimo_file, label=label)
        dfs.append(df_fimo)
    motif_df = pd.concat(dfs, ignore_index=True)

    # Read the GTF annotation (genes)
    gene_df = read_gtf(gtf_file, feature_type="gene")

    # Plot everything
    plot_yeast_ideogram_with_genes(motif_df, gene_df, out_png=out_png)

if __name__ == "__main__":
    main()
