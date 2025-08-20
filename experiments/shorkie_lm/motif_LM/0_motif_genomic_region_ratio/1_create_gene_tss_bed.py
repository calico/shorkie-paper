#!/usr/bin/env python3

import sys
import pandas as pd
import re

def parse_gtf_attribute(attributes_str, key="gene_id"):
    """
    Parses the GTF attributes column to extract the value for a given key.
    Example attributes string:
      gene_id "YAL069W"; gene_version "1"; gene_name "RLM1";
    Returns the string value if found, or "NA" if not found.
    """
    pattern = rf'{key}\s+"([^"]+)"'
    match = re.search(pattern, attributes_str)
    if match:
        return match.group(1)
    else:
        return "NA"

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <input.gtf> <genes.bed> <tss.bed>")
        sys.exit(1)

    gtf_file = sys.argv[1]
    genes_bed = sys.argv[2]
    tss_bed = sys.argv[3]

    # Define column names for the GTF file
    col_names = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ]

    # Read GTF into a DataFrame
    df = pd.read_csv(gtf_file, sep="\t", comment="#", names=col_names)

    # Filter for entries that describe an entire gene.
    # Adjust if your GTF uses 'transcript' or something else.
    df_genes = df[df["feature"] == "gene"].copy()

    # Parse out gene_id or gene_name
    df_genes["gene_id"] = df_genes["attribute"].apply(parse_gtf_attribute, key="gene_id")
    df_genes["gene_name"] = df_genes["attribute"].apply(parse_gtf_attribute, key="gene_name")

    # Use gene_name or gene_id for the BED "name" field
    df_genes["bed_name"] = df_genes.apply(
        lambda row: row["gene_name"] if row["gene_name"] != "NA" else row["gene_id"],
        axis=1
    )

    ################################################
    # Prepare the DataFrame for genes.bed (spanning
    # the entire gene region).
    ################################################
    df_genes["bed_chrom"] = "chr" + df_genes["seqname"].astype(str)
    df_genes["bed_start"] = df_genes["start"] - 1  # convert to 0-based
    df_genes["bed_end"] = df_genes["end"]         # GTF end is inclusive, BED is exclusive
    df_genes["bed_score"] = "."

    # Reorder into standard BED columns
    bed_cols = ["bed_chrom", "bed_start", "bed_end", "bed_name", "bed_score", "strand"]
    df_genes_bed = df_genes[bed_cols].copy()

    # Ensure start is not negative
    df_genes_bed["bed_start"] = df_genes_bed["bed_start"].clip(lower=0)

    ################################################
    # Create tss.bed (1bp region at the TSS).
    # For '+' strand: TSS = start - 1 (0-based)
    # For '-' strand: TSS = end - 1   (0-based)
    ################################################
    def compute_tss(row):
        if row["strand"] == "+":
            return row["start"] - 1  # 0-based
        else:
            return row["end"] - 1    # 0-based

    df_genes["tss"] = df_genes.apply(compute_tss, axis=1)
    df_genes["tss_start"] = df_genes["tss"].clip(lower=0)
    df_genes["tss_end"] = df_genes["tss_start"] + 1

    df_genes["bed_chrom_tss"] = "chr" + df_genes["seqname"].astype(str)

    tss_bed_cols = ["bed_chrom_tss", "tss_start", "tss_end", "bed_name", "bed_score", "strand"]
    df_tss_bed = df_genes[tss_bed_cols].copy()
    df_tss_bed.columns = ["bed_chrom", "bed_start", "bed_end", "bed_name", "bed_score", "strand"]

    ##############################################
    # SORTING: define a custom chromosome order.
    # If your data has different or extra contigs,
    # you can update this list or handle unknowns.
    ##############################################
    chr_order = [
        "chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI",
        "chrVII", "chrVIII", "chrIX", "chrX", "chrXI", "chrXII",
        "chrXIII", "chrXIV", "chrXV", "chrXVI"
    ]
    chr_map = {name: i for i, name in enumerate(chr_order)}

    # Add a 'chr_rank' column to map chromosome names to a numeric rank
    df_genes_bed["chr_rank"] = df_genes_bed["bed_chrom"].map(chr_map).fillna(9999)
    df_tss_bed["chr_rank"]   = df_tss_bed["bed_chrom"].map(chr_map).fillna(9999)

    # Sort by chr_rank, then by bed_start
    df_genes_bed.sort_values(by=["chr_rank", "bed_start"], inplace=True)
    df_tss_bed.sort_values(by=["chr_rank", "bed_start"], inplace=True)

    # Drop the temporary 'chr_rank' column
    df_genes_bed.drop(columns=["chr_rank"], inplace=True)
    df_tss_bed.drop(columns=["chr_rank"], inplace=True)

    ###################################
    # Write out the final sorted BEDs #
    ###################################
    df_genes_bed.to_csv(genes_bed, sep="\t", header=False, index=False)
    df_tss_bed.to_csv(tss_bed, sep="\t", header=False, index=False)

    print(f"Done! Created:\n  {genes_bed}\n  {tss_bed}")

if __name__ == "__main__":
    main()
