#!/usr/bin/env python

import pandas as pd
import argparse
import os
from collections import defaultdict

def load_motif_hits(bed_file):
    """
    Load motif hits from a BED file.
    Expects at least 6 columns: chrom, start, end, name, score, qval.
    Converts the qval column to numeric.
    """
    df = pd.read_csv(bed_file, sep='\t', header=None)
    if df.shape[1] < 6:
        raise ValueError("Motif BED file must have at least 6 columns: chrom, start, end, name, score, qval")
    df.columns = ["chrom", "start", "end", "name", "qval", "strand"]
    # Convert qval to numeric; if conversion fails, NaN is set
    # print(df["qval"])
    df["qval"] = pd.to_numeric(df["qval"], errors='coerce')
    return df

def load_chip_peaks(peak_file):
    """
    Load ChIP-peak regions from a BED file.
    Expects at least 6 columns: chrom, start, end, name, score, strand.
    """
    df = pd.read_csv(peak_file, sep='\t', header=None)
    if df.shape[1] < 6:
        raise ValueError("ChIP-peak BED file must have at least 6 columns: chrom, start, end, name, score, strand")
    df.columns = ["chrom", "start", "end", "name", "score", "strand"]
    return df

def check_chip_peak_overlap(motif_df, chip_peak_df):
    """
    For each motif hit, check if it overlaps any ChIP-peak region.
    
    Overlap is defined by the condition:
       motif_start < chip_peak_end and motif_end > chip_peak_start.
    
    Adds a new boolean column 'chip_peak_overlap' to motif_df.
    """
    # Build a dictionary of chip peaks by chromosome for fast lookup.
    chip_peaks_by_chrom = defaultdict(list)
    for idx, row in chip_peak_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        chip_peaks_by_chrom[chrom].append((start, end))
    
    overlaps = []
    for idx, row in motif_df.iterrows():
        chrom = row["chrom"]
        motif_start = row["start"]
        motif_end = row["end"]
        found_overlap = False
        # Check if there are chip peaks on the same chromosome.
        if chrom in chip_peaks_by_chrom:
            for peak_start, peak_end in chip_peaks_by_chrom[chrom]:
                if motif_start < peak_end and motif_end > peak_start:
                    found_overlap = True
                    break
        overlaps.append(found_overlap)
    
    motif_df["chip_peak_overlap"] = overlaps
    return motif_df

def compute_overlap_ratio(motif_df):
    """
    Compute the ratio of motif hits that overlap a ChIP-peak.
    """
    total = len(motif_df)
    if total == 0:
        return 0.0
    overlapping = motif_df["chip_peak_overlap"].sum()
    return overlapping / total

def main():
    parser = argparse.ArgumentParser(
        description="Compare TF-Modisco motif hits to ChIP-peak regions from a BED file and compute overlap ratio."
    )
    parser.add_argument("--motif_bed", required=True, help="Path to TF-Modisco motif hits BED file.")
    parser.add_argument("--chip_peak_bed", required=True, help="Path to ChIP-peak BED file.")
    parser.add_argument("--output", default="motif_chip_peak_overlap.csv",
                        help="Output CSV file with motif hits and ChIP-peak overlap information.")
    parser.add_argument("--qval_threshold", type=float, default=0.5,
                        help="Threshold for qval to filter motif hits (default: 0.5).")
    args = parser.parse_args()
    
    # Load motif hits and chip peaks.
    motif_df = load_motif_hits(args.motif_bed)
    chip_peak_df = load_chip_peaks(args.chip_peak_bed)
    
    # Filter out motif hits with qval greater than the threshold.
    initial_count = len(motif_df)
    motif_df = motif_df[motif_df["qval"] <= args.qval_threshold]
    filtered_count = len(motif_df)
    print(f"Filtered motif hits based on qval threshold {args.qval_threshold}: {filtered_count} remaining out of {initial_count}")
    
    # Check for overlap between motif hits and chip peaks.
    motif_df = check_chip_peak_overlap(motif_df, chip_peak_df)
    
    # Compute overall overlap ratio.
    overlap_ratio = compute_overlap_ratio(motif_df)
    num_overlapping = motif_df["chip_peak_overlap"].sum()
    total_hits = len(motif_df)
    print(f"Overlap ratio: {overlap_ratio:.3f} ({num_overlapping} overlapping out of {total_hits})")
    
    # Write out the results.
    motif_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
