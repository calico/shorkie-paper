#!/usr/bin/env python

import pandas as pd
import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def load_motif_hits(bed_file):
    """
    Load motif hits from a BED file.
    Expects at least 6 columns: chrom, start, end, name, score, strand.
    """
    df = pd.read_csv(bed_file, sep='\t', header=None)
    if df.shape[1] < 6:
        raise ValueError("Motif BED file must have at least 6 columns: chrom, start, end, name, score, strand")
    df.columns = ["chrom", "start", "end", "name", "score", "strand"]
    return df

def load_chip_peaks(peak_file):
    """
    Load Chip‑peak regions from a BED file.
    Expects at least 6 columns: chrom, start, end, name, score, strand.
    """
    df = pd.read_csv(peak_file, sep='\t', header=None)
    if df.shape[1] < 6:
        raise ValueError("Chip‑peak BED file must have at least 6 columns: chrom, start, end, name, score, strand")
    df.columns = ["chrom", "start", "end", "name", "score", "strand"]
    return df

def check_chip_peak_overlap(motif_df, chip_peak_df):
    """
    For each motif hit, check if it overlaps any Chip‑peak region.
    Overlap is defined as: motif_start < peak_end and motif_end > peak_start.
    Adds a boolean column 'chip_peak_overlap' to motif_df.
    """
    # Build a dictionary of chip peaks by chromosome for faster lookup.
    chip_peaks_by_chrom = defaultdict(list)
    for idx, row in chip_peak_df.iterrows():
        chrom = row["chrom"]
        chip_peaks_by_chrom[chrom].append((row["start"], row["end"]))
    
    overlaps = []
    for idx, row in motif_df.iterrows():
        chrom = row["chrom"]
        m_start = row["start"]
        m_end = row["end"]
        found = False
        if chrom in chip_peaks_by_chrom:
            for peak_start, peak_end in chip_peaks_by_chrom[chrom]:
                if m_start < peak_end and m_end > peak_start:
                    found = True
                    break
        overlaps.append(found)
    motif_df["chip_peak_overlap"] = overlaps
    return motif_df

def find_top_windows(motif_df, window_size=16384, top_n=20):
    """
    Using the overlapping motif hits (filtered in motif_df), find windows of size 'window_size'
    (using the motif midpoints) that contain the highest number of hits.
    
    Returns a list of tuples: (chrom, window_start, window_end, count)
    """
    windows = []
    # Process each chromosome separately.
    for chrom in motif_df['chrom'].unique():
        df_chrom = motif_df[motif_df['chrom'] == chrom].copy()
        if df_chrom.empty:
            continue
        # Compute the midpoint for each motif hit.
        df_chrom['mid'] = ((df_chrom['start'] + df_chrom['end']) // 2).astype(int)
        mids = sorted(df_chrom['mid'].tolist())
        n = len(mids)
        i = 0
        j = 0
        while i < n:
            # Slide j forward until mids[j] is outside the window starting at mids[i].
            while j < n and mids[j] < mids[i] + window_size:
                j += 1
            count = j - i
            window_start = mids[i]
            window_end = window_start + window_size
            windows.append((chrom, window_start, window_end, count))
            i += 1
    # Sort windows by count in descending order and return the top_n.
    windows_sorted = sorted(windows, key=lambda x: x[3], reverse=True)
    return windows_sorted[:top_n]

def find_random_windows(motif_df, window_size=16384, top_n=20):
    """
    Randomly sample windows from the overlapping motif hits.
    For each randomly selected motif hit, use its midpoint to define a window of size 'window_size'.
    Returns a list of tuples: (chrom, window_start, window_end, count),
    where count is the number of overlapping motifs in that window.
    """
    sample_n = min(top_n, len(motif_df))
    # For reproducibility, you can set a random_state.
    sampled = motif_df.sample(n=sample_n, random_state=42)
    windows = []
    for idx, row in sampled.iterrows():
        chrom = row['chrom']
        mid = (row['start'] + row['end']) // 2
        win_start = max(0, mid - window_size // 2)
        win_end = win_start + window_size
        # Count number of overlapping motifs in this window.
        subset = motif_df[(motif_df['chrom'] == chrom) &
                          (motif_df['start'] < win_end) &
                          (motif_df['end'] > win_start)]
        count = len(subset)
        windows.append((chrom, win_start, win_end, count))
    return windows

def plot_window(chrom, win_start, win_end, chip_peak_df, motif_df, out_file):
    """
    Create a plot for the given window (chrom, win_start, win_end) showing two tracks:
      - Top track: Chip‑peak regions in the window.
      - Bottom track: TF‑Modisco motif hits in the window.
    Regions are drawn as rectangles relative to the window.
    """
    # Filter Chip‑peak regions overlapping the window.
    chip_df = chip_peak_df[(chip_peak_df['chrom'] == chrom) &
                           (chip_peak_df['end'] > win_start) &
                           (chip_peak_df['start'] < win_end)]
    # Filter motif hits (all, not just overlapping ones) that intersect the window.
    motif_win_df = motif_df[(motif_df['chrom'] == chrom) &
                            (motif_df['end'] > win_start) &
                            (motif_df['start'] < win_end)]
    
    # Create a figure with two subplots (tracks).
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 4))
    
    # --- Top track: Chip‑peak regions ---
    axes[0].set_title(f"Chip‑Exo Peaks: {chrom}:{win_start}-{win_end}")
    for idx, row in chip_df.iterrows():
        # Clip the region to the window boundaries.
        start = max(row['start'], win_start)
        end = min(row['end'], win_end)
        width = end - start
        # Draw rectangle: x position relative to win_start.
        rect = patches.Rectangle((start - win_start, 0.3), width, 0.4, color='blue', alpha=0.7)
        axes[0].add_patch(rect)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    
    # --- Bottom track: Motif hits ---
    axes[1].set_title(f"TF‑Modisco Motif Hits: {chrom}:{win_start}-{win_end}")
    for idx, row in motif_win_df.iterrows():
        start = max(row['start'], win_start)
        end = min(row['end'], win_end)
        width = end - start
        rect = patches.Rectangle((start - win_start, 0.3), width, 0.4, color='red', alpha=0.7)
        axes[1].add_patch(rect)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    
    # Set x-axis limits: coordinate relative to window.
    axes[1].set_xlim(0, win_end - win_start)
    # Label the x-axis in genomic coordinates.
    xticks = axes[1].get_xticks()
    axes[1].set_xticklabels([int(t + win_start) for t in xticks])
    
    axes[1].set_xlabel("Genomic Coordinate")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 20 windows (size 16384 bp) with TF‑Modisco motif hits and Chip‑Exo peaks. "
                    "Either select the top windows by overlapping motif count or randomly select windows."
    )
    parser.add_argument("--motif_bed", required=True, help="Path to TF‑Modisco motif hits BED file.")
    parser.add_argument("--chip_peak_bed", required=True, help="Path to Chip‑peak BED file.")
    parser.add_argument("--output_dir", default="window_plots", help="Directory to save the window plots.")
    parser.add_argument("--window_size", type=int, default=16384, help="Window size in bp (default: 16384).")
    parser.add_argument("--top_n", type=int, default=20, help="Number of windows to visualize (default: 20).")
    parser.add_argument("--random_windows", action="store_true",
                        help="If specified, randomly select windows instead of using the top windows.")
    args = parser.parse_args()
    
    # Load input data.
    motif_df = load_motif_hits(args.motif_bed)
    chip_peak_df = load_chip_peaks(args.chip_peak_bed)
    
    # Flag motif hits that overlap Chip‑peaks.
    motif_df = check_chip_peak_overlap(motif_df, chip_peak_df)
    
    # Filter for motifs that overlap Chip‑peaks.
    overlapping_motifs = motif_df[motif_df["chip_peak_overlap"] == True].copy()
    if overlapping_motifs.empty:
        print("No overlapping motifs found!")
        return
    
    # Determine windows: either top windows by count or random windows.
    if args.random_windows:
        windows = find_random_windows(overlapping_motifs, window_size=args.window_size, top_n=args.top_n)
        print("Randomly selected windows:")
    else:
        windows = find_top_windows(overlapping_motifs, window_size=args.window_size, top_n=args.top_n)
        print("Top windows (chrom, start, end, count):")
    for win in windows:
        print(win)
    
    # Create output directory if needed.
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate a plot for each window.
    for idx, window in enumerate(windows):
        chrom, win_start, win_end, count = window
        out_file = os.path.join(args.output_dir, f"window_{idx+1}_{chrom}_{win_start}_{win_end}.png")
        print(f"Plotting window {idx+1}: {chrom}:{win_start}-{win_end} with {count} motif hits.")
        plot_window(chrom, win_start, win_end, chip_peak_df, motif_df, out_file)
    
if __name__ == "__main__":
    main()
