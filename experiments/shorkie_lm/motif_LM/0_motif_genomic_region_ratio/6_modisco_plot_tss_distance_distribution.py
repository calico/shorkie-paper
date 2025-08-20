#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def load_distances(filepath):
    """
    Loads the last column (distance) from a bedtools closest output file.
    Returns a Pandas Series of distances (ints).
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, sep="\t", header=None)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    if df.shape[1] < 13:
        print(f"Skipping {filepath}: unexpected number of columns.")
        return None
    
    # Distance is the last column
    distances = df.iloc[:, -1]
    distances = pd.to_numeric(distances, errors="coerce").dropna()
    return distances

def plot_motif_vs_bg_distance(motif_distances, bg_distances, motif_id, out_png):
    """
    Plots an overlaid histogram of motif_distances vs. bg_distances,
    saving the figure to out_png.
    """
    plt.figure(figsize=(8, 6))

    # Real motif distribution
    plt.hist(
        motif_distances, bins=100, color="blue", alpha=0.5,
        edgecolor="black", label="Motif"
    )
    # Background distribution
    plt.hist(
        bg_distances, bins=100, color="red", alpha=0.5,
        edgecolor="black", label="Random BG"
    )

    plt.title(f"Motif {motif_id} - Distance to Closest TSS")
    plt.xlabel("Distance to Closest TSS (bp)")
    plt.ylabel("Frequency")
    # Vertical line at 0 to indicate TSS
    plt.axvline(0, color="black", linestyle="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved: {out_png}")

def main():
    """
    Usage:
        python plot_motif_vs_bg_distance.py <tss_stats_dir> <output_dir>
    Where <tss_stats_dir> has files like:
        motif_1_tss_stats.txt       (real)
        motif_1_bg_tss_stats.txt    (background)
        motif_2_tss_stats.txt
        motif_2_bg_tss_stats.txt
        ...
    """
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <tss_stats_dir> <output_dir>")
        sys.exit(1)

    tss_stats_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Loop over the possible motif IDs, e.g. 1..80
    for i in range(1, 81):
        motif_file = os.path.join(tss_stats_dir, f"motif_{i}_tss_stats.txt")
        bg_file = os.path.join(tss_stats_dir, f"motif_{i}_bg_tss_stats.txt")
        
        if not os.path.exists(motif_file) or not os.path.exists(bg_file):
            # Skip if either file is missing
            continue

        # Load distances
        motif_distances = load_distances(motif_file)
        bg_distances = load_distances(bg_file)

        if motif_distances is None or bg_distances is None:
            continue  # skip if loading failed
        
        # Plot
        out_png = os.path.join(output_dir, f"motif_{i}_tss_compare.png")
        plot_motif_vs_bg_distance(motif_distances, bg_distances, i, out_png)

if __name__ == "__main__":
    main()
