#!/usr/bin/env python3
from optparse import OptionParser
import numpy as np
import os
import h5py
import pandas as pd
import pyranges as pr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
import pysam
import pyBigWig
from collections import defaultdict
import re

from yeast_helpers import *  # Assumes you have functions such as make_seq_1hot

FIG_SIZE = (7, 6)

###########################################################################
# Compute average for one time point (averaging over selected tracks)
###########################################################################
def compute_time_average_for_one_time(dataset, seq_idx, track_indices):
    # Convert track_indices to a numpy array for fancy indexing
    track_indices = np.array(track_indices, dtype=int)
    data_slice = dataset[seq_idx, :, :, track_indices]  # shape: (L, 4, #tracks)
    data_avg = data_slice.mean(axis=-1)  # shape: (L, 4)
    return data_avg

###########################################################################
# Parse attributes field from a GTF line.
###########################################################################
def parse_gtf_attributes(attr_str):
    attrs = {}
    for part in attr_str.split(';'):
        part = part.strip()
        if not part:
            continue
        # Expecting key "value" pairs.
        tokens = part.split(' ')
        if len(tokens) >= 2:
            key = tokens[0]
            val = tokens[1].replace('"', '')
            attrs[key] = val
    return attrs

###########################################################################
# Main routine to load input files, process bigwig coverages, and visualize 
# average coverage for fixed regions and gene regions.
###########################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--exp_dir', dest='exp_dir', default='', type='str',
                      help='Experiment directory [Default: %default]')
    parser.add_option('--plot_start', dest='plot_start', default=0, type='int',
                      help='Start position for fixed-region plotting [Default: %default]')
    parser.add_option('--plot_end', dest='plot_end', default=500, type='int',
                      help='End position for fixed-region plotting [Default: %default]')
    parser.add_option('--target_tf', dest='target_tf', default='MSN2_', type='str',
                      help='Target transcription factor identifier substring [Default: %default]')
    parser.add_option('--chrom', dest='chrom', default='chrXI', type='str',
                      help='Chromosome to extract coverage from (for fixed region plotting) [Default: %default]')
    parser.add_option('--out_dir', dest='out_dir', default='./plots', type='str',
                      help='Directory to store the output images [Default: %default]')
    parser.add_option('--bed', dest='bed', default='', type='str',
                      help='BED file with gene windows [Default: %default]')
    parser.add_option('--gtf', dest='gtf', default='', type='str',
                      help='GTF file for full gene regions [Default: %default]')
    parser.add_option('--exp_type', dest='exp_type', default='seq_experiment', type='str',
                      help='Experiment type [Default: %default]')
    parser.add_option('--libsize_file', dest='libsize_file', default='', type='str',
                      help='CSV file containing library sizes with columns "file" and "library_size" [Default: %default]')
    (options, args) = parser.parse_args()

    # Set directories and file paths (update these paths as needed)
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'
    target_f = f"{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt"

    # Open the fasta file using pysam
    fasta_open = pysam.Fastafile(fasta_file)

    # Read the target file into a DataFrame
    target_df = pd.read_csv(target_f, sep="\t")
    print("First few lines of the original target file:")
    print(target_df.head())

    # Filter the target DataFrame based on the target TF identifier (e.g., 'MSN2_')
    target_tf = options.target_tf
    target_df = target_df[target_df['identifier'].str.contains(target_tf)]
    print(f"\nTarget file filtered by '{target_tf}':")
    print(target_df.head())

    # Add a 'timepoint' column by extracting time information from the 'identifier'
    target_df['timepoint'] = target_df['identifier'].str.extract(r'_(T\d+)_', expand=False)
    # Also extract the numeric value to sort correctly.
    target_df['time_numeric'] = target_df['timepoint'].str.replace('T', '').astype(int)
    target_df = target_df.sort_values(by='time_numeric').reset_index(drop=True)
    print("\nTarget file with timepoint column added and sorted:")
    print(target_df.head())

    # Create a mapping from timepoint to a blue shade from light to dark.
    # T0 will be a lighter blue and later timepoints will be darker.
    unique_timepoints = sorted(target_df['timepoint'].unique(), key=lambda x: int(x.replace('T', '')))
    n_tp = len(unique_timepoints)
    timepoint_colors = {}
    for i, tp in enumerate(unique_timepoints):
        # Adjust the range (0.3 to 0.8) as desired; lower value is lighter.
        shade = 0.3 + (0.5 * i / (n_tp - 1)) if n_tp > 1 else 0.5
        timepoint_colors[tp] = mpl.cm.Blues(shade)
    print("\nTimepoint to color mapping:")
    for tp, col in timepoint_colors.items():
        print(f"{tp}: {col}")

    # Load library sizes if provided.
    libsize_dict = {}
    if options.libsize_file:
        try:
            libsize_df = pd.read_csv(options.libsize_file)
            # Expecting columns: 'file' and 'library_size'
            libsize_dict = dict(zip(libsize_df['file'], libsize_df['library_size']))
            print(f"\nLoaded library sizes for {len(libsize_dict)} files from {options.libsize_file}.")
        except Exception as e:
            print(f"Error reading library size file {options.libsize_file}: {e}")

    # Set the offset for track indices in the h5py dataset.
    track_offset = 1148

    print("\nStarting to process the scores file...")
    print("fasta_file: ", fasta_file)
    print("target_f: ", target_f)
    print("target_tf: ", target_tf)
    print("track_offset: ", track_offset)
    print("plot_start: ", options.plot_start)
    print("plot_end: ", options.plot_end)
    print("chrom: ", options.chrom)

    # Process the h5py scores file (if needed)
    with h5py.File(f"{options.exp_dir}/scores.h5", 'r') as h5_file:
        scores = ['logSED']  # You can add other score types if desired.
        for score in scores:
            dataset = h5_file[score]  # Expected shape: (N_sequences, L, 4, N_tracks)
            N_sequences = dataset.shape[0]
            L = dataset.shape[1]
            N_tracks = dataset.shape[-1]
            print(f"\nFor score '{score}': {N_sequences} sequences, length = {L}, {N_tracks} tracks.")

            # Group the target DataFrame by timepoint and process each group separately.
            for tp, group_df in target_df.groupby('timepoint'):
                print(f"\nProcessing timepoint: {tp}")
                # Get the raw track indices for the current timepoint and adjust by subtracting the offset.
                selected_track_indices = group_df['index'].astype(int).tolist()
                adjusted_track_indices = [idx - track_offset for idx in selected_track_indices]
                print("Raw track indices:", selected_track_indices)
                print("Adjusted track indices:", adjusted_track_indices)
                # (Optional) Process the dataset further using compute_time_average_for_one_time if needed.
    
    ###########################################################################
    # Part 1: Compute and store average bigwig coverage for a fixed region
    # (using options.plot_start/plot_end and options.chrom)
    ###########################################################################
    print("\nStarting to process bigwig coverage files for fixed region...")
    timepoint_coverage = {}
    for tp, group_df in target_df.groupby('timepoint'):
        print(f"\nProcessing bigwig files for timepoint: {tp}")
        coverages = []
        for idx, row in group_df.iterrows():
            bw_file = row['file']  # Each row should have a 'file' column with the path to the BigWig file.
            try:
                bw = pyBigWig.open(bw_file)
            except Exception as e:
                print(f"Error opening {bw_file}: {e}")
                continue
            # Extract coverage from the specified fixed region.
            cov = np.array(bw.values(options.chrom, options.plot_start, options.plot_end))
            cov = np.nan_to_num(cov)
            bw.close()

            print("bw_file: ", bw_file)
            # Normalize using library size if available.
            if libsize_dict and bw_file in libsize_dict:
                lib_size = libsize_dict[bw_file]
                if lib_size > 0:
                    norm_cov = cov / lib_size * 1e6
                else:
                    norm_cov = cov
            else:
                print("No libsize_dict or bw_file not in libsize_dict")
                norm_cov = cov
            coverages.append(norm_cov)
        if coverages:
            avg_cov = np.mean(coverages, axis=0)
            timepoint_coverage[tp] = avg_cov
            print(f"Computed average (normalized) coverage for timepoint {tp}.")
        else:
            print(f"No valid coverage data for timepoint {tp}.")

    # Create output directories if they do not exist.
    os.makedirs(options.out_dir, exist_ok=True)

    # Save a combined plot for the fixed region.
    if timepoint_coverage:
        combined_dir = os.path.join(options.out_dir, os.path.basename(os.path.normpath(options.exp_dir)), "fixed_region", "combined")
        os.makedirs(combined_dir, exist_ok=True)
        plt.figure(figsize=FIG_SIZE)
        x = np.arange(options.plot_start, options.plot_end)
        for tp in sorted(timepoint_coverage.keys(), key=lambda x: int(x.replace('T', ''))):
            avg_cov = timepoint_coverage[tp]
            plt.plot(x, avg_cov, label=f"Timepoint {tp}", color=timepoint_colors[tp])
        plt.xlabel("Genomic Position")
        plt.ylabel("Normalized Coverage (RPM)")
        plt.title("Combined Average BigWig Coverage by Timepoint (Fixed Region)")
        plt.legend()
        # Dynamically adjust y-limits for fixed region plot.
        all_vals = np.concatenate(list(timepoint_coverage.values()))
        y_min, y_max = all_vals.min(), all_vals.max()
        y_range = y_max - y_min
        if y_range == 0:
            plt.ylim([y_min - 1, y_max + 1])
        else:
            plt.ylim([max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range])
        combined_filename = os.path.join(combined_dir, "combined_coverage.png")
        plt.tight_layout()
        plt.savefig(combined_filename)
        plt.close()
        print(f"Combined fixed-region plot saved to {combined_filename}")
    else:
        print("No coverage data to plot for the fixed region.")

    ###########################################################################
    # Part 2: Process gene regions based on a BED file and a GTF file.
    # For each gene in the BED file, obtain the full gene region from the GTF,
    # extend 50 bp upstream and downstream, then compute the summed (aggregated)
    # normalized coverage for each gene across timepoints. Two plots are generated:
    # a bar chart and a trend (line) plot.
    ###########################################################################
    if options.bed and options.gtf:
        print("\nProcessing gene regions from BED and GTF files...")
        # Read the BED file. Assuming no header and columns: chrom, start, end, gene, score, strand.
        bed_df = pd.read_csv(options.bed, sep="\t", header=None,
                             names=["chrom", "start", "end", "gene", "score", "strand"])
        print("BED file preview:")
        print(bed_df.head())

        # Load the GTF file. Skip comment lines.
        gtf_df = pd.read_csv(options.gtf, sep="\t", comment='#', header=None,
                             names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
        # Filter for rows with the 'gene' feature.
        gene_gtf_df = gtf_df[gtf_df["feature"]=="gene"].copy()

        # Build a mapping from gene name to its region (chrom, start, end, strand).
        gene_region_dict = {}
        for idx_gtf, row in gene_gtf_df.iterrows():
            attrs = parse_gtf_attributes(row["attribute"])
            gene_name = attrs.get("gene", None)
            if gene_name is None:
                gene_name = attrs.get("gene_name", None)
            if gene_name is None:
                gene_name = attrs.get("gene_id", None)
            if gene_name:
                if gene_name in gene_region_dict:
                    cur_chrom, cur_start, cur_end, cur_strand = gene_region_dict[gene_name]
                    new_start = min(cur_start, row["start"])
                    new_end = max(cur_end, row["end"])
                    gene_region_dict[gene_name] = (row["chrom"], new_start, new_end, row["strand"])
                else:
                    gene_region_dict[gene_name] = (row["chrom"], row["start"], row["end"], row["strand"])

        # For each gene in the BED file, look up its full region, extend by 50 bp,
        # and then compute & plot the summed normalized bigwig coverage (gene expression)
        # from each timepoint.
        for idx, bed_row in bed_df.iterrows():
            gene = bed_row["gene"]
            if gene not in gene_region_dict:
                print(f"Gene {gene} not found in GTF file.")
                continue
            region_chrom, region_start, region_end, region_strand = gene_region_dict[gene]
            region_chrom = 'chr' + region_chrom if not region_chrom.startswith('chr') else region_chrom
            # Extend the region by 50 bp upstream and 50 bp downstream.
            extended_start = max(0, region_start - 50)
            extended_end = region_end + 50
            print(f"Gene {gene} region: {region_chrom}:{extended_start}-{extended_end}, strand {region_strand}")
           
            gene_timepoint_cov = {}
            for tp, group_df in target_df.groupby('timepoint'):
                coverages = []
                for i, row in group_df.iterrows():
                    bw_file = row['file']
                    try:
                        bw = pyBigWig.open(bw_file)
                    except Exception as e:
                        print(f"Error opening {bw_file}: {e}")
                        continue
                    try:
                        cov = np.array(bw.values(region_chrom, extended_start, extended_end))
                    except Exception as e:
                        print(f"Error fetching values from {bw_file} for region {region_chrom}:{extended_start}-{extended_end}: {e}")
                        cov = np.zeros(extended_end - extended_start)
                    cov = np.nan_to_num(cov)
                    bw.close()
                    # Normalize using library size if available.
                    if libsize_dict and bw_file in libsize_dict:
                        lib_size = libsize_dict[bw_file]
                        if lib_size > 0:
                            norm_cov = cov / lib_size * 1e6
                        else:
                            norm_cov = cov
                    else:
                        norm_cov = cov
                    coverages.append(norm_cov)
                if coverages:
                    avg_cov = np.mean(coverages, axis=0)
                    gene_timepoint_cov[tp] = avg_cov
            # Create output directory for this gene.
            exp_basename = os.path.basename(os.path.normpath(options.exp_dir))
            gene_dir = os.path.join(options.out_dir, options.exp_type, exp_basename,
                                    f"{idx:03d}_{gene}_{region_chrom}_{extended_start}-{extended_end}")
            os.makedirs(gene_dir, exist_ok=True)

            # Compute a single expression (summed normalized coverage) per timepoint.
            gene_expression = {}
            for tp, cov_profile in gene_timepoint_cov.items():
                gene_expression[tp] = cov_profile.sum()

            # Sort timepoints numerically (assuming format "T0", "T1", ...)
            sorted_tps = sorted(gene_expression.keys(), key=lambda x: int(x.replace('T', '')))
            sorted_expr = [gene_expression[tp] for tp in sorted_tps]

            # Plot 1: Bar chart for gene expression.
            plt.figure(figsize=FIG_SIZE)
            plt.bar(sorted_tps, sorted_expr, color=[timepoint_colors[tp] for tp in sorted_tps])
            plt.xlabel("Timepoint")
            plt.ylabel("Gene Expression (Summed RPM)")
            plt.title(f"Gene Expression for {gene}")
            # Dynamically adjust y-limits for bar chart.
            y_min, y_max = min(sorted_expr), max(sorted_expr)
            y_range = y_max - y_min
            if y_range == 0:
                plt.ylim([y_min - 1, y_max + 1])
            else:
                plt.ylim([max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range])
            plt.tight_layout()
            bar_filename = os.path.join(gene_dir, f"{gene}_bar_chart.png")
            plt.savefig(bar_filename)
            plt.close()
            print(f"Bar chart for gene {gene} saved to {bar_filename}")

            # Plot 2: Trend plot (line plot) for gene expression.
            plt.figure(figsize=FIG_SIZE)
            plt.plot(sorted_tps, sorted_expr, marker='o', linestyle='-', color='blue')
            plt.xlabel("Timepoint")
            plt.ylabel("Gene Expression (Summed RPM)")
            plt.title(f"Gene Expression Trend for {gene}")
            # Dynamically adjust y-limits for trend plot.
            if y_range == 0:
                plt.ylim([y_min - 1, y_max + 1])
            else:
                plt.ylim([max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range])
            plt.tight_layout()
            trend_filename = os.path.join(gene_dir, f"{gene}_trend_plot.png")
            plt.savefig(trend_filename)
            plt.close()
            print(f"Trend plot for gene {gene} saved to {trend_filename}")

if __name__ == "__main__":
    main()
