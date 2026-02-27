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
from collections import defaultdict
import re

from yeast_helpers import *  # Assumes you have functions such as make_seq_1hot

##########################################
# Helper function to plot ACGT letters   #
##########################################
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
        "UP": TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
        "DN": TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
        "(": TextPath((-0.25, 0), "(", size=1, prop=fp),
        ".": TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")": TextPath((-0.1, 0), ")", size=1, prop=fp)
    }
    COLOR_SCHEME = {
        'G': 'orange',
        'A': 'green',
        'C': 'blue',
        'T': 'red',
        'UP': 'green',
        'DN': 'red',
        '(': 'black',
        '.': 'black',
        ')': 'black'
    }
    text = LETTERS[letter]
    chosen_color = COLOR_SCHEME[letter] if color is None else color
    t = mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)
    return p

#############################################
# Function to plot sequence logo            #
#############################################
def plot_seq_scores(importance_scores, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, save_figs=False, fig_name="default"):
    importance_scores = importance_scores.T
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    # Plot letters ordered by absolute magnitude (largest first)
    for j in range(importance_scores.shape[1]):
        sort_index = np.argsort(-np.abs(importance_scores[:, j]))
        pos_height = 0.0
        neg_height = 0.0
        for ii in range(4):
            i = sort_index[ii]
            if i == 0:
                nt = 'A'
            elif i == 1:
                nt = 'C'
            elif i == 2:
                nt = 'G'
            elif i == 3:
                nt = 'T'
            nt_prob = importance_scores[i, j]
            if nt_prob >= 0.0:
                dna_letter_at(letter=nt, x=j + 0.5, y=pos_height, yscale=nt_prob, ax=ax)
                pos_height += nt_prob
            else:
                dna_letter_at(letter=nt, x=j + 0.5, y=neg_height, yscale=nt_prob, ax=ax)
                neg_height += nt_prob
    plt.xlim((0, importance_scores.shape[1]))
    plt.yticks([] if not plot_y_ticks else None)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        plt.ylim(np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
                 np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores)))
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
    plt.show()

#############################################
# Plot sequence logo given the PWM scores    #
#############################################
def visualize_input_ism(att_grad_wt, plot_start=0, plot_end=100, save_figs=False, fig_name='', figsize=(12, 3)):
    scores_wt = att_grad_wt[plot_start:plot_end, :]
    y_min = np.min(scores_wt)
    y_max = np.max(scores_wt)
    y_max_abs = max(np.abs(y_min), np.abs(y_max))
    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs
    plot_seq_scores(scores_wt, y_min=y_min, y_max=y_max,
                    figsize=figsize, plot_y_ticks=False,
                    save_figs=save_figs, fig_name=fig_name)
    
#############################################
# Compute average for one time point       #
#############################################
def compute_time_average_for_one_time(dataset, seq_idx, track_indices):
    # Convert and sort track_indices to a numpy array for fancy indexing
    track_indices = np.sort(np.array(track_indices, dtype=int))
    data_slice = dataset[seq_idx, :, :, track_indices]  # shape: (L, 4, #tracks)
    data_avg = data_slice.mean(axis=-1)  # shape: (L, 4)
    return data_avg


###########################################################################
# Main routine: plot two logos per sequence for T45 vs. T0 differences        #
###########################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--exp_dir', dest='exp_dir', default='', type='str',
                      help='Experiment directory [Default: %default]')
    parser.add_option('--plot_start', dest='plot_start', default=0, type='int',
                      help='Start position for plotting [Default: %default]')
    parser.add_option('--plot_end', dest='plot_end', default=500, type='int',
                      help='End position for plotting [Default: %default]')
    parser.add_option('--target_tf', dest='target_tf', default='MSN2_', type='str',
                      help='Target transcription factor identifier substring [Default: %default]')
    (options, args) = parser.parse_args()

    # Set directories and file paths (update these paths as needed)
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'
    target_f = f"{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt"

    # Open the fasta file using pysam
    fasta_open = pysam.Fastafile(fasta_file)

    # Read and filter the target file
    target_df = pd.read_csv(target_f, sep="\t")
    print("First few lines of the original target file:")
    print(target_df.head())

    target_tf = options.target_tf
    target_df = target_df[target_df['identifier'].str.contains(target_tf)]
    print(f"\nTarget file filtered by '{target_tf}':")
    print(target_df.head())

    # Extract time point (e.g., 'T0', 'T1', ...) from the identifier using regex.
    target_df['timepoint'] = target_df['identifier'].str.extract(r'_(T\d+)_', expand=False)
    # Sort by time so that the order is consistent.
    target_df = target_df.sort_values(by='timepoint').reset_index(drop=True)
    print("\nTarget file with timepoint column added:")
    print(target_df.head())

    # Get the track indices for T0 and T45
    target_df_t0 = target_df[target_df['timepoint'] == 'T0']
    target_df_T45 = target_df[target_df['timepoint'] == 'T45']
    if target_df_t0.empty or target_df_T45.empty:
        raise ValueError("Target file does not contain both T0 and T45 timepoints.")

    track_offset = 1148
    selected_track_indices_t0 = target_df_t0['index'].astype(int).tolist()
    adjusted_track_indices_t0 = [idx - track_offset for idx in selected_track_indices_t0]
    selected_track_indices_T45 = target_df_T45['index'].astype(int).tolist()
    adjusted_track_indices_T45 = [idx - track_offset for idx in selected_track_indices_T45]

    print("T0 raw track indices:", selected_track_indices_t0)
    print("T0 adjusted track indices:", adjusted_track_indices_t0)
    print("T45 raw track indices:", selected_track_indices_T45)
    print("T45 adjusted track indices:", adjusted_track_indices_T45)

    # Open the scores file (assumes h5 file with a dataset named 'logSED')
    with h5py.File(f"{options.exp_dir}/scores.h5", 'r') as h5_file:
        scores = ['logSED']
        for score in scores:
            dataset = h5_file[score]  # shape: (N_sequences, L, 4, N_tracks)
            N_sequences = dataset.shape[0]
            L = dataset.shape[1]
            N_tracks = dataset.shape[-1]
            print(f"\nFor score '{score}': {N_sequences} sequences, length = {L}, {N_tracks} tracks.")

            # Create output directories for the two types of difference logos
            diff_dir_norm = f"{options.exp_dir}/dna_logo_example/{options.target_tf}logos/{score}/T0_vs_T45_diff_normalized"
            diff_dir_raw = f"{options.exp_dir}/dna_logo_example/{options.target_tf}logos/{score}/T0_vs_T45_diff_raw"
            os.makedirs(diff_dir_norm, exist_ok=True)
            os.makedirs(diff_dir_raw, exist_ok=True)

            # Lists to collect tensors for NPZ saving
            diff_tensors_norm = []
            diff_tensors_raw = []

            # Process each sequence
            for seq_idx in range(N_sequences):
                plot_start = options.plot_start
                plot_end = options.plot_end

                # Read sequence coordinates
                chr_16k = str(h5_file['chr'][seq_idx].decode("utf-8"))
                start_16k = int(h5_file['start'][seq_idx])
                end_16k = int(h5_file['end'][seq_idx])
                print(f"[score={score}] seq_idx={seq_idx} --> {chr_16k}:{start_16k}-{end_16k}")

                # Get the one-hot representation of the reference sequence
                sequence_one_hot = make_seq_1hot(
                    fasta_open,
                    chr_16k,
                    start_16k,
                    end_16k,
                    end_16k - start_16k
                ).astype("float32")
                print("\t>> sequence_one_hot shape:", sequence_one_hot.shape)

                # Compute PWM averages for T0 and T45
                pwm_t0 = compute_time_average_for_one_time(dataset, seq_idx, track_indices=adjusted_track_indices_t0)
                pwm_T45 = compute_time_average_for_one_time(dataset, seq_idx, track_indices=adjusted_track_indices_T45)

                ##########################################
                # 1. Normalized Difference Calculation   #
                ##########################################
                # Mean-normalize each PWM (subtract per-position average)
                mean_t0 = np.mean(pwm_t0, axis=-1, keepdims=True)
                pwm_norm_t0 = pwm_t0 - np.tile(mean_t0, (1, 4))
                mean_T45 = np.mean(pwm_T45, axis=-1, keepdims=True)
                pwm_norm_T45 = pwm_T45 - np.tile(mean_T45, (1, 4))
                # Compute the normalized difference (T45 minus T0)
                diff_norm = pwm_norm_T45 - pwm_norm_t0
                # Multiply by one-hot so that only the reference nucleotide is displayed
                diff_norm_viz = diff_norm * sequence_one_hot
                # Limit the region to be plotted
                diff_norm_viz = diff_norm_viz[plot_start:plot_end, :]

                # Define output filename for the normalized difference logo
                fig_out_norm = f"{diff_dir_norm}/logo_diff_norm_seq_{seq_idx}_{chr_16k}_{start_16k}_{end_16k}"
                visualize_input_ism(
                    diff_norm_viz,
                    plot_start=0,
                    plot_end=diff_norm_viz.shape[0],
                    save_figs=True,
                    fig_name=fig_out_norm,
                    figsize=(100, 3)
                )
                print(f"Saved normalized difference DNA logo for sequence {seq_idx} at {fig_out_norm}.png")
                diff_tensors_norm.append(diff_norm.T)  # Transpose to shape (4, L)

                ##########################################
                # 2. Raw Difference Calculation          #
                ##########################################
                # Simply compute the raw difference (without mean normalization)
                diff_raw = pwm_T45 - pwm_t0
                diff_raw_viz = diff_raw * sequence_one_hot
                diff_raw_viz = diff_raw_viz[plot_start:plot_end, :]

                # Define output filename for the raw difference logo
                fig_out_raw = f"{diff_dir_raw}/logo_diff_raw_seq_{seq_idx}_{chr_16k}_{start_16k}_{end_16k}"
                visualize_input_ism(
                    diff_raw_viz,
                    plot_start=0,
                    plot_end=diff_raw_viz.shape[0],
                    save_figs=True,
                    fig_name=fig_out_raw,
                    figsize=(100, 3)
                )
                print(f"Saved raw difference DNA logo for sequence {seq_idx} at {fig_out_raw}.png")
                diff_tensors_raw.append(diff_raw.T)  # Transpose to shape (4, L)

            # Stack and save NPZ files for normalized differences
            diff_tensors_norm = np.stack(diff_tensors_norm, axis=0)  # Shape: (N_sequences, 4, L)
            out_npz_norm = f"{diff_dir_norm}/diff_normalized.npz"
            np.savez(out_npz_norm, arr_0=diff_tensors_norm)
            print(f"Saved concatenated normalized difference tensors at {out_npz_norm}")

            # Stack and save NPZ files for raw differences
            diff_tensors_raw = np.stack(diff_tensors_raw, axis=0)  # Shape: (N_sequences, 4, L)
            out_npz_raw = f"{diff_dir_raw}/diff_raw.npz"
            np.savez(out_npz_raw, arr_0=diff_tensors_raw)
            print(f"Saved concatenated raw difference tensors at {out_npz_raw}")

if __name__ == "__main__":
    main()
