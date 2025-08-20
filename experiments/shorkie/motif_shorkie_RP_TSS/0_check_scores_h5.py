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

from yeast_helpers import *  # Assumes you have this module with functions such as make_seq_1hot

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
# Plot sequence logo with matched y-axis    #
#############################################
def visualize_input_ism(att_grad_wt, plot_start=0, plot_end=100, save_figs=False, fig_name='', figsize=(12, 3)):
    scores_wt = att_grad_wt[plot_start:plot_end, :]

    # Get logo bounds
    y_min = np.min(scores_wt)
    y_max = np.max(scores_wt)
    y_max_abs = max(np.abs(y_min), np.abs(y_max))
    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs

    plot_seq_scores(
        scores_wt, y_min=y_min, y_max=y_max,
        figsize=figsize,
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + '_wt',
    )


#############################################
# Function to plot sequence logo            #
#############################################
def plot_seq_scores(importance_scores, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, save_figs=False, fig_name="default"):
    importance_scores = importance_scores.T

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot all 4 letters per position (sorted by magnitude)
    for j in range(importance_scores.shape[1]):
        sort_index = np.argsort(-np.abs(importance_scores[:, j]))  # largest magnitude first

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
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=pos_height,
                    yscale=nt_prob,
                    ax=ax,
                    color=None
                )
                pos_height += nt_prob
            else:
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=neg_height,
                    yscale=nt_prob,
                    ax=ax,
                    color=None
                )
                neg_height += nt_prob

    plt.xlim((0, importance_scores.shape[1]))
    plt.yticks([] if not plot_y_ticks else None)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        plt.ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
        )
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
    plt.show()


###########################################################################
# Compute average for one time point (averaging over selected tracks)        #
###########################################################################
def compute_time_average_for_one_time(dataset, seq_idx, track_indices):
    # Convert track_indices to a numpy array for fancy indexing
    track_indices = np.array(track_indices, dtype=int)
    data_slice = dataset[seq_idx, :, :, track_indices]  # shape: (L, 4, #tracks)
    data_avg = data_slice.mean(axis=-1)  # shape: (L, 4)
    return data_avg

def select_tracks_for_TS(target_df, TS):
    selected_target_df = target_df[target_df['identifier'].str.contains(TS)]
    return selected_target_df

###########################################################################
# Main routine to load & plot averaged DNA logos for T0 tracks              #
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
    (options, args) = parser.parse_args()

    # Example root directory and fasta file (update as needed)
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'
    fasta_open = pysam.Fastafile(fasta_file)

    # Read and filter the target file
    target_f = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt"
    target_df = pd.read_csv(target_f, sep="\t")
    print("First few lines of target file:")
    print(target_df.head())
    print("Number of rows in target file:", len(target_df)) 
    
    # Select only rows for target tracks (e.g., MSN2)
    target_df = select_tracks_for_TS(target_df, 'MSN2_')
    print("Filtered target_df (T0):")
    print(target_df.head())
    print("Number of rows in filtered target file:", len(target_df))    

    # Get the selected track indices (as a list)
    selected_track_indices = target_df['index'].astype(int).tolist()
    print("Selected track indices (raw):", selected_track_indices)
    
    # The h5py dataset for 'logSED' has 3053 tracks corresponding to global track indices
    # starting at 1148 and going to 4200. Thus, to index into the dataset (which is 0-indexed),
    # subtract the offset (1148) from each target index.
    track_offset = 1148
    selected_track_indices = [idx - track_offset for idx in selected_track_indices]
    print("Converted track indices (dataset indices):", selected_track_indices)

    with h5py.File(f"{options.exp_dir}/scores.h5", 'r') as h5_file:
        scores = ['logSED']  # or other score types if desired
        for score in scores:
            dataset = h5_file[score]  # shape: (N_sequences, L, 4, N_tracks)
            print(f"Loaded dataset for score '{score}'.")
            print("Dataset shape:", dataset.shape)
            N_sequences = dataset.shape[0]
            L = dataset.shape[1]
            N_tracks = dataset.shape[-1]
            print(f"For score '{score}': {N_sequences} sequences, length = {L}, {N_tracks} tracks.")


if __name__ == "__main__":
    main()
