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
    
    # Select only rows where the identifier contains '_T0_'
    target_df = target_df[target_df['identifier'].str.contains('_T0_')]
    print("Filtered target_df (T0):")
    print(target_df.head())

    # Get the selected track indices for T0 (as a list)
    selected_track_indices = target_df['index'].astype(int).tolist()
    
    # The h5py dataset for 'logSED' has 3053 tracks corresponding to global track indices
    # starting at 1148 and going to 4200. Thus, to index into the dataset (which is 0-indexed),
    # subtract the offset (1148) from each target index.
    track_offset = 1148
    selected_track_indices = [idx - track_offset for idx in selected_track_indices]
    with h5py.File(f"{options.exp_dir}/scores.h5", 'r') as h5_file:
        scores = ['logSED']  # or other score types if desired
        for score in scores:
            dataset = h5_file[score]  # shape: (N_sequences, L, 4, N_tracks)
            N_sequences = dataset.shape[0]
            L = dataset.shape[1]
            N_tracks = dataset.shape[-1]
            print(f"For score '{score}': {N_sequences} sequences, length = {L}, {N_tracks} tracks.")

            # (Optional) Verify that the adjusted indices fall within range [0, N_tracks)
            if max(selected_track_indices) >= N_tracks or min(selected_track_indices) < 0:
                raise ValueError("One or more adjusted track indices are out-of-range.")
            
            # Create an output directory for the averaged T0 logo
            track_dir = f"{options.exp_dir}/dna_logo_example/logos/{score}/T0_average"
            os.makedirs(track_dir, exist_ok=True)

            # Prepare lists to later stack logos for all sequences
            viz_pwm_all_seqs_true = []
            viz_pwm_all_seqs_pred = []

            for seq_idx in range(N_sequences):
                plot_start = 0
                plot_end = L

                # Read sequence coordinates from the h5 file
                chr_16k = str(h5_file['chr'][seq_idx].decode("utf-8"))
                start_16k = int(h5_file['start'][seq_idx])
                end_16k = int(h5_file['end'][seq_idx])
                print(f"[score={score}, T0 average] seq_idx={seq_idx} --> {chr_16k}:{start_16k}-{end_16k}")

                # Get the one-hot representation of the reference sequence
                sequence_one_hot = make_seq_1hot(
                    fasta_open,
                    chr_16k,
                    start_16k,
                    end_16k,
                    end_16k - start_16k
                ).astype("float32")
                print("\t>> sequence_one_hot shape:", sequence_one_hot.shape)

                # Get PWM data averaged over the selected T0 tracks (using the adjusted indices)
                pwm_data = compute_time_average_for_one_time(
                    dataset,
                    seq_idx=seq_idx,
                    track_indices=selected_track_indices
                )

                # Mean-normalize the PWM (weighting across nucleotides)
                mean_pwm = np.mean(pwm_data, axis=-1)               # shape: (L,)
                mean_pwm_expanded = mean_pwm[..., None]               # shape: (L,1)
                mean_pwm_tiled = np.tile(mean_pwm_expanded, (1, 4))     # shape: (L,4)
                pwm_data_norm = pwm_data - mean_pwm_tiled             # shape: (L,4)

                # Compute two versions:
                # 1) Only the reference motif (one-hot multiplied)
                viz_pwm = pwm_data_norm * sequence_one_hot
                # 2) The full normalized PWM (if needed)
                viz_pwm_4 = pwm_data_norm

                # Limit the region to plot
                viz_pwm = viz_pwm[plot_start:plot_end, :]
                viz_pwm_4 = viz_pwm_4[plot_start:plot_end, :]
                sequence_one_hot_clip = sequence_one_hot[plot_start:plot_end, :]

                # Collect for later stacking (if desired)
                viz_pwm_all_seqs_true.append(sequence_one_hot_clip)
                viz_pwm_all_seqs_pred.append(viz_pwm)

                # Define output filename for this sequence
                fig_out_ref_pred = f"{track_dir}/logo_lm_ref_pred_{seq_idx}_{chr_16k}_{start_16k}_{end_16k}"
                # Plot and save the DNA logo (averaged T0)
                visualize_input_ism(
                    viz_pwm,
                    plot_start=plot_start,
                    plot_end=plot_end,
                    save_figs=True,
                    fig_name=fig_out_ref_pred,
                    figsize=(16.5, 3)
                )
                print(f"Saved DNA logo for T0 average, sequence {seq_idx}")

            # After processing all sequences, stack and save NPZ files
            big_tensor_true = np.stack(viz_pwm_all_seqs_true, axis=0)  # shape: (N_sequences, plot_length, 4)
            big_tensor_pred = np.stack(viz_pwm_all_seqs_pred, axis=0)
            # Transpose to shape (N_sequences, 4, plot_length)
            big_tensor_true = np.transpose(big_tensor_true, (0, 2, 1))
            big_tensor_pred = np.transpose(big_tensor_pred, (0, 2, 1))

            out_npz_true = f"{track_dir}/true.npz"
            np.savez(out_npz_true, arr_0=big_tensor_true)
            out_npz_pred = f"{track_dir}/pred.npz"
            np.savez(out_npz_pred, arr_0=big_tensor_pred)
            print(f"Saved concatenated tensors for T0 average at:")
            print(f"  {out_npz_true}")
            print(f"  {out_npz_pred}")

if __name__ == "__main__":
    main()
