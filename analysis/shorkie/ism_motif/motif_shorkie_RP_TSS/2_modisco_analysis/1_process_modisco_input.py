#!/usr/bin/env python3
from optparse import OptionParser

import numpy as np
import pandas as pd
import os
import h5py
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

# Import helper functions from your local module. 
# Make sure make_seq_1hot(...) is defined in yeast_helpers or otherwise accessible
from yeast_helpers import make_seq_1hot

########################################
# Helper functions for plotting logos  #
########################################

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
        "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
        "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)
    }
    COLOR_SCHEME = {
        'G': 'orange',
        'A': 'green',
        'C': 'blue',
        'T': 'red',
        'UP': 'green',
        'DN': 'red',
        '(' : 'black',
        '.' : 'black',
        ')' : 'black'
    }

    text = LETTERS[letter]
    chosen_color = COLOR_SCHEME[letter] if color is None else color

    t = (mpl.transforms.Affine2D().scale(globscale, yscale*globscale)
         + mpl.transforms.Affine2D().translate(x, y)
         + ax.transData)
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)
    return p


def plot_seq_scores(importance_scores, figsize=(16, 2), plot_y_ticks=True,
                    y_min=None, y_max=None, save_figs=False, fig_name="default"):

    importance_scores = importance_scores.T  # shape (4, L)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Using a multi-letter approach (plot all 4 letters per position)
    for j in range(importance_scores.shape[1]):
        # Sort indices by descending absolute value
        sort_index = np.argsort(-np.abs(importance_scores[:, j]))  # largest magnitude first
        pos_height = 0.0
        neg_height = 0.0

        for ii in range(4):
            i = sort_index[ii]
            nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}[i]
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
    if plot_y_ticks:
        plt.yticks(fontsize=12)
    else:
        plt.yticks([], [])

    # Auto or custom y-lims
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    else:
        abs_max = np.max(np.abs(importance_scores))
        plt.ylim(-abs_max - 0.1*abs_max, abs_max + 0.1*abs_max)

    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")

    plt.show()


def visualize_input_ism(att_grad_wt, plot_start=0, plot_end=100,
                        save_figs=False, fig_name='', figsize=(12, 3)):
    scores_wt = att_grad_wt[plot_start:plot_end, :]

    # Compute y-min/y-max
    y_min = np.min(scores_wt)
    y_max = np.max(scores_wt)
    y_max_abs = max(np.abs(y_min), np.abs(y_max))
    y_min -= 0.05 * y_max_abs
    y_max += 0.05 * y_max_abs

    plot_seq_scores(
        scores_wt,
        y_min=y_min,
        y_max=y_max,
        figsize=figsize,
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + '_wt',
    )

##################################
# Example parsing helper         #
##################################

def parse_time_points(target_file):
    """
    Example function for reading a target file.
    """
    time2inds = defaultdict(list)
    with open(target_file, 'r') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            cols = line.strip().split('\t')
            if cols[0] == 'index':
                continue
            track_label = cols[1]  # e.g. "ARG80_T0_S757"
            match = re.search(r'(T\d+)', track_label)
            if match:
                time_str = match.group(1)
                time2inds[time_str].append(idx - 1)
    return time2inds


def compute_time_average_for_one_time(dataset, seq_idx, track_indices):
    """
    Example function for averaging across multiple tracks (not used in final snippet).
    """
    data_slice = dataset[seq_idx, :, :, track_indices]   # shape (L,4,#tracks_for_that_time)
    data_avg = data_slice.mean(axis=-1)  # shape => (L, 4)
    return data_avg

##################################
# Main routine                   #
##################################

def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--exp_dir', dest='exp_dir', default='', type='str',
                      help='Experiment directory to store results [Default: %default]')
    (options, args) = parser.parse_args()

    # Root / data settings
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'
    fasta_open = pysam.Fastafile(fasta_file)

    exp_type = "exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"
    targets_file = f'{root_dir}/seq_experiment/{exp_type}/16bp/cleaned_sheet_RNA-Seq.txt'

    # Load target file into DataFrame
    targets_df = pd.read_csv(targets_file, sep='\t')
    print("Full targets_df shape:", targets_df.shape)

    # For demonstration, let's keep only T0
    targets_df = targets_df[targets_df['identifier'].str.contains('T0')]
    print("Filtered T0 targets_df shape:", targets_df.shape)

    # Prepare track offset and indices
    track_offset = 1148
    selected_track_indices = targets_df['index'].astype(int).tolist()
    subset_indices = [idx - track_offset for idx in selected_track_indices]
    print("Raw track indices:", selected_track_indices[:10], "... (total)", len(selected_track_indices))
    print("Adjusted track indices:", subset_indices[:10], "... (total)", len(subset_indices))

    # Score files to load
    # score_files = [
    #     f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/gene_exp_motif_test_RP/f0c0/part{idx}/scores.h5"
    #     for idx in range(7)
    # ]
    score_files = [
        f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/gene_exp_motif_test_TSS/f0c0/part{idx}/scores.h5"
        for idx in range(6)
    ]
    score_files.append(f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/gene_exp_motif_test_TSS_select/f0c0/part0/scores.h5")

    scores = ['logSED']  # you can add more if needed

    # Prepare dictionaries to store concatenated results
    pred_scores = {}
    ref_scores = {}

    # Ensure exp_dir exists
    if options.exp_dir == '':
        options.exp_dir = f"results/"
    os.makedirs(options.exp_dir, exist_ok=True)

    # Loop over each type of score (e.g., logSED)
    for score in scores:
        collected_scores = []
        collected_ref_scores = []

        # Iterate over each file
        for score_file in score_files:
            with h5py.File(score_file, 'r') as h5_file:
                # The main dataset is e.g. h5_file['logSED']
                dataset = h5_file[score][:]  # shape: (N_sequences, L, 4, N_tracks)
                print(f"Loaded dataset '{score}' with shape {dataset.shape} from {score_file}")

                # Subset the last dimension by your track indices
                subset_dataset = dataset[..., subset_indices]
                # shape => (N_sequences, L, 4, len(subset_indices))

                # Mean across channels (axis=2 is the base/nucleotide axis)
                # NOTE: If you want to average across the 4 nucleotides, set axis=2
                # If you want to average across tracks, set axis=-1. 
                # The snippet suggests we do: np.mean(subset_dataset, axis=2). 
                # Adjust to your actual usage:
                mean_pwm = np.mean(subset_dataset, axis=2)  # => shape (N_sequences, L, #subset_tracks)

                # Expand so we can tile back
                #  shape => (N_sequences, L, 1, #subset_tracks)
                mean_pwm_expanded = np.expand_dims(mean_pwm, axis=-2)

                # Broadcast to shape (N_sequences, L, 4, #subset_tracks)
                mean_pwm_tiled = np.tile(mean_pwm_expanded, (1, 1, subset_dataset.shape[2], 1))

                # Finally, compute normalization
                pwm_data_norm = subset_dataset - mean_pwm_tiled  # shape (N_sequences, L, 4, #subset_tracks)
                collected_scores.append(pwm_data_norm)

                # Grab reference sequences for each entry
                N_sequences = subset_dataset.shape[0]
                for seq_idx in range(N_sequences):
                    chr_500 = str(h5_file['chr'][seq_idx].decode("utf-8"))
                    start_500 = int(h5_file['start'][seq_idx])
                    end_500   = int(h5_file['end'][seq_idx])

                    # reference sequence in 1-hot
                    seq_len = end_500 - start_500
                    sequence_one_hot = make_seq_1hot(
                        fasta_open,
                        chr_500,
                        start_500,
                        end_500,
                        seq_len
                    ).astype("float32")
                    
                    # Append with a new batch dimension: (1, L, 4)
                    collected_ref_scores.append(sequence_one_hot[None, ...])

        # Once all files are read, concatenate along axis=0
        # collected_scores: each item => shape (N_sequences, L, 4, #subset_tracks)
        # collected_ref_scores: each item => shape (1, L, 4)
        pred_scores[score] = np.concatenate(collected_scores, axis=0)
        ref_scores[score]  = np.concatenate(collected_ref_scores, axis=0)

        # pred_scores[score] => shape (N_total, L, 4, #subset_tracks)
        # ref_scores[score] => shape (N_total, L, 4)
        print(f"{score} pred shape: {pred_scores[score].shape}")
        print(f"{score} ref shape: {ref_scores[score].shape}")

    ##############################################
    # Save the final NPZ files for each score
    ##############################################
    for score in scores:
        out_dir = f"{options.exp_dir}/{score}"
        os.makedirs(out_dir, exist_ok=True)

        ############################################################
        # Process and save ref_scores
        ############################################################
        # shape => (num_sequences, L, 4)
        ref_arr = ref_scores[score]
        # Transpose to (num_sequences, 4, L)
        ref_arr_trans = np.transpose(ref_arr, (0, 2, 1))

        # Number of tracks in pred data
        num_tracks = pred_scores[score].shape[-1]

        # Tile the reference array along the first dimension (num_tracks times)
        # so each track gets the same reference
        ref_tiled = np.tile(ref_arr_trans, (num_tracks, 1, 1))
        ref_out_npz = f"{out_dir}/ref.npz"
        np.savez(ref_out_npz, arr_0=ref_tiled)
        print(f"Saved ref.npz with shape {ref_tiled.shape} in {out_dir}")

        ############################################################
        # Process and save pred_scores
        ############################################################
        # shape => (num_sequences, L, 4, num_tracks)
        pred_arr = pred_scores[score]
        num_tracks = pred_arr.shape[-1]
        print(f"Processing {num_tracks} tracks for score '{score}'...")

        # Collect all track arrays after transposing
        track_list = []
        for track_idx in range(num_tracks):
            # Extract track data => shape (num_sequences, L, 4)
            track_data = pred_arr[..., track_idx]
            # Transpose => shape (num_sequences, 4, L)
            track_trans = np.transpose(track_data, (0, 2, 1))
            track_list.append(track_trans)

        # Concatenate all tracks along the first dimension => shape (num_seq * num_tracks, 4, L)
        pred_concat = np.concatenate(track_list, axis=0)
        print(f"Concatenated pred shape: {pred_concat.shape}")

        pred_out_npz = f"{out_dir}/pred.npz"
        np.savez(pred_out_npz, arr_0=pred_concat)
        print(f"Saved pred.npz with shape {pred_concat.shape} in {out_dir}")


if __name__ == "__main__":
    main()
