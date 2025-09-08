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
    Returns a dictionary mapping time points (e.g. 'T0') to a list of row indices.
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
            match = re.search(r'_(T\d+)_', track_label)
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
    parser.add_option('--target_tf', dest='target_tf', default='MSN2_', type='str',
                      help='Target transcription factor identifier substring [Default: %default]')
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

    # Filter targets by the target TF identifier substring provided in the option
    targets_df = targets_df[targets_df['identifier'].str.contains(options.target_tf)]
    print("targets_df: ", targets_df)
    print("Filtered targets_df shape (by TF substring '{}'):".format(options.target_tf), targets_df.shape)

    # Extract time point (e.g., 'T0', 'T1', ...) from the identifier using regex.
    targets_df['time'] = targets_df['identifier'].str.extract(r'_(T\d+)_', expand=False)
    # Sort by time so that the order is consistent.
    targets_df = targets_df.sort_values(by='time').reset_index(drop=True)

    # Prepare track offset and indices.
    track_offset = 1148
    # Create a new column that contains the adjusted h5 file index.
    targets_df['subset_index'] = targets_df['index'].astype(int) - track_offset
    # Get the list of adjusted indices in sorted order.
    total_subset_indices = targets_df['subset_index'].tolist()
    
    # Build a mapping from time point to the positions (in the sorted DataFrame) and also the actual h5 indices.
    group2positions = targets_df.groupby('time').indices  # positions in the sorted DataFrame
    group2h5_indices = {time: [total_subset_indices[i] for i in pos_list]
                         for time, pos_list in group2positions.items()}
    print("Time point groups and their number of tracks:")
    for time, h5_idx in group2h5_indices.items():
        print("  {}: {} tracks".format(time, len(h5_idx)))

    # Select score files based on the exp_dir option.
    if options.exp_dir == "gene_exp_motif_test_TSS/f0c0/":
        score_files = [
            f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/gene_exp_motif_test_TSS/f0c0/part{idx}/scores.h5"
            for idx in range(6)
        ]
        score_files.append(f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/gene_exp_motif_test_TSS_select/f0c0/part0/scores.h5")
    elif options.exp_dir == "gene_exp_motif_test_RP/f0c0/":
        score_files = [
            f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/gene_exp_motif_test_RP/f0c0/part{idx}/scores.h5"
            for idx in range(6)
        ]
    elif options.exp_dir == "gene_exp_motif_test_Proteasome/f0c0/":
        score_files = [
            f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/gene_exp_motif_test_Proteasome/f0c0/part{idx}/scores.h5"
            for idx in range(4)
        ]
    else:
        # list all directories (full directory path) in the specified exp_dir
        dirs = os.listdir(f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/{options.exp_dir}")
        score_files = [
            f"{root_dir}/experiments/motif_LM_fine_tuned_RP_TSS/" + options.exp_dir + directory + "/scores.h5"
            for directory in dirs
        ]
    scores = ['logSED']  # You can add more if needed

    print("score_files: ", score_files)

    # Prepare dictionaries to store concatenated results for each score and each time point.
    # Structure: collected_scores[score][time] is a list of arrays (one per file).
    # Similarly, collected_ref_scores[score][time] is a list of reference arrays.
    collected_scores = {score: {time: [] for time in group2h5_indices} for score in scores}
    collected_ref_scores = {score: {time: [] for time in group2h5_indices} for score in scores}

    score = scores[0]
    # Loop over each score file.
    for score_file in score_files:
        with h5py.File(score_file, 'r') as h5_file:
            # Load the dataset for the current score.
            dataset = h5_file[score][:]  # shape: (N_sequences, L, 4, N_tracks)
            print(f"Loaded dataset '{score}' with shape {dataset.shape} from {score_file}")

            # Process each time point group separately.
            for time, h5_idx_list in group2h5_indices.items():
                # Subset the last dimension using the h5 indices for this time point.
                subset_dataset_group = dataset[..., h5_idx_list]  # shape: (N_sequences, L, 4, num_tracks_group)
                # Mean across nucleotides (axis=2) to compute the average for each track.
                mean_pwm = np.mean(subset_dataset_group, axis=2)  # shape: (N_sequences, L, num_tracks_group)
                # Expand so we can tile back.
                mean_pwm_expanded = np.expand_dims(mean_pwm, axis=-2)  # shape: (N_sequences, L, 1, num_tracks_group)
                # Broadcast to shape (N_sequences, L, 4, num_tracks_group)
                mean_pwm_tiled = np.tile(mean_pwm_expanded, (1, 1, subset_dataset_group.shape[2], 1))
                # Compute normalized scores.
                pwm_data_norm_group = subset_dataset_group - mean_pwm_tiled  # shape: (N_sequences, L, 4, num_tracks_group)
                collected_scores[score][time].append(pwm_data_norm_group)

            # Extract reference sequences for each sequence in the current file.
            N_sequences = dataset.shape[0]
            ref_list_file = []
            for seq_idx in range(N_sequences):
                chr_500 = str(h5_file['chr'][seq_idx].decode("utf-8"))
                start_500 = int(h5_file['start'][seq_idx])
                end_500   = int(h5_file['end'][seq_idx])
                seq_len = end_500 - start_500
                sequence_one_hot = make_seq_1hot(
                    fasta_open,
                    chr_500,
                    start_500,
                    end_500,
                    seq_len
                ).astype("float32")
                ref_list_file.append(sequence_one_hot[None, ...])
            # Concatenate references from this file (shape: (N_sequences, L, 4)).
            ref_arr_file = np.concatenate(ref_list_file, axis=0)
            # Append the same reference array to each time group.
            for time in group2h5_indices:
                collected_ref_scores[score][time].append(ref_arr_file)

    ##############################################
    # Calculate difference relative to T0 and save NPZ files per time point
    ##############################################
    # Compute baseline (T0) averages.
    if 'T0' not in group2h5_indices:
        print("No T0 group found! Cannot compute difference relative to T0.")
        return
    else:
        pred_T0_concat = np.concatenate(collected_scores[score]['T0'], axis=0)  # shape: (N_total_T0, L, 4, n_tracks_T0)
        baseline_pred = np.mean(pred_T0_concat, axis=-1)  # shape: (N_total_T0, L, 4)
        ref_T0_concat = np.concatenate(collected_ref_scores[score]['T0'], axis=0)  # shape: (N_total_T0, L, 4)
        baseline_ref = ref_T0_concat

    # Create the top-level output directory.
    out_dir_score = os.path.join("results", options.exp_dir, options.target_tf)
    os.makedirs(out_dir_score, exist_ok=True)

    # Process each time group.
    for time in group2h5_indices:
        # Concatenate and average across tracks for predictions.
        pred_concat = np.concatenate(collected_scores[score][time], axis=0)  # shape: (N_total, L, 4, n_tracks)
        pred_mean = np.mean(pred_concat, axis=-1)  # shape: (N_total, L, 4)
        # Concatenate reference arrays.
        ref_concat = np.concatenate(collected_ref_scores[score][time], axis=0)  # shape: (N_total, L, 4)

        # Compute difference relative to T0.
        if time == 'T0':
            diff_pred = np.zeros_like(pred_mean)
            diff_ref = ref_concat
        else:
            diff_pred = pred_mean - baseline_pred
            diff_ref = ref_concat

        # Transpose to (N_total, 4, L) for saving.
        diff_pred_trans = np.transpose(diff_pred, (0, 2, 1))
        diff_ref_trans = np.transpose(diff_ref, (0, 2, 1))

        # Create output subdirectory for this time point.
        out_dir_time = os.path.join(out_dir_score, time)
        os.makedirs(out_dir_time, exist_ok=True)

        # Save the difference arrays.
        pred_out_npz = os.path.join(out_dir_time, "pred_diff.npz")
        np.savez(pred_out_npz, arr_0=diff_pred_trans)
        print(f"Saved pred.npz (difference) with shape {diff_pred_trans.shape} in {out_dir_time}")

        ref_out_npz = os.path.join(out_dir_time, "ref_diff.npz")
        np.savez(ref_out_npz, arr_0=diff_ref_trans)
        print(f"Saved ref.npz (difference) with shape {diff_ref_trans.shape} in {out_dir_time}")

if __name__ == "__main__":
    main()
