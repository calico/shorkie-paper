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
from yeast_helpers import *
import pysam
from collections import defaultdict
import re

#Helper function to plot ACGT letters at a given position
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
                "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
                "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
                "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
                "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
                "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
                "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
                "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
                ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
    COLOR_SCHEME = {'G': 'orange',#'orange', 
                    'A': 'green',#'red', 
                    'C': 'blue',#'blue', 
                    'T': 'red',#'darkgreen',
                    'UP': 'green', 
                    'DN': 'red',
                    '(': 'black',
                    '.': 'black', 
                    ')': 'black'}


    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]
    if color is not None :
        chosen_color = color

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


#Plot pair of sequence logos with matched y-axis height
def visualize_input_ism(att_grad_wt, plot_start=0, plot_end=100, save_figs=False, fig_name='', figsize=(12, 3)):
    scores_wt = att_grad_wt[plot_start:plot_end, :]
    print(">> scores_wt: ", scores_wt)

    #Get logo bounds
    y_min = np.min(scores_wt)
    y_max = np.max(scores_wt)
    y_max_abs = max(np.abs(y_min), np.abs(y_max))

    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs
    
    print("y_min = " + str(round(y_min, 8)))
    print("y_max = " + str(round(y_max, 8)))

    #Plot wt logo
    print("--- WT ---")
    plot_seq_scores(
        scores_wt, y_min=y_min, y_max=y_max,
        figsize=figsize,
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + '_wt',
    )


#Function to plot sequence logo
def plot_seq_scores(importance_scores, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, save_figs=False, fig_name="default") :

    importance_scores = importance_scores.T

    print(f"importance_scores: {importance_scores}")
    print(f"importance_scores: {importance_scores.shape}")
    fig = plt.figure(figsize=figsize)
    
    ref_seq = ""
    ax = plt.gca()

    # # 1. Plot single nt per position.
    # #Loop over reference sequence letters
    # for j in range(importance_scores.shape[1]) :
    #     argmax_nt = np.argmax(np.abs(importance_scores[:, j]))
        
    #     if argmax_nt == 0 :
    #         ref_seq += "A"
    #     elif argmax_nt == 1 :
    #         ref_seq += "C"
    #     elif argmax_nt == 2 :
    #         ref_seq += "G"
    #     elif argmax_nt == 3 :
    #         ref_seq += "T"
    
    # #Loop over reference sequence letters and draw
    # for i in range(0, len(ref_seq)) :
    #     mutability_score = np.sum(importance_scores[:, i])
    #     color = None
    #     dna_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)
    
    # -------------------------------------------------------------------------
    # 2) Plot all 4 letters per position (uncommented & fixed)
    # -------------------------------------------------------------------------
    logo_height = 1.0
    height_base = (1.0 - logo_height) / 2.0  # Typically (1 - 1) / 2 = 0

    for j in range(importance_scores.shape[1]):
        # Sort indices by descending absolute value
        sort_index = np.argsort(-np.abs(importance_scores[:, j]))  # largest magnitude first

        # Running sums for stacking
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
            print(f"{i}, {j}, {nt}: {nt_prob}")
            # Decide whether to stack up or down
            if nt_prob >= 0.0:
                # Plot above x-axis
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=pos_height,
                    yscale=nt_prob,  # or nt_prob * some scale factor
                    ax=ax,
                    color=None
                )
                pos_height += nt_prob
            else:
                # Plot below x-axis
                # Option 1: Use negative yscale => letter might be upside down
                # dna_letter_at(nt, j + 0.5, neg_height, nt_prob, ax, color=None)
                #
                # Option 2: Keep letters upright => supply a positive scale and shift y down
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=neg_height,   # shift down by nt_prob
                    yscale=nt_prob,         # use positive for letter height
                    ax=ax,
                    color=None
                )
                neg_height += nt_prob

    # Use the shape of importance_scores instead of len(ref_seq)
    plt.xlim((0, importance_scores.shape[1]))

    if plot_y_ticks:
        plt.yticks(fontsize=12)
    else:
        plt.yticks([], [])

    # Set axis limits
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        plt.ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
        )

    # Draw a horizontal line at y=0
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1)

    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")

    plt.show()


def parse_time_points(target_file):
    """
    Reads a target file (tab-delimited) where
    columns might look like:
        idx    ARG80_T0_S757   /path/to/bw ... 
    Returns a dictionary: 
        time2inds = {
           'T0':  [1148],
           'T5':  [1149],
           'T10': [1150],
           ...
        }
    """
    # target_df = pd.read_csv(target_file, sep='\t')
    # print("target_df: ", target_df)

    time2inds = defaultdict(list)
    with open(target_file, 'r') as f:
        # for line in f:
        for idx, line in enumerate(f):
            print("idx: ", idx)
            if not line.strip():
                continue
            cols = line.strip().split('\t')
            if cols[0] == 'index':
                continue
            print("cols: ", cols)
            # track_idx = int(cols[0])
            track_label = cols[1]  # e.g. "ARG80_T0_S757"
            
            # Extract the time portion: T0, T5, T10, etc.
            # A simple approach with regex:
            match = re.search(r'(T\d+)', track_label)  # captures T0, T5, T10, ...
            if match:
                time_str = match.group(1)  # e.g. 'T0'
                time2inds[time_str].append(idx-1)
            # else: ignore lines that do not match the expected pattern
    return time2inds


def compute_time_average_for_one_time(dataset, seq_idx, track_indices):
    """
    Given a dataset of shape (N_sequences, L, 4, N_tracks),
    and a list of track_indices for a single time point,
    returns the average over those tracks.
    """
    # Subset the dataset on the last dimension
    data_slice = dataset[seq_idx, :, :, track_indices]   # shape (L, 4, #tracks_for_that_time)
    
    # Average across the last dimension
    data_avg = data_slice.mean(axis=-1)  # shape => (L, 4)
    
    return data_avg


##################################
# Main routine to load & plot    #
##################################
def main():
    """
    Example routine: 
      1) load a dataset (e.g., D1),
      2) select a single sequence & track,
      3) plot the DNA logo for positions 0–200.
    """
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--exp_dir', dest='exp_dir', default='', type='str', help='Experiment directory [Default: %default]')
    parser.add_option('--time_label', dest='time_label', default='', type='str', help='Time label [Default: %default]')
    (options,args) = parser.parse_args()

    root_dir='/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'
    #Initialize fasta sequence extractor
    fasta_open = pysam.Fastafile(fasta_file)
    seq_len = 16384
    targets_file = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq/16bp/targets_rnaseq.txt'
    time2inds = parse_time_points(targets_file)
    print("time2inds: ", time2inds) 

    # Just define which time points you want to visualize
    # Or you can do something like: times_to_plot = sorted(time2inds.keys())
    # times_to_plot = ["T0", "T5", "T10", "T15", "T30", "T45", "T60", "T90"]

    scores = ['SUM', 'logSUM']    
    # Prepare data structures for storing all sequences per time label
    # We'll store dict-of-dicts: e.g. { "SUM": {"T0": [], "T5": []... }, "logSUM": {"T0": [], ...}, ... }
    with h5py.File(f"{options.exp_dir}/scores.h5", 'r') as h5_file:
        for score in scores:
            dataset = h5_file[score]  # shape (N_sequences, L, 4, N_tracks)
            print("dataset shape: ", dataset.shape)  # e.g., (N_sequences=137, L=500, 4, N_tracks=3053)

            N_sequences = dataset.shape[0]
            L = dataset.shape[1]
            print("Found dataset shape: ", dataset.shape)
            print(f"Number of sequences (N_sequences) = {N_sequences}, length = {L}")

            # # For each time point, do:
            # for time_label in times_to_plot:
            time_label = options.time_label
                
            # Creating output directory
            os.makedirs(f"{options.exp_dir}/dna_logo_example/logos/{score}/{time_label}/", exist_ok=True)
            
            track_indices = time2inds[time_label]  # e.g. [1148, 1153, ...] 
            # Loop over all sequences
            viz_pwm_all_seqs_pred = []
            viz_pwm_all_seqs_true = []
            for seq_idx in range(N_sequences):
                plot_start = 0
                plot_end   = 500  # or up to L, etc.
                
                chr_16k = str(h5_file['chr'][seq_idx].decode("utf-8"))
                start_16k = int(h5_file['start'][seq_idx])
                end_16k = int(h5_file['end'][seq_idx])
                print(f"\t[score={score}, time={time_label}] seq_idx={seq_idx} --> {chr_16k}:{start_16k}-{end_16k}")

                # Grab reference sequence in 1-hot
                sequence_one_hot = make_seq_1hot(
                    fasta_open, 
                    chr_16k, 
                    start_16k, 
                    end_16k, 
                    end_16k - start_16k
                ).astype("float32")

                print("\t>> sequence_one_hot shape: ", sequence_one_hot.shape)  # Should match (L,4) or (end-start,4)
                
                # Average across the relevant track indices for this time → shape (L,4)
                pwm_data = compute_time_average_for_one_time(
                    dataset,
                    seq_idx=seq_idx,
                    track_indices=track_indices
                )

                # Optionally do your “mean across nucleotides” weighting:
                mean_pwm = np.mean(pwm_data, axis=-1)               # shape: (L,)
                mean_pwm_expanded = mean_pwm[..., None]             # shape: (L,1)
                # print("mean_pwm_expanded: ", mean_pwm_expanded)
                mean_pwm_tiled = np.tile(mean_pwm_expanded, (1, 4)) # shape: (L,4)
                # print("mean_pwm_tiled: ", mean_pwm_tiled)
                pwm_data_norm = pwm_data - mean_pwm_tiled           # shape: (L,4)
                # print("pwm_data_norm: ", pwm_data_norm)
                # Only show the reference motif
                viz_pwm = pwm_data_norm * sequence_one_hot          # shape: (L,4)
                viz_pwm_4 = pwm_data_norm                               # shape: (L,4)

                # If you only want the first 500bp:
                viz_pwm = viz_pwm[plot_start:plot_end, :]           # shape: (500,4) 
                viz_pwm_4 = viz_pwm_4[plot_start:plot_end, :]           # shape: (500,4) 
                # Collect for this (score, time_label)
                viz_pwm_all_seqs_true.append(sequence_one_hot)
                viz_pwm_all_seqs_pred.append(viz_pwm)

                # # Plot the DNA logo for these positions
                # fig_out = f"{options.exp_dir}/dna_logo_example/logos/{score}/{time_label}/logo_lm_ref_pred_{seq_idx}_{chr_16k}_{start_16k}_{end_16k}"
                # visualize_input_ism(
                #     viz_pwm, 
                #     plot_start=plot_start, 
                #     plot_end=plot_end, 
                #     save_figs=True, 
                #     fig_name=fig_out, 
                #     figsize=(100,3)
                # )

                # fig_out = f"{options.exp_dir}/dna_logo_example/logos/{score}/{time_label}/logo_lm_pred_{seq_idx}_{chr_16k}_{start_16k}_{end_16k}"
                # visualize_input_ism(
                #     viz_pwm_4, 
                #     plot_start=plot_start, 
                #     plot_end=plot_end, 
                #     save_figs=True, 
                #     fig_name=fig_out, 
                #     figsize=(100,3)
                # )

                # fig_out = f"{options.exp_dir}/dna_logo_example/logos/{score}/{time_label}/logo_lm_true_{seq_idx}_{chr_16k}_{start_16k}_{end_16k}"
                # visualize_input_ism(
                #     sequence_one_hot, 
                #     plot_start=plot_start, 
                #     plot_end=plot_end, 
                #     save_figs=True, 
                #     fig_name=fig_out, 
                #     figsize=(100,3)
                # )
                # print(f"Saved DNA logo for time {time_label} → {fig_out}.png")


                # Now that we've looped over all seq_idx, `viz_pwm_all_seqs` is
                # a list of shape (N_sequences) each with (plot_end - plot_start, 4).
                # We can stack them along axis=0 to get:
                #    big_tensor shape = (N_sequences, (plot_end - plot_start), 4)
                big_tensor_true = np.stack(viz_pwm_all_seqs_true, axis=0)
                big_tensor_pred = np.stack(viz_pwm_all_seqs_pred, axis=0)

                # Change shape to (137, 4, 500)
                big_tensor_true = np.transpose(big_tensor_true, (0, 2, 1))
                big_tensor_pred = np.transpose(big_tensor_pred, (0, 2, 1))
                print(">> big_tensor_true: ", big_tensor_true.shape)
                print(">> big_tensor_pred: ", big_tensor_pred.shape)

                # Save as .npz for this (score, time_label)
                out_npz_true = f"{options.exp_dir}/dna_logo_example/logos/{score}/{time_label}/true.npz"
                np.savez(out_npz_true, arr_0=big_tensor_true)

                out_npz_pred = f"{options.exp_dir}/dna_logo_example/logos/{score}/{time_label}/pred.npz"
                np.savez(out_npz_pred, arr_0=big_tensor_pred)
                
                print(f"Saved concatenated tensor for [score={score}, time={time_label}] at {out_npz_true}")
                print(f"Saved concatenated tensor for [score={score}, time={time_label}] at {out_npz_pred}")


                # plot_dna_logo(
                #     viz_pwm,
                #     sequence_template=full_seq_template,
                #     figsize=(100,3),
                #     logo_height=1.0,
                #     plot_start=plot_start,
                #     plot_end=plot_end,
                #     plot_sequence_template=False,
                #     save_figs=True,
                #     fig_name=fig_out
                # )

if __name__ == "__main__":
    main()
