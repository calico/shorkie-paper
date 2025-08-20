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
def visualize_input_ism(att_grad_wt,
                        plot_start: int = 0,
                        plot_end: int = None,
                        save_figs: bool = False,
                        fig_name: str = '',
                        figsize: tuple = (12, 3)):
    """
    Pad an importance‐score matrix of shape (L, 4) from 80 → 110 bp
    by adding 17 zero‐rows at the front and 14 at the back, then
    plot the logo over [plot_start:plot_end], highlighting the
    center of the original (unpadded) att_grad_wt.
    """
    # ─── pad to 110 bp: 17 zeros front, 14 zeros back ───
    pad_front = np.zeros((18, att_grad_wt.shape[1]), dtype=att_grad_wt.dtype)
    pad_back  = np.zeros((14, att_grad_wt.shape[1]), dtype=att_grad_wt.dtype)
    padded   = np.vstack([pad_front, att_grad_wt, pad_back])

    # determine end of plotting region
    if plot_end is None:
        plot_end = padded.shape[0]

    # slice out the region to plot
    scores_wt = padded[plot_start:plot_end, :]

    # compute highlight index: center of att_grad_wt within padded coords, 
    # then relative to sliced window
    orig_center = pad_front.shape[0] + (att_grad_wt.shape[0] // 2) - 1
    highlight_idx = orig_center - plot_start

    # compute y‐axis bounds
    y_min = np.min(scores_wt)
    y_max = np.max(scores_wt)
    y_max_abs = max(abs(y_min), abs(y_max))
    y_min -= 0.05 * y_max_abs
    y_max += 0.05 * y_max_abs

    # delegate to the shared plotting routine, passing highlight_idx
    plot_seq_scores(
        scores_wt,
        figsize=figsize,
        plot_y_ticks=False,
        y_min=y_min,
        y_max=y_max,
        save_figs=save_figs,
        fig_name=fig_name + '_wt',
        highlight_idx=highlight_idx
    )


#############################################
# Function to plot sequence logo            #
#############################################
def plot_seq_scores(importance_scores,
                    figsize=(16, 2),
                    plot_y_ticks=True,
                    y_min=None,
                    y_max=None,
                    save_figs=False,
                    fig_name="default",
                    highlight_idx=None):
    """
    Draw a sequence‐logo‐style plot of shape (L,4), with an optional
    light-blue background at the specified x‐position.
    """
    importance_scores = importance_scores.T  # shape now (4, L)
    L = importance_scores.shape[1]

    fig, ax = plt.subplots(figsize=figsize)

    # ─── highlight background if requested ────────────────────────────
    if highlight_idx is not None and 0 <= highlight_idx < L:
        ax.axvspan(highlight_idx,
                   highlight_idx + 1,
                   facecolor='lightblue',
                   alpha=0.3,
                   zorder=0)
    # ──────────────────────────────────────────────────────────────────

    # Plot all 4 letters per position (sorted by magnitude)
    for j in range(L):
        sort_index = np.argsort(-np.abs(importance_scores[:, j]))
        pos_height = 0.0
        neg_height = 0.0

        for ii in range(4):
            i = sort_index[ii]
            nt = ['A', 'C', 'G', 'T'][i]
            nt_val = importance_scores[i, j]

            if nt_val >= 0:
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=pos_height,
                    yscale=nt_val,
                    ax=ax
                )
                pos_height += nt_val
            else:
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=neg_height,
                    yscale=nt_val,
                    ax=ax
                )
                neg_height += nt_val

    # adjust axes
    ax.set_xlim(0, L)
    if not plot_y_ticks:
        ax.set_yticks([])

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        vmin = np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores))
        vmax = np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
        ax.set_ylim(vmin, vmax)

    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
    plt.show()


###########################################################################
# Compute average for one time point (averaging over selected tracks)     #
###########################################################################
def compute_time_average_for_one_time(dataset, seq_idx, track_indices):
    track_indices = np.array(track_indices, dtype=int)
    data_slice = dataset[seq_idx, :, :, track_indices]  # shape: (L, 4, #tracks)
    return data_slice.mean(axis=-1)                     # shape: (L, 4)


###########################################################################
# Main routine to load & plot averaged DNA logos for T0 tracks           #
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

    output_dir = f"gene_exp_motif_eqtl/"
    os.makedirs(f"{output_dir}/viz_logo/", exist_ok=True)

    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'
    fasta_open = pysam.Fastafile(fasta_file)

    target_f = f"{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt"
    target_df = pd.read_csv(target_f, sep="\t")
    target_df = target_df[target_df['identifier'].str.contains('_T0_')]
    selected_track_indices = (target_df['index'].astype(int) - 1148).tolist()

    # Example loop—adjust SLURM array bounds as needed
    for task_id in range(951):
        dataset_dir = f"{output_dir}/f0c0/{task_id}/"
        with h5py.File(f"{dataset_dir}/scores.h5", 'r') as h5_file:
            dataset = h5_file['logSED']
            N_sequences, L, _, N_tracks = dataset.shape

            for seq_idx in range(N_sequences):
                # load sequence coords
                chr_ = h5_file['chr'][seq_idx].decode()
                start = int(h5_file['start'][seq_idx])
                end   = int(h5_file['end'][seq_idx])

                # if (chr_16k == "chrIV" and start_16k == 512352 and end_16k == 512432) or \
                #     (chr_16k == "chrVII" and start_16k == 656536 and end_16k == 656616) or \
                #         (chr_16k == "chrIV" and start_16k == 657555 and end_16k == 657635) or \
                #             (chr_16k == "chrI" and start_16k == 189229 and end_16k == 189309):
                # if (chr_16k == "chrIV" and start_16k == 657555 and end_16k == 657635) :
                if (chr_ == "chrIV" and start == 507678 and end == 507758) :

                    seq1hot = make_seq_1hot(fasta_open, chr_, start, end, end - start).astype('float32')
                    pwm_avg  = compute_time_average_for_one_time(dataset, seq_idx, selected_track_indices)
                    mean_pwm = pwm_avg.mean(axis=-1, keepdims=True)
                    pwm_norm = pwm_avg - np.tile(mean_pwm, (1, 4))

                    # viz_pwm = pwm_norm[options.plot_start:options.plot_end, :]

                    # fig_name = f"{output_dir}/viz_logo/logo_lm_ref_pred_{chr_}_{start}_{end}"
                    # visualize_input_ism(
                    #     viz_pwm,
                    #     plot_start=options.plot_start,
                    #     plot_end=options.plot_end,
                    #     save_figs=True,
                    #     fig_name=fig_name,
                    #     figsize=(16.5, 3)
                    # )

                    # mask by reference
                    masked = pwm_norm * seq1hot  # (L_raw,4)

                    # select region
                    start_pt = options.plot_start
                    end_pt   = options.plot_end or masked.shape[0]
                    to_plot = masked[start_pt:end_pt, :]

                    # plot & save
                    fig_name = f"{output_dir}/viz_logo/logo_lm_ref_pred_{chr_}_{start}_{end}"
                    visualize_input_ism(
                        to_plot,
                        plot_start=0,
                        plot_end=None,
                        save_figs=True,
                        fig_name=fig_name,
                        figsize=(16, 3)
                    )


    # (Optionally stack and save NPZs here...)

if __name__ == "__main__":
    main()
