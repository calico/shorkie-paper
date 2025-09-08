#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch

# ##########################################
# Helper function to plot ACGT letters      #
# ##########################################
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

    transform = (
        mpl.transforms.Affine2D()
        .scale(globscale, globscale * yscale)
        .translate(x, y)
        + ax.transData
    )
    patch = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=transform)
    ax.add_artist(patch)
    return patch

# #############################################
# Plot sequence logo with matched y-axis      #
# #############################################
def visualize_input_ism(att_grad_wt,
                        plot_start=0,
                        plot_end=None,
                        save_figs=False,
                        fig_name='',
                        figsize=(12, 3),
                        margin_frac=0.10):
    """
    att_grad_wt: (L × 4) importance matrix for a 110-bp window
    Highlights the center of the middle 80-bp (i.e. the ISM region flanked by 17bp and 13bp).
    """
    L_total = att_grad_wt.shape[0]
    if plot_end is None:
        plot_end = L_total
    scores = att_grad_wt[plot_start:plot_end, :]

    # compute symmetric y‐bounds with margin
    ymax = np.max(scores)
    ymin = np.min(scores)
    y_abs = max(abs(ymin), abs(ymax))
    margin = margin_frac * y_abs
    y_min, y_max = -y_abs - margin, y_abs + margin

    # compute highlight index: center of the 80bp ISM region
    left_pad = 17
    right_pad = 13
    core_len = L_total - left_pad - right_pad  # should be 80
    center_idx = left_pad + (core_len // 2)
    highlight_idx = center_idx - plot_start

    plot_seq_scores(
        scores,
        y_min=y_min,
        y_max=y_max,
        figsize=figsize,
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + '_full',
        highlight_idx=highlight_idx
    )

# #############################################
# Function to plot sequence logo              #
# #############################################
def plot_seq_scores(importance_scores,
                    figsize=(20, 2),
                    plot_y_ticks=True,
                    y_min=None,
                    y_max=None,
                    save_figs=False,
                    fig_name="default",
                    highlight_idx=None):
    """
    importance_scores: (plot_len × 4)
    highlight_idx: if not None, draw a light-blue span at [idx, idx+1]
    """
    imp = importance_scores.T  # now (4 × plot_len)
    L = imp.shape[1]

    fig, ax = plt.subplots(figsize=figsize)

    # highlight background at the specified index
    if highlight_idx is not None and 0 <= highlight_idx < L:
        ax.axvspan(highlight_idx,
                   highlight_idx + 1,
                   facecolor='lightblue',
                   alpha=0.3,
                   zorder=0)

    # at each position, draw letters in order of descending |score|
    for j in range(L):
        order = np.argsort(-np.abs(imp[:, j]))
        pos_cum = 0.0
        neg_cum = 0.0

        for idx in order:
            nt = ['A', 'C', 'G', 'T'][idx]
            val = imp[idx, j]
            if val >= 0:
                dna_letter_at(nt, x=j + 0.5, y=pos_cum, yscale=val, ax=ax)
                pos_cum += val
            else:
                dna_letter_at(nt, x=j + 0.5, y=neg_cum, yscale=val, ax=ax)
                neg_cum += val

    # extend x‐limits to avoid cutting off edge letters
    ax.set_xlim(-0.5, L + 0.5)
    if not plot_y_ticks:
        ax.set_yticks([])

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        low = imp.min() - 0.1 * abs(imp).max()
        high = imp.max() + 0.1 * abs(imp).max()
        ax.set_ylim(low, high)

    ax.axhline(0.0, color='black', linewidth=1)
    plt.tight_layout()

    if save_figs:
        plt.savefig(f"{fig_name}.png", dpi=300, transparent=True)
    plt.show()

# #############################################
# Build a 110×4 Δ matrix + original bases     #
# #############################################
def build_importance_matrix(df_seq, seq_len=110):
    """
    df_seq: DataFrame with columns ['pos','orig_base','mut_base','delta']
    returns: mat (seq_len×4) and orig_bases list
    """
    mat = np.zeros((seq_len, 4), dtype=float)
    nt_to_idx = {'A':0, 'C':1, 'G':2, 'T':3}
    orig_bases = ['N'] * seq_len

    for _, r in df_seq.iterrows():
        p = int(r['pos'])
        orig_bases[p] = r['orig_base']
        idx = nt_to_idx[r['mut_base']]
        mat[p, idx] = float(r['delta'])

    # enforce 0 at reference base
    for p, b in enumerate(orig_bases):
        if b in nt_to_idx:
            mat[p, nt_to_idx[b]] = 0.0

    return mat, orig_bases

# #############################################
# Main: iterate, normalize, plot ref‐avg logo #
# #############################################
def main():
    parser = argparse.ArgumentParser(
        description="Plot normalized ISM logos plus ref-base averages"
    )
    parser.add_argument("--input",
                        required=True,
                        help="Path to results/ism_results.tsv")
    parser.add_argument("--output-dir",
                        default="results/",
                        help="Where to save PNGs")
    parser.add_argument("--seq-len",
                        type=int,
                        default=110,
                        help="Sequence length (default: 110)")
    parser.add_argument("--save",
                        action="store_true",
                        help="If set, save each logo to --output-dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input, sep="\t")

    nt_to_idx = {'A':0, 'C':1, 'G':2, 'T':3}
    meta_cols = ['sid_index','Chr','ChrPos','tss_dist']

    for (sid, chrn, pos0, dist), subdf in df.groupby(meta_cols, as_index=False):
        seq_id = f"{chrn}_{pos0}_{dist}"
        print(f"Processing {seq_id}")

        # if (chrn == 'chrIV' and pos0 == 507718) or \
        #     (chrn == 'chrIV' and pos0 == 512392) or \
        #     (chrn == 'chrVII' and pos0 == 942366) or \
        #     (chrn == 'chrXI' and pos0 == 288774) or \
        #     (chrn == 'chrXIV' and pos0 == 546133) or \
        #     (chrn == 'chrII' and pos0 == 606590) or \
        #     (chrn == 'chrVII' and pos0 == 584683) or \
        #     (chrn == 'chrXI' and pos0 == 604356) or \
        #     (chrn == 'chrII' and pos0 == 606590):

        if (chrn == 'chrVII' and pos0 == 584683):
            # Build delta‐matrix and reference bases
            mat, orig_bases = build_importance_matrix(subdf, seq_len=args.seq_len)
            mat = -mat  

            # 1) mean‐normalize the matrix (global mean)
            mat_norm = mat - np.mean(mat)

            # 2) build ref‐average matrix
            ref_mat = np.zeros_like(mat_norm)
            for p, b in enumerate(orig_bases):
                if b in nt_to_idx:
                    ref_idx = nt_to_idx[b]
                    vals = np.delete(mat_norm[p, :], ref_idx)
                    ref_mat[p, ref_idx] = np.mean(vals)

            # Plot reference‐average logo with highlight in the middle of the 80bp core
            print(f"Plotting {seq_id} ref-avg")
            fig_name_ref = os.path.join(args.output_dir, seq_id + "_refavg")
            visualize_input_ism(
                att_grad_wt=ref_mat,
                plot_start=0,
                plot_end=args.seq_len,
                save_figs=args.save,
                fig_name=fig_name_ref,
                figsize=(20, 2),
                margin_frac=0.10
            )

if __name__ == "__main__":
    main()
