#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

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

    # Build transformation: scale the letter and move it to (x, y)
    t = mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)
    return p

##########################################
# Plot DNA logo using conservation score #
##########################################
def plot_dna_logo(pwm, sequence_template, figsize=(12, 3), logo_height=1.0, 
                  plot_start=0, plot_end=164, plot_sequence_template=False, 
                  save_figs=False, fig_name=None, annotate_positions=None, annotate_labels=None):
    # Slice the PWM and sequence template for the region to be plotted
    pwm = np.copy(pwm[plot_start: plot_end, :])
    sequence_template = sequence_template[plot_start: plot_end]

    # Normalize PWM rows to sum to 1
    pwm += 0.0001
    for j in range(pwm.shape[0]):
        pwm[j, :] /= np.sum(pwm[j, :])

    # Compute per-position entropy and conservation (2 - entropy)
    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    height_base = (1.0 - logo_height) / 2.

    # Plot each position
    for j in range(pwm.shape[0]):
        sort_index = np.argsort(pwm[j, :])
        for ii in range(4):
            i = sort_index[ii]
            # Map index to nucleotide
            if i == 0:
                nt = 'A'
            elif i == 1:
                nt = 'C'
            elif i == 2:
                nt = 'G'
            elif i == 3:
                nt = 'T'

            nt_prob = pwm[j, i] * conservation[j]
            color = None
            # If sequence_template is provided and not the placeholder '$', then use it
            if sequence_template[j] != '$':
                color = 'black'
                if plot_sequence_template and nt == sequence_template[j]:
                    nt_prob = 2.0
                else:
                    nt_prob = 0.0

            if ii == 0:
                dna_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, ax, color=color)
            else:
                prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                dna_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax, color=color)

    # Optionally add annotations
    if annotate_positions is not None:
        for pos, label in zip(annotate_positions, annotate_labels):
            x_pos = pos - plot_start + 0.5
            if 0 <= x_pos <= plot_end - plot_start:
                ax.axvline(x=x_pos, color='red', linestyle='--')
                ax.text(x_pos, 2, label, color='red', ha='center', va='bottom', fontsize=12)

    plt.xlim((0, plot_end - plot_start))
    plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)
    for axis in fig.axes:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    plt.tight_layout()
    if save_figs and fig_name is not None:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")
    plt.show()
    plt.clf()

##########################################
# Main routine                           #
##########################################
def main():
    # Load the PWM numpy array from file (assumed shape: (1003, 4))
    pwm_data = np.load("/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/dependencies_DNALM/all_prbs.npy")
    # pwm_data = np.load("/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/dependencies_DNALM/all_prbs_chrX_607855_608355.npy")
    print("pwm_data shape: ", pwm_data.shape)   
    # print("pwm_data: ", pwm_data)   

    # Create a dummy sequence template.
    # By using a string of '$' characters, we let the conservation score drive the letter heights.
    sequence_template = "$" * pwm_data.shape[0]

    # Define the region to plot (from 690 to 800)
    plot_start = 690
    plot_end = 986
    # plot_start = 97
    # plot_end = 207

    # Plot the DNA logo using the conservation score
    plot_dna_logo(pwm=pwm_data, 
                  sequence_template=sequence_template, 
                  figsize=(35, 3), 
                  logo_height=1.0, 
                  plot_start=plot_start, 
                  plot_end=plot_end, 
                  plot_sequence_template=False, 
                  save_figs=True, 
                  fig_name="dna_logo_conservation_full")
                #   fig_name="dna_logo_conservation_chrX_607855_608355")

if __name__ == "__main__":
    main()
