#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
# Check if modisco is available, if not, skip that part or warn
try:
    from modisco.visualization import viz_sequence
    MODISCO_AVAILABLE = True
except ImportError:
    MODISCO_AVAILABLE = False
    print("Warning: modisco not found. Skipping plot_weights.")

# Parameters
OUTPUT_DIR = "viz_smt3_unmasked_output"
INFERENCE_FILE = "inference_smt3_output/preds_smt3_unmasked.npz"
PAD_BP = 512
PLOT_START = 203
PLOT_END = 311
# PLOT_END = 497
LOGO_HEIGHT = 1.0


# SMT3 Gene Info (YDR510W)
GENE_CHROM = "chrIV"
GENE_START = 1469400 # 0-based inclusive (GTF start - 1)
GENE_END = 1469705   # 0-based exclusive (GTF end)
GENE_STRAND = "+"

# Windows (MUST match 3_inference_smt3.py / 3_inference_smt3_unmasked.py)
SMT3_WINDOWS = [
    ("chrIV", 1454592, 1470976),
    ("chrIV", 1458688, 1475072),
    ("chrIV", 1462784, 1479168),
    ("chrIV", 1466880, 1483264),
]

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
        # plt.savefig(fig_name + ".eps") # Skipping EPS as per original
    plt.close()

def norm_pwm(pwm):
    pwm += 0.0001
    p_mean = np.mean(pwm, axis=1, keepdims=True)
    pwm = pwm * np.log(pwm / p_mean)
    return pwm

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"Loading inference results from {INFERENCE_FILE}")
    try:
        data = np.load(INFERENCE_FILE)
    except FileNotFoundError:
        print(f"Error: File {INFERENCE_FILE} not found.")
        return

    x_true_all = data['x_true'] # (N, L, 4)
    x_pred_all = data['x_pred'] # (N, L, 4)
    y_true_all = data['label'] # (N,)
    
    x_true_upstream_accum = np.zeros((PAD_BP, 4))
    x_pred_upstream_accum = np.zeros((PAD_BP, 4))
    count = 0
    
    for i in range(len(SMT3_WINDOWS)):
        chrom, start, end = SMT3_WINDOWS[i]
        x_true_seq = x_true_all[i]
        x_pred_seq = x_pred_all[i]
        
        # Calculate relative start of gene in window
        # Gene Start is 1469400. Window Start is 'start'.
        rel_start = GENE_START - start
        
        print(f"Window {i} ({chrom}:{start}-{end}) - Rel Start: {rel_start}")
        
        # Check if gene start is within window (plus padding space)
        # We need PAD_BP upstream
        if rel_start < PAD_BP:
            print(f"Skipping Window {i}: Not enough upstream context ({rel_start} < {PAD_BP})")
            continue
        if rel_start >= len(x_true_seq):
            print(f"Skipping Window {i}: Gene start beyond window end")
            continue
            
        # Extract Upstream Segment (PAD_BP length ending at gene start)
        # SMT3 is on + strand, so upstream is [start - pad, start]
        # x_true_seq is (L, 4)
        
        upstream_slice_start = rel_start - PAD_BP
        upstream_slice_end = rel_start
        
        upstream_true = x_true_seq[upstream_slice_start:upstream_slice_end]
        upstream_pred = x_pred_seq[upstream_slice_start:upstream_slice_end]
        
        if upstream_true.shape[0] != PAD_BP:
            print(f"Skipping Window {i}: Slice shape mismatch {upstream_true.shape}")
            continue

        # Generate Individual Plots
        indiv_dir = os.path.join(OUTPUT_DIR, "SMT3_individual")
        os.makedirs(indiv_dir, exist_ok=True)
        
        # # Plot True
        # plot_dna_logo(
        #     upstream_true, sequence_template="$" * PAD_BP,
        #     figsize=(35, 3), plot_start=PLOT_START, plot_end=PLOT_END,
        #     save_figs=True, fig_name=os.path.join(indiv_dir, f"win{i}_{start}_x_true")
        # )
        
        # # Plot Pred
        # plot_dna_logo(
        #     upstream_pred, sequence_template="$" * PAD_BP,
        #     figsize=(35, 3), plot_start=PLOT_START, plot_end=PLOT_END,
        #     save_figs=True, fig_name=os.path.join(indiv_dir, f"win{i}_{start}_x_pred")
        # )
        
        # Accumulate
        x_true_upstream_accum += upstream_true
        x_pred_upstream_accum += upstream_pred
        count += 1
        
    # Calculate Average
    if count > 0:
        x_true_avg = x_true_upstream_accum / count
        x_pred_avg = x_pred_upstream_accum / count
        
        # Plot Averages
        avg_dir = os.path.join(OUTPUT_DIR, "SMT3_average")
        os.makedirs(avg_dir, exist_ok=True)
        
        print(f"Plotting averages (Count: {count})...")
        
        plot_dna_logo(
            x_true_avg, sequence_template="$" * PAD_BP,
            figsize=(35, 3), plot_start=PLOT_START, plot_end=PLOT_END,
            save_figs=True, fig_name=os.path.join(avg_dir, "average_x_true")
        )
        
        plot_dna_logo(
            x_pred_avg, sequence_template="$" * PAD_BP,
            figsize=(35, 3), plot_start=PLOT_START, plot_end=PLOT_END,
            save_figs=True, fig_name=os.path.join(avg_dir, "average_x_pred")
        )
        
    #     if MODISCO_AVAILABLE:
    #         # Plot MODISCO style weights (Log Odds)
    #         x_true_norm = norm_pwm(x_true_avg)
    #         x_pred_norm = norm_pwm(x_pred_avg)
            
    #         plt.figure(figsize=(25, 1))
    #         viz_sequence.plot_weights(x_true_norm, subticks_frequency=10)
    #         plt.title("Average PWM (x_true) - SMT3")
    #         plt.savefig(os.path.join(avg_dir, "average_modisco_x_true.png"), transparent=False, dpi=300)
    #         plt.close()
            
    #         plt.figure(figsize=(25, 1))
    #         viz_sequence.plot_weights(x_pred_norm, subticks_frequency=10)
    #         plt.title("Average PWM (x_pred) - SMT3")
    #         plt.savefig(os.path.join(avg_dir, "average_modisco_x_pred.png"), transparent=False, dpi=300)
    #         plt.close()
        
    # else:
    #     print("No valid windows found for averaging.")

if __name__ == "__main__":
    main()
