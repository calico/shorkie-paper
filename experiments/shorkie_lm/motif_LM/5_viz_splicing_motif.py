import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import os

def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = ((np.log((ppm + pseudocount) / (1 + pseudocount * alphabet_len)) / np.log(2))
          * ppm - (np.log(background) * background / np.log(2))[None, :])
    return np.sum(ic, axis=1)


def _plot_weights(array, path, figsize=(10,3), nucleotide_width=0.7, base_height=3):
    """Plot weights as a sequence logo and save to file."""
    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'

    num_positions = len(df)
    figure_width = num_positions * nucleotide_width
    figsize = (figure_width, base_height)
    print("figsize: ", figsize)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    
    crp_logo = logomaker.Logo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())

    # Remove x and y axis labels and ticks
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(path, dpi=300)
    plt.tight_layout()
    plt.close()



def sequence_to_pwm(seq):
    """Convert a nucleotide sequence into a one-hot PWM matrix."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    pwm = np.zeros((len(seq), 4))
    for i, nucleotide in enumerate(seq.upper()):
        if nucleotide in mapping:
            pwm[i, mapping[nucleotide]] = 1.0
        else:
            raise ValueError(f"Invalid nucleotide: {nucleotide}")
    return pwm

def make_logo(seq, logo_dir, nucleotide_width=0.7, base_height=3):
    if seq == 'NA':
        return
    
    # Convert sequence to PWM and compute per-position information content
    background = np.array([0.25, 0.25, 0.25, 0.25])
    pwm = sequence_to_pwm(seq)
    ic = compute_per_position_ic(pwm, background, 0.001)
    
    # Multiply each row of the PWM by its information content
    logo_array = pwm * ic[:, None]
    
    # Ensure the output directory exists
    os.makedirs(logo_dir, exist_ok=True)
    output_path = os.path.join(logo_dir, f"{seq}.png")
    
    _plot_weights(logo_array, path=output_path, nucleotide_width=nucleotide_width, base_height=base_height)
    print(f"Logo saved to {output_path}")

if __name__ == "__main__":
    # Define the motifs to visualize
    motifs = ["GTATGT", "TACTAAC", "CAT"]
    logo_dir = "viz_motifs"
    nucleotide_width = 0.7
    base_height = 3

    for motif in motifs:
        make_logo(motif, logo_dir, nucleotide_width, base_height)
