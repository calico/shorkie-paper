# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logomaker

def read_pfm(filename):
    """
    Reads a PFM file and returns a position probability matrix (PPM) of shape (num_positions, 4)
    with columns in order A, C, G, T.
    """
    nucleotides_data = {}
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            tokens = line.strip().split()
            letter = tokens[0].upper()
            try:
                values = list(map(float, tokens[1:]))
            except ValueError:
                continue
            nucleotides_data[letter] = values

    # Ensure order A, C, G, T
    ordered = []
    for letter in ['A', 'C', 'G', 'T']:
        ordered.append(nucleotides_data.get(letter, [0] * len(next(iter(nucleotides_data.values())))))
    matrix = np.array(ordered).T

    # Normalize to probabilities if needed
    row_sums = matrix.sum(axis=1, keepdims=True)
    ppm = matrix / row_sums
    return ppm


def compute_per_position_ic(ppm, background, pseudocount):
    """
    Compute per-position information content for a PPM.
    """
    alphabet_len = len(background)
    ic = ((np.log((ppm + pseudocount) / (1 + pseudocount * alphabet_len)) / np.log(2)) * ppm
          - (np.log(background) * background / np.log(2))[None, :])
    return np.sum(ic, axis=1)


def _plot_weights(array, path, nucleotide_width=0.7, base_height=3):
    """
    Plot a DNA logo from the weighted matrix and save it.
    """
    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'
    num_positions = len(df)
    figsize = (num_positions * nucleotide_width, base_height)

    fig, ax = plt.subplots(figsize=figsize)
    logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_logo_from_ppm(ppm, background, pseudocount, nucleotide_width, base_height, output_path):
    """
    Compute IC, weight the PPM by IC, and plot the logo.
    """
    ic = compute_per_position_ic(ppm, background, pseudocount)
    weighted = ppm * ic[:, None]
    _plot_weights(weighted, output_path, nucleotide_width, base_height)


if __name__ == "__main__":
    # Input PFM and output prefixes
    # pfm_file = 'MA0350.2.pfm'
    pfm_file = 'MA0390.1.pfm'
    base_name, _ = os.path.splitext(pfm_file)
    output_fwd = f'{base_name}_forward.png'
    output_rev = f'{base_name}_revcomp.png'

    # Logo parameters
    nucleotide_width = 0.7
    base_height = 3
    pseudocount = 0.001
    background = np.array([0.25, 0.25, 0.25, 0.25])

    # Read PFM
    pfm = read_pfm(pfm_file)
    print(f"PFM shape: {pfm.shape}")

    # Forward logo
    plot_logo_from_ppm(pfm, background, pseudocount,
                       nucleotide_width, base_height, output_fwd)
    print(f"Forward DNA logo saved to: {output_fwd}")

    # Reverse complement: reverse positions and swap A<->T, C<->G
    rc_ppm = pfm[::-1, :][:, [3, 2, 1, 0]]
    plot_logo_from_ppm(rc_ppm, background, pseudocount,
                       nucleotide_width, base_height, output_rev)
    print(f"Reverse-complement DNA logo saved to: {output_rev}")
