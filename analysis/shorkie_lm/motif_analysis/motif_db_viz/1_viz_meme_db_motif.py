# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

# logomaker import
import logomaker

def read_meme(filename):
    motifs = {}

    with open(filename, "r") as infile:
        motif, width, i = None, None, 0

        for line in infile:
            if motif is None:
                if line[:5] == 'MOTIF':
                    motif = line.split()[1]
                else:
                    continue

            elif width is None:
                if line[:6] == 'letter':
                    width = int(line.split()[5])
                    pwm = np.zeros((width, 4))

            elif i < width:
                pwm[i] = list(map(float, line.split()))
                i += 1

            else:
                motifs[motif] = pwm
                motif, width, i = None, None, 0

    return motifs

def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = ((np.log((ppm + pseudocount) / (1 + pseudocount * alphabet_len)) / np.log(2))
          * ppm - (np.log(background) * background / np.log(2))[None, :])
    return np.sum(ic, axis=1)

def reverse_complement_pwm(ppm):
    """
    Compute the reverse complement of a PWM.
    Assumes the columns are in the order: A, C, G, T.
    This function reverses the row order and swaps the columns (A <-> T and C <-> G).
    """
    # Reverse the rows and swap columns: A->T, C->G, G->C, T->A.
    return ppm[::-1, :][:, [3, 2, 1, 0]]


def _plot_weights(array, path, nucleotide_width=0.7, base_height=3):
    """
    Plot a DNA logo from the weighted matrix and save it to a file.
    The weighted matrix should be the PPM multiplied by the per-position information content.
    """
    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'
    
    num_positions = len(df)
    figure_width = num_positions * nucleotide_width
    figsize = (figure_width, base_height)
    print("figsize:", figsize)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())
    
    # Remove axis labels and ticks
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_logo(match, logo_dir, motifs, nucleotide_width, base_height):
    if match == 'NA':
        return

    background = np.array([0.25, 0.25, 0.25, 0.25])
    ppm = motifs[match]
    # Clean the name to use in filenames
    match_clean = match.replace("/", "_")
    
    # Original motif logo
    ic = compute_per_position_ic(ppm, background, 0.001)
    _plot_weights(ppm * ic[:, None],
                  path='{}/{}.png'.format(logo_dir, match_clean),
                  nucleotide_width=nucleotide_width, base_height=base_height)
    
    # Reverse complement motif logo
    rc_ppm = reverse_complement_pwm(ppm)
    ic_rc = compute_per_position_ic(rc_ppm, background, 0.001)
    _plot_weights(rc_ppm * ic_rc[:, None],
                  path='{}/{}_rc.png'.format(logo_dir, match_clean),
                  nucleotide_width=nucleotide_width, base_height=base_height)

# Example usage
if __name__ == "__main__":
    nucleotide_width = 0.7
    base_height = 3

    ##############################
    # Visualize MEME database motifs and their reverse complements
    ##############################
    motif_db_dir = 'viz_self_motif_db/'
    os.makedirs(motif_db_dir, exist_ok=True)
    meme_motif_db = '/home/kchao10/tools/motif_databases/YEAST/merged_meme.meme'
    motifs = read_meme(meme_motif_db)
    print("motifs: ", len(motifs))
    
    for gene, motif in motifs.items():
        print("gene: ", gene)
        print("motif: ", motif)
        make_logo(gene, motif_db_dir, motifs, nucleotide_width, base_height)
