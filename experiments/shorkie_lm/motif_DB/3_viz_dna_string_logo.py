#!/usr/bin/env python3
"""
Simple script to visualize a DNA sequence (including ambiguous positions) as a sequence logo.
Usage:
  python dna_logo.py --sequence G(C/A)GATGAG(A/C)TGA --output_logo seq_logo.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import argparse


def parse_motif(motif_str):
    """
    Parse a motif string containing optional parentheses with slash-separated ambiguous bases.
    E.g. 'G(C/A)GATGAG(A/C)TGA' -> ['G', 'C/A', 'G', ...]
    """
    tokens = []
    i = 0
    while i < len(motif_str):
        if motif_str[i] == '(':
            end = motif_str.find(')', i)
            if end == -1:
                raise ValueError("Unmatched '(' in motif string")
            tokens.append(motif_str[i+1:end])
            i = end + 1
        else:
            tokens.append(motif_str[i])
            i += 1
    return tokens


def sequence_to_ppm(sequence_tokens, pseudocount=0.0):
    """
    Convert a list of tokens into a position probability matrix (L x 4).
    - Single bases 'A','C','G','T' get one-hot.
    - 'N' (unknown) gets uniform distribution [0.25,0.25,0.25,0.25].
    - Ambiguous tokens like 'C/A' are split equally.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequence_tokens)
    ppm = np.zeros((L, 4))
    for i, token in enumerate(sequence_tokens):
        tok = token.upper()
        if tok == 'N':
            ppm[i, :] = 0.25
        elif '/' in tok:
            alleles = tok.split('/')
            weight = 1.0 / len(alleles)
            for allele in alleles:
                idx = mapping.get(allele)
                if idx is not None:
                    ppm[i, idx] = weight
        else:
            idx = mapping.get(tok)
            if idx is not None:
                ppm[i, idx] = 1.0
            else:
                # fallback to uniform for any other unknown symbol
                ppm[i, :] = 0.25
    return ppm


def compute_ic(ppm, background=None, pseudocount=0.001):
    """
    Compute per-position information content vector (L,).
    """
    if background is None:
        background = np.array([0.25] * ppm.shape[1])
    alpha = len(background)
    # Information content per letter
    info = np.log2((ppm + pseudocount) / (background * (1 + pseudocount * alpha))) * ppm
    ic = np.sum(info, axis=1)
    return ic


def plot_sequence_logo(sequence, output_path, nuc_width=0.7, base_height=3):
    """
    Generate and save a sequence logo for the given motif (string or token list).
    """
    # If string, parse ambiguous tokens
    if isinstance(sequence, str):
        tokens = parse_motif(sequence)
    else:
        tokens = sequence

    ppm = sequence_to_ppm(tokens)
    ic = compute_ic(ppm)
    weighted = ppm * ic[:, None]

    df = pd.DataFrame(weighted, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'
    fig, ax = plt.subplots(figsize=(len(tokens) * nuc_width, base_height))
    logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize a DNA sequence (with ambiguities) as a logo.")
    # parser.add_argument('--sequence', '-s', required=True,
    #                     help="Motif string (e.g. G(C/A)GATGAG(A/C)TGA)")
    parser.add_argument('--output_logo', '-o', default=None,
                        help='Path to save the logo image')
    parser.add_argument('--nuc_width', type=float, default=0.7,
                        help='Width per nucleotide')
    parser.add_argument('--base_height', type=float, default=3,
                        help='Height of the logo base')
    args = parser.parse_args()

    # Determine output path
    # sequence = "G(C/A)GATGAG(A/C)TGA"
    sequence = "TATATA"
    # sequence = "G"
    output_path = args.output_logo or f"{sequence}.png"
    plot_sequence_logo(sequence, output_path,
                       args.nuc_width, args.base_height)
    print(f"Logo saved to {output_path}")


if __name__ == '__main__':
    main()
