#!/usr/bin/env python3
from optparse import OptionParser
import numpy as np
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import pysam
import re

from yeast_helpers import make_seq_1hot  # sequence one-hot helper

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
    }
    COLORS = {'G':'orange','A':'green','C':'blue','T':'red'}

    text = LETTERS[letter]
    col  = COLORS[letter] if color is None else color

    transform = (
        mpl.transforms.Affine2D()
        .scale(globscale, yscale * globscale)
        .translate(x, y)
        + ax.transData
    )
    patch = PathPatch(text, lw=0, fc=col, alpha=alpha, transform=transform)
    ax.add_artist(patch)
    return patch

#############################################
# Plot sequence logo with matched y-axis    #
#############################################
def plot_seq_scores(imp, figsize=(16,2), y_min=None, y_max=None, save_fig=False, fname=None):
    # imp: (L,4) importance or PWM values for A,C,G,T
    L = imp.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    for pos in range(L):
        order = np.argsort(-np.abs(imp[pos]))
        pos_height, neg_height = 0.0, 0.0
        for idx in order:
            nt = "ACGT"[idx]
            val = imp[pos, idx]
            if val >= 0:
                dna_letter_at(nt, x=pos+0.5, y=pos_height, yscale=val, ax=ax)
                pos_height += val
            else:
                dna_letter_at(nt, x=pos+0.5, y=neg_height, yscale=val, ax=ax)
                neg_height += val

    ax.set_xlim(0, L)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_yticks([])
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        mx = np.max(np.abs(imp))
        ax.set_ylim(-mx*1.05, mx*1.05)
    plt.tight_layout()
    if save_fig and fname:
        plt.savefig(fname + ".png", dpi=300, transparent=True)
    plt.show()

###########################################################################
# Average over selected tracks and pick first 4 channels                 #
###########################################################################
def compute_pwm(dataset, seq_idx, track_ids):
    # dataset: (N_seq, L, C=170, T)
    arr = dataset[seq_idx, :, :, track_ids]      # -> (L, C, #tracks)
    avg = arr.mean(axis=-1)                     # -> (L, C)
    return avg[:, :4]                           # only A,C,G,T channels

###########################################################################
# Main routine                                                           #
###########################################################################
def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('--exp_dir',   dest='exp_dir',   default='',    type='str', help='Experiment directory')
    parser.add_option('--plot_start',dest='plot_start',default=0,     type='int', help='Start pos')
    parser.add_option('--plot_end',  dest='plot_end',  default=None,  type='int', help='End pos')
    opts, _ = parser.parse_args()

    target_f = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt"

    # load track sheet and select T0
    sheet = pd.read_csv(
        target_f, sep="\t"
    )
    sheet = sheet[sheet['identifier'].str.contains('_T0_')]
    raw_ids = sheet['index'].astype(int).tolist()
    offset  = 1148
    track_ids = [i - offset for i in raw_ids]

    # open HDF5
    h5 = h5py.File(os.path.join(opts.exp_dir, 'scores.h5'), 'r')
    pwm_ds  = h5['logSED']     # (N_seq, L, 170, T)
    seqs_ds = h5['seqs']       # (N_seq, L, 4)
    labels  = [lbl.decode('utf-8') for lbl in h5['label'][:]]

    N_seq, L = pwm_ds.shape[0], pwm_ds.shape[1]
    start, end = opts.plot_start, opts.plot_end or L

    # prepare output
    outdir = os.path.join(opts.exp_dir, 'dna_logo_example/logos/logSED/T0_average')
    os.makedirs(outdir, exist_ok=True)

    root_dir="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML"
    fasta_path = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'

    fasta = pysam.Fastafile(fasta_path)

    # viz_all_true = []
    # viz_all_pred = []
    viz_true, viz_pred = [], []

    for si in range(N_seq):

        # 
        # parse coords from label e.g. chrIV:65375_C / chrIV:65375_A / chrIV:65375_G / chrIV:65375_T
        m = re.match(r"(.+?):(\d+)_([ACGT])", labels[si])
        if not m:
            raise ValueError(f"Cannot parse label '{labels[si]}'")
        chrom, start16, nt = m.group(1), int(m.group(2)), str(m.group(3))
        print("chrom: ", chrom)
        print("start16: ", start16)
        print("nt: ", nt)

        # m = re.match(r"(.+?):(\d+)-(\d+)", labels[si])
        # if not m:
        #     raise ValueError(f"Cannot parse label '{labels[si]}'")
        # chrom, start16, end16 = m.group(1), int(m.group(2)), int(m.group(3))


        # use seqs_ds for reference one-hot
        seq1hot = seqs_ds[si]        # (L,4)
        seq1hot = seq1hot[:, :4]  # (L,4)
        # compute PWM averaged over T0 tracks
        pwm = compute_pwm(pwm_ds, si, track_ids)
        # mean-normalize
        mean_v = pwm.mean(axis=1, keepdims=True)
        pwm_norm = pwm - mean_v
        # reference-only PWM
        print("pwm_norm.shape: ", pwm_norm.shape)
        print("seq1hot.shape: ", seq1hot.shape)
        viz = pwm_norm * seq1hot
        # clip
        clip_viz = viz[start:end, :]
        clip_seq = seq1hot[start:end, :]

        viz_true.append(clip_seq)
        viz_pred.append(clip_viz)

        # plot and save
        mn, mx = clip_viz.min(), clip_viz.max()
        rng = max(abs(mn), abs(mx))
        fname = os.path.join(outdir, f"logo_seq_{chrom}:{start16}_{nt}_{si:02d}")
        plot_seq_scores(
            clip_viz,
            figsize=(min(200, (end-start)/5),3),
            y_min=-rng*1.05, y_max=rng*1.05,
            save_fig=True, fname=fname
        )
        print(f"Saved logo for seq {si}: {fname}.png")

    # save NPZ
    true_arr = np.stack(viz_true, axis=0)
    pred_arr = np.stack(viz_pred, axis=0)
    np.savez(os.path.join(outdir, 'true.npz'), true_arr)
    np.savez(os.path.join(outdir, 'pred.npz'), pred_arr)
    print(f"Saved NPZ in {outdir}")

    h5.close()

if __name__ == "__main__":
    main()
