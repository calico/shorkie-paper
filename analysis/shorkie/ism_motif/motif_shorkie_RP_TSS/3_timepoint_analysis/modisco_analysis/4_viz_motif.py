#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

import logomaker

########################################
# Helper: read a MEME file (if needed)
########################################
def read_meme(filename):
    motifs = {}
    with open(filename, "r") as infile:
        motif, width, i = None, None, 0
        for line in infile:
            if motif is None:
                if line.startswith('MOTIF'):
                    motif = line.split()[1]
                else:
                    continue
            elif width is None:
                if line.startswith('letter'):
                    width = int(line.split()[5])
                    pwm = np.zeros((width, 4))
            elif i < width:
                pwm[i] = list(map(float, line.split()))
                i += 1
            else:
                motifs[motif] = pwm
                motif, width, i = None, None, 0
    return motifs

########################################
# Helper: compute per-position IC
########################################
def compute_per_position_ic(ppm, background, pseudocount):
    """
    ppm: shape (width, 4)
    background: shape (4,) – e.g. [0.25, 0.25, 0.25, 0.25]
    """
    ic = (np.log((ppm + pseudocount) / (background + pseudocount)) / np.log(2)) * ppm
    return np.sum(ic, axis=1)

########################################
# Helper: plot weights as a sequence logo
# (with an optional global_ylim)
########################################
def _plot_weights(array, path, nucleotide_width=0.7, base_height=3, global_ylim=None):
    """
    Plot 'array' as a sequence logo and save to file.
    If global_ylim is given as (ymin, ymax), all logos share that vertical range.
    """
    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'
    
    num_positions = len(df)
    figure_width = num_positions * nucleotide_width
    figsize = (figure_width, base_height)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    crp_logo = logomaker.Logo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    
    if global_ylim is not None:
        plt.ylim(global_ylim[0], global_ylim[1])
    else:
        data_min = df.sum(axis=1).min()
        data_max = df.sum(axis=1).max()
        plt.ylim(min(data_min, 0), data_max)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

########################################
# Gather global y-limits for all motifs
########################################
def find_global_ylim(modisco_h5py_files,
                     pattern_groups=('pos_patterns', 'neg_patterns'),
                     trim_threshold=0.3,
                     trim_extension=4):
    """
    Parse all modisco HDF5 files, compute the sum across A/C/G/T at each position,
    track the min and max of these sums across *all* patterns in *all* files.

    Returns (global_min, global_max).
    """
    global_min = np.inf
    global_max = -np.inf

    for modisco_file in modisco_h5py_files:
        if not os.path.isfile(modisco_file):
            continue
        with h5py.File(modisco_file, 'r') as modisco_results:
            for group_name in pattern_groups:
                if group_name not in modisco_results.keys():
                    continue
                metacluster = modisco_results[group_name]
                
                def sort_key(x):
                    idx = int(x[0].split("_")[-1])  # pattern_0 -> 0, pattern_1 -> 1, etc.
                    return idx

                for pattern_name, pattern in sorted(metacluster.items(), key=sort_key):
                    cwm_fwd = np.array(pattern['contrib_scores'][:])  # shape (L, 4)
                    cwm_rev = cwm_fwd[::-1, ::-1]

                    score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
                    score_rev = np.sum(np.abs(cwm_rev), axis=1)

                    trim_thresh_fwd = np.max(score_fwd) * trim_threshold
                    trim_thresh_rev = np.max(score_rev) * trim_threshold

                    pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
                    pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]
                    
                    if len(pass_inds_fwd) > 0:
                        start_fwd = max(pass_inds_fwd.min() - trim_extension, 0)
                        end_fwd   = min(pass_inds_fwd.max() + trim_extension + 1, len(score_fwd))
                    else:
                        start_fwd, end_fwd = 0, len(score_fwd)

                    if len(pass_inds_rev) > 0:
                        start_rev = max(pass_inds_rev.min() - trim_extension, 0)
                        end_rev   = min(pass_inds_rev.max() + trim_extension + 1, len(score_rev))
                    else:
                        start_rev, end_rev = 0, len(score_rev)

                    trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
                    trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

                    sums_fwd = trimmed_cwm_fwd.sum(axis=1)
                    sums_rev = trimmed_cwm_rev.sum(axis=1)

                    cur_min = min(sums_fwd.min(), sums_rev.min())
                    cur_max = max(sums_fwd.max(), sums_rev.max())

                    if cur_min < global_min:
                        global_min = cur_min
                    if cur_max > global_max:
                        global_max = cur_max

    # Force the lower bound not to exceed 0 if you want:
    global_min = min(global_min, 0.0)
    return (global_min, global_max)

########################################
# Main driver
########################################
if __name__ == "__main__":
    
    # Example definitions:
    # exp_dirs = ["gene_exp_motif_test_TSS/f0c0", "gene_exp_motif_test_RP/f0c0"]

    # exp_dirs = ["gene_exp_motif_test_MSN4_targets/f0c0"]
    # target_tfs = ["MSN4"]

    exp_dirs = ["gene_exp_motif_test_MSN2_targets/f0c0"]
    target_tfs = ["MSN2"]

    # exp_dirs = ["gene_exp_motif_test_MET4_targets/f0c0"]
    # target_tfs = ["MET4"]
    time_points = ["T0", "T5", "T10", "T15", "T30", "T45", "T60", "T90"]

    # exp_dirs = ["gene_exp_motif_test_SWI4_targets/f0c0"]
    # target_tfs = ["SWI4"]
    # time_points = ["T0", "T5", "T10", "T20", "T40", "T70", "T120", "T180"]

    scores = ["logSED"]
    n = 10000
    w = 500

    # Build list of modisco .h5 files
    modisco_h5py_files = []
    for exp_dir in exp_dirs:
        for tf in target_tfs:
            for tp in time_points:
                score = scores[0]  # only one score in example
                # e.g. "results/gene_exp_motif_test_TSS/f0c0/MSN2_/T0/modisco_results_10000_500_diff.h5"
                path = f"results/{exp_dir}/{tf}/{tp}/modisco_results_{n}_{w}_diff.h5"
                modisco_h5py_files.append(path)

    # Root for all output
    output_root = "viz_self_modisco_sharedY"
    os.makedirs(output_root, exist_ok=True)

    # PASS 1: find global Y-limits
    (global_ymin, global_ymax) = find_global_ylim(
        modisco_h5py_files,
        pattern_groups=['pos_patterns','neg_patterns'],
        trim_threshold=0.3,
        trim_extension=4
    )
    # If you wish to override global_ymax or global_ymin, you can do so here:
    # global_ymin = -0.001
    # global_ymax = 0.0063
    # global_ymin = -0.001
    # global_ymax = 0.0210 
    
    # MSN2    
    global_ymin = -0.001
    global_ymax = 0.026

    print(f"Global Y-axis: (min, max) = ({global_ymin:.4f}, {global_ymax:.4f})")

    # PASS 2: plot each motif
    for modisco_file in modisco_h5py_files:
        if not os.path.isfile(modisco_file):
            print(f"[WARN] File not found: {modisco_file}")
            continue

        # Remove leading "results/" so we can replicate its structure under 'output_root'
        rel_path = os.path.relpath(modisco_file, start="results")
        # e.g. "gene_exp_motif_test_TSS/f0c0/MSN2_/T0/modisco_results_10000_500_diff.h5"

        # Remove '.h5' so we can name a directory after the file
        rel_path_no_ext = rel_path.replace(".h5", "")
        # e.g. "gene_exp_motif_test_TSS/f0c0/MSN2_/T0/modisco_results_10000_500_diff"

        # Build the full output path
        # e.g. "viz_self_modisco_sharedY/gene_exp_motif_test_TSS/f0c0/MSN2_/T0/modisco_results_10000_500_diff"
        logo_dir = os.path.join(output_root, rel_path_no_ext)
        os.makedirs(logo_dir, exist_ok=True)

        # Open the modisco file and loop over patterns
        with h5py.File(modisco_file, 'r') as hf:
            for group_name in ['pos_patterns','neg_patterns']:
                if group_name not in hf:
                    continue
                metacluster = hf[group_name]

                # Also create a subdirectory for each group (pos/neg) if desired
                # e.g. ".../modisco_results_10000_500_diff/pos_patterns"
                group_dir = os.path.join(logo_dir, group_name)
                os.makedirs(group_dir, exist_ok=True)

                def sort_key(x):
                    return int(x[0].split("_")[-1])  # pattern_0 -> 0, pattern_1 -> 1, etc.

                for pattern_name, pattern in sorted(metacluster.items(), key=sort_key):
                    cwm_fwd = np.array(pattern['contrib_scores'][:])   # (L, 4)
                    cwm_rev = cwm_fwd[::-1, ::-1]

                    score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
                    score_rev = np.sum(np.abs(cwm_rev), axis=1)

                    trim_threshold = 0.3
                    trim_extension = 4
                    trim_thresh_fwd = score_fwd.max() * trim_threshold
                    trim_thresh_rev = score_rev.max() * trim_threshold

                    pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
                    pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

                    if len(pass_inds_fwd) > 0:
                        start_fwd = max(pass_inds_fwd.min() - trim_extension, 0)
                        end_fwd   = min(pass_inds_fwd.max() + trim_extension + 1, len(score_fwd))
                    else:
                        start_fwd, end_fwd = 0, len(score_fwd)

                    if len(pass_inds_rev) > 0:
                        start_rev = max(pass_inds_rev.min() - trim_extension, 0)
                        end_rev   = min(pass_inds_rev.max() + trim_extension + 1, len(score_rev))
                    else:
                        start_rev, end_rev = 0, len(score_rev)

                    trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
                    trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

                    # Construct final filename for forward & reverse
                    # e.g. ".../pos_patterns/pattern_0.cwm.fwd.png"
                    pattern_prefix = pattern_name  # e.g. "pattern_0"
                    fwd_filename = f"{pattern_prefix}.cwm.fwd.png"
                    rev_filename = f"{pattern_prefix}.cwm.rev.png"

                    fwd_out_path = os.path.join(group_dir, fwd_filename)
                    rev_out_path = os.path.join(group_dir, rev_filename)

                    # Plot logos
                    _plot_weights(trimmed_cwm_fwd,
                                  path=fwd_out_path,
                                  nucleotide_width=0.7,
                                  base_height=3,
                                  global_ylim=(global_ymin, global_ymax))

                    _plot_weights(trimmed_cwm_rev,
                                  path=rev_out_path,
                                  nucleotide_width=0.7,
                                  base_height=3,
                                  global_ylim=(global_ymin, global_ymax))

                    print(f"[OK] {group_name}.{pattern_name} -> {fwd_out_path}, {rev_out_path}")

    print("[Done] All motifs plotted with a shared y-axis scale.")
