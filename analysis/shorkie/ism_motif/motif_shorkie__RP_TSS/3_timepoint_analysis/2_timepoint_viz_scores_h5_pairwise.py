#!/usr/bin/env python3
from optparse import OptionParser
import numpy as np
import os
import h5py
import pandas as pd
import pysam
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from yeast_helpers import make_seq_1hot

# Additional imports for plotting letters
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch

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
def visualize_input_ism(att_grad_wt, plot_start=0, plot_end=100, save_figs=False, fig_name='', figsize=(12, 3)):
    scores_wt = att_grad_wt[plot_start:plot_end, :]

    # Get logo bounds
    y_min = np.min(scores_wt)
    y_max = np.max(scores_wt)
    y_max_abs = max(np.abs(y_min), np.abs(y_max))
    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs

    plot_seq_scores(
        scores_wt, y_min=y_min, y_max=y_max,
        figsize=figsize,
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + '_wt',
    )

#############################################
# Function to plot sequence logo            #
#############################################
def plot_seq_scores(importance_scores, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, save_figs=False, fig_name="default"):
    importance_scores = importance_scores.T

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot all 4 letters per position (sorted by magnitude)
    for j in range(importance_scores.shape[1]):
        sort_index = np.argsort(-np.abs(importance_scores[:, j]))  # largest magnitude first

        pos_height = 0.0
        neg_height = 0.0
        for ii in range(4):
            i = sort_index[ii]
            if i == 0:
                nt = 'A'
            elif i == 1:
                nt = 'C'
            elif i == 2:
                nt = 'G'
            elif i == 3:
                nt = 'T'
            
            nt_prob = importance_scores[i, j]
            if nt_prob >= 0.0:
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=pos_height,
                    yscale=nt_prob,
                    ax=ax,
                    color=None
                )
                pos_height += nt_prob
            else:
                dna_letter_at(
                    letter=nt,
                    x=j + 0.5,
                    y=neg_height,
                    yscale=nt_prob,
                    ax=ax,
                    color=None
                )
                neg_height += nt_prob

    plt.xlim((0, importance_scores.shape[1]))
    plt.yticks([] if not plot_y_ticks else None)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        plt.ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
        )
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
    plt.show()

###########################################################################
# Compute average for one time point (averaging over selected tracks)        #
###########################################################################
def compute_time_average_for_one_time(dataset, seq_idx, track_indices):
    # Convert and sort track_indices to a numpy array for fancy indexing
    track_indices = np.sort(np.array(track_indices, dtype=int))
    data_slice = dataset[seq_idx, :, :, track_indices]  # shape: (L, 4, #tracks)
    data_avg = data_slice.mean(axis=-1)  # shape: (L, 4)
    return data_avg

def calculate_sequence_distances(seq_data, timepoints):
    """Calculate pairwise distance matrix for a single sequence"""
    n_tps = len(timepoints)
    dist_matrix = np.zeros((n_tps, n_tps))
    
    # Get data in consistent order
    sorted_data = [seq_data[tp] for tp in timepoints]
    
    # Calculate all pairwise combinations
    for i in range(n_tps):
        for j in range(n_tps):
            if i <= j:
                # Flatten arrays and calculate Euclidean distance
                dist = np.linalg.norm(sorted_data[i].flatten() - sorted_data[j].flatten())
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
                
    return dist_matrix

def main():
    parser = OptionParser()
    parser.add_option('--exp_dir', default='', help='Experiment directory')
    parser.add_option('--plot_start', type=int, default=0, help='Plot start position')
    parser.add_option('--plot_end', type=int, default=500, help='Plot end position')
    parser.add_option('--target_tf', default='MSN2_', help='Target TF identifier')
    options, _ = parser.parse_args()

    # Configuration and data loading
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'
    target_f = f"{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq.txt"

    fasta_open = pysam.Fastafile(fasta_file)
    target_df = pd.read_csv(target_f, sep="\t")
    target_df = target_df[target_df['identifier'].str.contains(options.target_tf)]

    # # Extract timepoints
    # target_df['timepoint'] = target_df['identifier'].str.extract(r'_(T\d+)_', expand=False)
    # # Sort by time so that the order is consistent.
    # target_df = target_df.sort_values(by='timepoint').reset_index(drop=True)
    # timepoints = sorted(target_df['timepoint'].unique(), key=lambda x: int(x[1:]))

    target_df['timepoint'] = target_df['identifier'].str.extract(r'_(T\d+)_', expand=False)
    # Sort by time so that the order is consistent.
    target_df = target_df.sort_values(by='timepoint').reset_index(drop=True)
    print("\nTarget file with timepoint column added:")
    print(target_df.head())
    
    timepoints = sorted(target_df['timepoint'].unique(), key=lambda x: int(x[1:]))
        
    track_offset = 1148

    print(f"{options.exp_dir}/scores.h5: ")

    with h5py.File(f"{options.exp_dir}/scores.h5", 'r') as h5_file:
        # Dictionary to store sequence data: {seq_idx: {timepoint: data}}
        seq_time_data = defaultdict(dict)
        num_seqs = h5_file['chr'].shape[0]

        # Process each timepoint group
        for tp, group_df in target_df.groupby('timepoint'):
            track_indices = [idx - track_offset for idx in group_df['index']]
            print(f"Processing timepoint {tp} with track indices: {track_indices}")
            
            # Validate track indices
            if max(track_indices) >= h5_file['logSED'].shape[-1] or min(track_indices) < 0:
                raise ValueError("Invalid track indices")

            # Process each sequence
            for seq_idx in range(num_seqs):
                # Get sequence coordinates
                chrom = str(h5_file['chr'][seq_idx].decode())
                start = int(h5_file['start'][seq_idx])
                end = int(h5_file['end'][seq_idx])

                # Get reference sequence and compute averaged PWM
                seq_onehot = make_seq_1hot(fasta_open, chrom, start, end, end - start)
                pwm_data = compute_time_average_for_one_time(h5_file['logSED'], seq_idx, track_indices)
                pwm_norm = pwm_data - np.mean(pwm_data, axis=-1, keepdims=True)
                seq_time_data[seq_idx][tp] = pwm_norm[options.plot_start:options.plot_end]

                print(f"Processed sequence {seq_idx} for timepoint {tp}: {chrom}_{start}_{end}")
                print(seq_time_data[seq_idx][tp].shape)
                


        # Prepare output directory and list to hold max distances
        output_dir = f"{options.exp_dir}/distance_analysis_{options.target_tf}"
        os.makedirs(output_dir, exist_ok=True)
        all_distances = []
        valid_seqs = 0
        max_distances = []  # To store summary of maximum distances per sequence

        # Process each sequence for distance calculations
        for seq_idx in range(num_seqs):
            seq_data = seq_time_data.get(seq_idx, {})
            
            # Check for complete timepoint data
            if not all(tp in seq_data for tp in timepoints):
                continue
            
            # Retrieve sequence coordinate info
            chrom = str(h5_file['chr'][seq_idx].decode())
            start = int(h5_file['start'][seq_idx])
            end = int(h5_file['end'][seq_idx])
            seq_id = f"{chrom}_{start}_{end}"
            
            # Calculate distance matrix
            dist_matrix = calculate_sequence_distances(seq_data, timepoints)
            print(f"dist_matrix shape: {dist_matrix.shape}")
            all_distances.append(dist_matrix)
            valid_seqs += 1

            # Save individual sequence results in a folder named by its coordinates
            seq_dir = f"{output_dir}/{seq_id}"
            os.makedirs(seq_dir, exist_ok=True)
            
            # Save distance matrix CSV
            pd.DataFrame(dist_matrix, index=timepoints, columns=timepoints)\
              .to_csv(f"{seq_dir}/distance_matrix.csv")
            
            # Plot and save heatmap
            plt.figure(figsize=(5.34, 4))
            plt.imshow(dist_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Euclidean Distance')
            plt.xticks(ticks=range(len(timepoints)), labels=timepoints, rotation=45)
            plt.yticks(ticks=range(len(timepoints)), labels=timepoints)
            plt.title(f'Pairwise Distance Matrix of \nDNA Sequence Logos for             Promoters')
            plt.tight_layout()
            plt.savefig(f"{seq_dir}/distance_heatmap.png", dpi=150)
            plt.close()

            # Calculate the maximum distance (most distant pair of timepoints) for this sequence
            max_distance = np.max(dist_matrix)
            max_distances.append((seq_id, max_distance))

        print(f"Processed {valid_seqs} valid sequences with complete data")
        if valid_seqs == 0:
            print("No valid sequences for averaging")
        else:
            # Calculate average distance matrix over all valid sequences
            avg_distance = np.mean(all_distances, axis=0)
            avg_dir = f"{output_dir}/average"
            os.makedirs(avg_dir, exist_ok=True)
            
            pd.DataFrame(avg_distance, index=timepoints, columns=timepoints)\
              .to_csv(f"{avg_dir}/average_distance_matrix.csv")
    
            plt.figure(figsize=(5.34, 4))
            plt.imshow(avg_distance, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Average Distance')
            plt.title(f'Average Distance Matrix ({valid_seqs} sequences)')
            plt.xticks(ticks=range(len(timepoints)), labels=timepoints, rotation=45)
            plt.yticks(ticks=range(len(timepoints)), labels=timepoints)
            plt.tight_layout()
            plt.savefig(f"{avg_dir}/average_distance_heatmap.png", dpi=300)
            plt.close()
            
            # Plot consecutive timepoint distances
            if len(timepoints) > 1:
                consecutive_dists = [avg_distance[i, i+1] for i in range(len(timepoints)-1)]
                tp_pairs = [f"{timepoints[i]}-{timepoints[i+1]}" for i in range(len(timepoints)-1)]
    
                plt.figure(figsize=(12,6))
                plt.plot(tp_pairs, consecutive_dists, 'o-', markersize=8)
                plt.xlabel('Consecutive Timepoint Pairs')
                plt.ylabel('Average Distance')
                plt.title(f'Consecutive Timepoint Distances ({valid_seqs} sequences)')
                plt.xticks(rotation=45)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{avg_dir}/consecutive_distances.png", dpi=300)
                plt.close()
    
            # Create summary CSV and bar chart for the maximum distances
            if max_distances:
                summary_df = pd.DataFrame(max_distances, columns=['Sequence', 'MaxDistance'])
                summary_csv_path = f"{output_dir}/max_distance_summary.csv"
                summary_df.to_csv(summary_csv_path, index=False)
        
                plt.figure(figsize=(12, 6))
                plt.bar(summary_df['Sequence'], summary_df['MaxDistance'])
                plt.xlabel('Sequence (chr_start_end)')
                plt.ylabel('Max Distance')
                plt.title('Max Distance per Sequence')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/max_distance_bar_chart.png", dpi=300)
                plt.close()

if __name__ == "__main__":
    main()
