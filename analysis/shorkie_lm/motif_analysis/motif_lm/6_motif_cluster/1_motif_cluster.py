#!/usr/bin/env python3
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import pandas as pd

def parse_motifs_html(motifs_html_file, qval_threshold):
    """
    Parse the motifs.html report to extract motif mapping.
    Expects the table to have columns: 'pattern', 'match0', 'qval0'.
    Returns a dictionary mapping modisco pattern IDs (with "/" as a separator)
    to a dict with keys 'best_match' and 'qval'.
    """
    dfs = pd.read_html(motifs_html_file)
    if not dfs:
        print("No tables found in motifs.html")
        return {}
    df = dfs[0]
    mapping = {}
    for idx, row in df.iterrows():
        raw_pattern = row['pattern']
        # Convert the raw pattern to the modisco key format, e.g., "pos_patterns/pattern_0"
        modisco_pattern_id = raw_pattern.replace(".", "/")
        try:
            qval = float(row['qval0'])
        except (ValueError, TypeError):
            qval = None
        best_match = row['match0'] if pd.notnull(row['match0']) else raw_pattern
        mapping[modisco_pattern_id] = {'best_match': best_match, 'qval': qval}
    return mapping

def extract_seqlet_positions(h5_filepath):
    """
    Opens the TF-MoDISco results HDF5 file and extracts the start and end positions,
    along with example index and reverse complement flag, for each seqlet in each motif.
    """
    seqlet_positions = {}
    with h5py.File(h5_filepath, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type in f:
                seqlet_positions[pattern_type] = {}
                for pattern_name in f[pattern_type].keys():
                    pattern_group = f[pattern_type][pattern_name]
                    if "seqlets" in pattern_group:
                        seqlets_group = pattern_group["seqlets"]
                        starts = seqlets_group["start"][:]
                        ends = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        seqlet_positions[pattern_type][pattern_name] = {
                            "start": starts,
                            "end": ends,
                            "example_idx": example_idxs,
                            "is_revcomp": revcomps
                        }
    return seqlet_positions

def extract_seqlet_data(h5_filepath, pattern_type='pos_patterns', pattern_name=None):
    """
    Extracts the seqlet sequence data (assumed to be one-hot encoded) for a given motif.
    If pattern_name is None, the first available pattern in the given pattern_type is used.
    
    It inspects available keys in the "seqlets" group and uses the first dataset found.
    """
    with h5py.File(h5_filepath, 'r') as f:
        if pattern_type not in f:
            raise ValueError(f"Pattern type {pattern_type} not found in file.")
        group = f[pattern_type]
        if pattern_name is None:
            pattern_name = list(group.keys())[0]
        pattern_group = group[pattern_name]
        if "seqlets" not in pattern_group:
            raise ValueError(f"No seqlets group found for pattern {pattern_name}.")
        seqlets_group = pattern_group["seqlets"]
        
        # Print available keys in the seqlets group for debugging
        available_keys = list(seqlets_group.keys())
        print(f"Available keys in seqlets group for pattern {pattern_name}: {available_keys}")
        
        # Try to use the dataset named "seqlets" if available, otherwise pick the first dataset.
        if "seqlets" in available_keys and isinstance(seqlets_group["seqlets"], h5py.Dataset):
            seqlet_data = seqlets_group["seqlets"][:]
            print("Using dataset 'seqlets' for sequence data.")
        else:
            seqlet_data = None
            for key in available_keys:
                if isinstance(seqlets_group[key], h5py.Dataset):
                    seqlet_data = seqlets_group[key][:]
                    print(f"Using dataset '{key}' for sequence data.")
                    break
            if seqlet_data is None:
                raise ValueError(f"Sequence data not found for pattern {pattern_name} in the seqlets group.")
    return pattern_name, seqlet_data

def perform_hierarchical_clustering(seqlets_data):
    """
    Performs hierarchical clustering on the seqlet data.
    
    Parameters:
        seqlets_data (np.ndarray): Array of shape (n_seqlets, L, 4).
    
    Returns:
        Z (np.ndarray): The linkage matrix from hierarchical clustering.
    """
    n_seqlets, L, n_channels = seqlets_data.shape
    # Flatten each seqlet (from L x 4 to a 1D vector)
    seqlets_flat = seqlets_data.reshape((n_seqlets, L * n_channels))
    # Compute pairwise distances (Euclidean by default; change metric if desired)
    distance_matrix = pdist(seqlets_flat, metric='euclidean')
    # Perform hierarchical clustering using Ward's method
    Z = linkage(distance_matrix, method='ward')
    return Z

def plot_dendrogram(Z, motif_name, n_seqlets, output_fig, qval0=None):
    """
    Plots the dendrogram for the given hierarchical clustering result.
    If qval0 is provided, it is included in the plot title.
    The figure width is dynamic and proportional to the number of seqlets.
    """
    # Dynamic width: e.g., at least 12, or 0.3 * number of seqlets (whichever is greater)
    fig_width = max(12, n_seqlets * 0.04)
    plt.figure(figsize=(fig_width, 6))
    dendrogram(Z,
               labels=[str(i) for i in range(n_seqlets)],
               leaf_rotation=90)
    if qval0 is not None:
        plt.title(f"Hierarchical Clustering Dendrogram for Motif '{motif_name}' (qval0: {qval0})")
    else:
        plt.title(f"Hierarchical Clustering Dendrogram for Motif '{motif_name}'")
    # plt.xlabel("Seqlet Index")
    plt.ylabel("Euclidean Distance")
    plt.tight_layout()
    plt.savefig(output_fig)
    plt.close()
    print(f"Saved dendrogram to: {output_fig}")

if __name__ == "__main__":
    # Path to your TF-MoDISco HDF5 results file.
    modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5"
    
    # Path to the motifs.html file for motif mapping.
    motifs_html = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/report_w16384_n100000/motifs.html"
    mapping_file = "motif_mapping.tsv"
    
    # Create mapping dictionary from motifs.html if available.
    mapping = {}
    if os.path.exists(motifs_html):
        mapping = parse_motifs_html(motifs_html, qval_threshold=0.5)
        print("Parsed motif mapping from HTML.")
        with open(mapping_file, "w") as f:
            f.write("modisco_pattern\tbest_match\tqval0\n")
            for key, value in mapping.items():
                f.write(f"{key}\t{value['best_match']}\t{value['qval']}\n")
        print("Saved motif mapping to:", mapping_file)
    else:
        print("No motifs.html found; using raw motif identifiers.")
    
    # Output directory for dendrogram figures.
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and print seqlet positions (for reference)
    positions = extract_seqlet_positions(modisco_h5)
    print("Seqlet positions extracted:")
    for pattern_type, patterns in positions.items():
        print(f"\nPattern type: {pattern_type}")
        for pattern_name, seqlet_info in patterns.items():
            print(f"  Pattern: {pattern_name} has {len(seqlet_info['start'])} seqlets")
    
    # Iterate through each pattern in both pos_patterns and neg_patterns
    for pattern_type in positions:
        for pattern_name in positions[pattern_type]:
            try:
                # Extract sequence data for the current pattern
                raw_motif_name, seqlet_data = extract_seqlet_data(modisco_h5, pattern_type=pattern_type, pattern_name=pattern_name)
                # Create the modisco key for mapping lookup.
                modisco_key = f"{pattern_type}/{pattern_name}"
                # If mapping exists, update motif name and get qval0
                if mapping and modisco_key in mapping:
                    motif_name = mapping[modisco_key]['best_match']
                    qval0 = mapping[modisco_key]['qval']
                else:
                    motif_name = raw_motif_name
                    qval0 = None
                print(f"\nExtracted seqlet sequence data for motif: {motif_name} in {pattern_type}")
                print("Seqlet data shape:", seqlet_data.shape)
                
                # Check if there are enough seqlets to perform clustering (need at least 2)
                if seqlet_data.shape[0] < 2:
                    print(f"Not enough seqlets in pattern {pattern_name} ({pattern_type}) to perform clustering.")
                    continue
                
                # Perform hierarchical clustering on the seqlet data
                Z = perform_hierarchical_clustering(seqlet_data)
                
                # Construct output file name for the dendrogram
                output_fig = os.path.join(output_dir, f"dendrogram_{pattern_type}_{pattern_name}.png")
                
                # Plot and save the dendrogram including the qval0 in the title if available.
                n_seqlets = seqlet_data.shape[0]
                if n_seqlets> 1000:
                    continue
                plot_dendrogram(Z, motif_name, n_seqlets, output_fig, qval0=qval0)
            except Exception as e:
                print(f"Error processing pattern {pattern_name} in {pattern_type}: {e}")
