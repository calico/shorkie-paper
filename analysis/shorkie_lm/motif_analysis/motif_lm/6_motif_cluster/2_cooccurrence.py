#!/usr/bin/env python3
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import matplotlib.gridspec as gridspec

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

def compute_cooccurrence_matrix(seqlet_positions, normalize=True, mapping=None):
    """
    Computes a co-occurrence matrix for motifs.
    
    For each motif (identified by the combination of pattern type and pattern name),
    the unique example indices where the motif is found are collected.
    
    If normalize is True, the co-occurrence is computed using the Jaccard index:
    
        J(i, j) = |S_i ∩ S_j| / |S_i ∪ S_j|
    
    If a mapping dict is provided, the raw modisco key (pattern_type/pattern_name) is
    replaced by the mapped motif name.
    
    Returns:
        motif_ids (list): List of motif identifiers (mapped if available).
        cooccurrence_matrix (np.ndarray): A square matrix of normalized co-occurrence values.
    """
    motif_ids = []
    motif_example_sets = {}
    
    for pattern_type, patterns in seqlet_positions.items():
        for pattern_name, info in patterns.items():
            # Create the raw modisco key (format: "pattern_type/pattern_name")
            modisco_key = f"{pattern_type}/{pattern_name}"
            if mapping is not None and modisco_key in mapping:
                mapped_name = mapping[modisco_key]['best_match']
            else:
                mapped_name = modisco_key.replace("/", "_")
            motif_ids.append(mapped_name)
            example_set = set(info['example_idx'].tolist())
            motif_example_sets[mapped_name] = example_set

    n = len(motif_ids)
    cooccurrence_matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            set_i = motif_example_sets[motif_ids[i]]
            set_j = motif_example_sets[motif_ids[j]]
            intersection = len(set_i.intersection(set_j))
            if normalize:
                union = len(set_i.union(set_j))
                value = intersection / union if union > 0 else 0.0
            else:
                value = intersection
            cooccurrence_matrix[i, j] = value
    
    return motif_ids, cooccurrence_matrix

def plot_sorted_cooccurrence(motif_ids, matrix, output_file):
    """
    Produces a heatmap of the co-occurrence matrix where rows and columns are
    sorted according to hierarchical clustering (using the Jaccard index distance).
    The phylogenetic tree is not shown.
    """
    # Compute the distance matrix and linkage matrix.
    distance_matrix = 1.0 - matrix
    Z = linkage(distance_matrix, method='average')
    # Get the order from the dendrogram, but do not plot it.
    dendro = dendrogram(Z, labels=motif_ids, no_plot=True)
    order = dendro['leaves']
    
    # Reorder the matrix and motif IDs.
    ordered_matrix = matrix[np.ix_(order, order)]
    ordered_ids = [motif_ids[i] for i in order]
    
    # Create a single plot for the heatmap.
    plt.figure(figsize=(18, 15))
    im = plt.imshow(ordered_matrix, interpolation='nearest', cmap='viridis')
    plt.title("Sorted Normalized Motif Co-occurrence Matrix (Jaccard Index)")
    plt.xlabel("Motifs")
    plt.ylabel("Motifs")
    
    plt.xticks(range(len(ordered_ids)), ordered_ids, rotation=90, fontsize=8)
    plt.yticks(range(len(ordered_ids)), ordered_ids, fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Jaccard index')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved sorted co-occurrence heatmap to: {output_file}")

if __name__ == "__main__":
    # Path to the TF-MoDISco results HDF5 file.
    modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5"
    
    # Path to the motifs.html file for motif mapping.
    motifs_html = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/report_w16384_n100000/motifs.html"
    
    # Output file to write the mapping results.
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
    
    # Create output directory for the figures.
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)
    output_fig = os.path.join(output_dir, "clustered_cooccurrence_matrix.png")
    
    # Extract seqlet positions from the TF-MoDISco file.
    positions = extract_seqlet_positions(modisco_h5)
    print("Seqlet positions extracted:")
    for pattern_type, patterns in positions.items():
        print(f"\nPattern type: {pattern_type}")
        for pattern_name, seqlet_info in patterns.items():
            print(f"  Pattern: {pattern_name} has {len(seqlet_info['start'])} seqlets")
    
    # Compute the normalized co-occurrence matrix using the mapping.
    motif_ids, cooccurrence_matrix = compute_cooccurrence_matrix(positions, normalize=True, mapping=mapping)
    print("\nMotif IDs:", motif_ids)
    print("Normalized co-occurrence matrix shape:", cooccurrence_matrix.shape)
    
    # Visualize and save the clustered co-occurrence matrix (with dendrogram).
    plot_sorted_cooccurrence(motif_ids, cooccurrence_matrix, output_fig)
