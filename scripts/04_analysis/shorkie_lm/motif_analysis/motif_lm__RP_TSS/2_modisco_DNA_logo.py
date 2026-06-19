#!/usr/bin/env python3
"""
Script to plot DNA logos for genomic regions defined in a BED file,
and overlay TF‑MoDISco motif (seqlet) hits with annotations.
In this version, each seqlet’s per-hit confidence is computed directly
from the contribution scores stored in the TF‑MoDISco HDF5 file.
Seqlets with confidence below a user-specified threshold are skipped during visualization.

Usage example:
    python plot_dna_logo_with_motifs.py \
         --modisco_h5 modisco_results.h5 \
         --seq_bed sequences.bed \
         --predictions_file predictions.npz \
         --out_dir output_directory \
         --model_arch MyModel \
         --motifs_html motifs.html \
         --confidence_threshold 0.5 \
         --trim_min_length 3

Notes:
  - The modisco HDF5 file is expected to contain groups “pos_patterns” and/or “neg_patterns”
    with motif “seqlets.”
  - The BED file is assumed to match the input sequences used for training/inference.
  - The predictions_file is assumed to contain arrays: x_true, x_pred, label, weight_scale.
  - The motifs_html option is retained for motif mapping if desired.
  - The new option --confidence_threshold specifies the minimum computed
    per-seqlet confidence to annotate the hit.
"""

import os
import sys
import tempfile
import shutil
from optparse import OptionParser

import numpy as np
import pandas as pd
import h5py
from Bio import motifs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties

########################################
# SEQUENCE & TRIMMING FUNCTIONS
########################################

def one_hot_to_sequence(x_one_hot):
    """
    Convert a one-hot encoded sequence (shape: L x 4, order: [A, C, G, T])
    to its string representation.
    """
    idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_chars = [idx_to_nt[np.argmax(x_one_hot[i, :])] for i in range(x_one_hot.shape[0])]
    return "".join(seq_chars)


def compute_trimming_offsets(modisco_h5, trim_threshold, pad=4):
    """
    For each pattern in the modisco HDF5 file, compute trimmed boundaries
    based on the summed absolute contribution scores.
    Returns a dictionary mapping pattern_id -> {
        'fwd_start', 'fwd_end', 'rev_start', 'rev_end', 'length'
    }.
    """
    offsets_dict = {}
    with h5py.File(modisco_h5, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type not in f:
                continue
            for pattern_name in f[pattern_type]:
                pattern = f[pattern_type][pattern_name]
                if "contrib_scores" not in pattern:
                    continue
                cwm = np.array(pattern["contrib_scores"][:])
                L = cwm.shape[0]
                # Forward trimming.
                score_fwd = np.sum(np.abs(cwm), axis=1)
                if np.max(score_fwd) == 0:
                    fwd_start, fwd_end = 0, L
                else:
                    trim_thresh_fwd = np.max(score_fwd) * trim_threshold
                    pass_inds = np.where(score_fwd >= trim_thresh_fwd)[0]
                    if len(pass_inds) == 0:
                        fwd_start, fwd_end = 0, L
                    else:
                        fwd_start = max(np.min(pass_inds) - pad, 0)
                        fwd_end   = min(np.max(pass_inds) + pad + 1, L)
                # Reverse trimming.
                cwm_rev = cwm[::-1, ::-1]
                score_rev = np.sum(np.abs(cwm_rev), axis=1)
                if np.max(score_rev) == 0:
                    rev_start_temp, rev_end_temp = 0, L
                else:
                    trim_thresh_rev = np.max(score_rev) * trim_threshold
                    pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]
                    if len(pass_inds_rev) == 0:
                        rev_start_temp, rev_end_temp = 0, L
                    else:
                        rev_start_temp = max(np.min(pass_inds_rev) - pad, 0)
                        rev_end_temp   = min(np.max(pass_inds_rev) + pad + 1, L)
                rev_trim_start = L - rev_end_temp
                rev_trim_end   = L - rev_start_temp

                pattern_id = f"{pattern_type}/{pattern_name}"
                offsets_dict[pattern_id] = {
                    'fwd_start': fwd_start,
                    'fwd_end': fwd_end,
                    'rev_start': rev_trim_start,
                    'rev_end': rev_trim_end,
                    'length': L
                }
    return offsets_dict


def get_trimmed_genomic_coords(row, trim_offsets):
    """
    Given a modisco hit (row from modisco_df) and the trimming offsets,
    return (trimmed_start, trimmed_stop) in genomic coordinates.
    """
    pattern_id = row["source_label"]  # e.g., "pos_patterns/pattern_0"
    offsets = trim_offsets.get(pattern_id, None)
    if offsets is None:
        return row["start"], row["stop"]

    L = offsets['length']
    if row["strand"] == '+':
        trimmed_start = row["start"] + offsets['fwd_start']
        trimmed_stop  = row["start"] + offsets['fwd_end']
    else:
        trimmed_start = row["start"] + (L - offsets['rev_end'])
        trimmed_stop  = row["start"] + (L - offsets['rev_start'])
    return trimmed_start, trimmed_stop


########################################
# NEW FUNCTION: Compute Seqlet Confidence
########################################

def compute_seqlet_confidence(contrib_scores, trim_start, trim_end):
    """
    Given the contribution score array (shape: L x 4) for a pattern and
    trimmed boundaries (trim_start, trim_end), compute a confidence score
    for a seqlet. Here, we use the sum of absolute contribution scores
    over the trimmed region as a proxy for confidence.
    
    Parameters:
      contrib_scores: numpy array of shape (L, 4)
      trim_start: integer start index after trimming
      trim_end: integer end index after trimming
      
    Returns:
      confidence: a float representing the confidence score.
    """
    trimmed_scores = contrib_scores[trim_start:trim_end, :]
    confidence = np.sum(np.abs(trimmed_scores))
    return confidence


########################################
# FUNCTIONS FOR BED & SEQLET MAPPING
########################################

def read_bed_files(bed_files):
    """
    Read one or more BED files.
    Each line is assumed to have at least 3 fields: chrom, start, end.
    Returns a list of tuples: (chrom, start, end, label, identifier).
    """
    bed_entries = []
    for bed_file in bed_files:
        with open(bed_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                chrom = fields[0]
                start = int(fields[1])
                end   = int(fields[2])
                label = fields[3] if len(fields) > 3 else ""
                identifier = fields[4] if len(fields) > 4 else ""
                bed_entries.append((chrom, start, end, label, identifier))
    return bed_entries


def extract_seqlet_positions(h5_filepath):
    """
    Extract seqlet positional data from the TF-MoDISco HDF5 file.
    Returns a dictionary: { pattern_type: { pattern_name: list_of_seqlets } }.
    Each seqlet is a dict with keys: 'example_idx', 'start', 'end', 'is_revcomp'.
    """
    results = {}
    with h5py.File(h5_filepath, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type in f:
                results[pattern_type] = {}
                for pattern_name in f[pattern_type]:
                    pattern_group = f[pattern_type][pattern_name]
                    if "seqlets" in pattern_group:
                        seqlets_group = pattern_group["seqlets"]
                        starts = seqlets_group["start"][:]
                        ends   = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        
                        seqlets_list = []
                        for i in range(len(starts)):
                            seqlets_list.append({
                                "example_idx": int(example_idxs[i]),
                                "start": int(starts[i]),
                                "end": int(ends[i]),
                                "is_revcomp": bool(revcomps[i])
                            })
                        results[pattern_type][pattern_name] = seqlets_list
    return results


def map_seqlet_to_genome(seqlet, bed_entries):
    """
    Map a seqlet (with keys: example_idx, start, end, is_revcomp) to genomic coordinates.
    Returns (chrom, genome_start, genome_end, strand).
    """
    ex_idx = seqlet["example_idx"]
    try:
        chrom, bed_start, bed_end, label, identifier = bed_entries[ex_idx]
    except IndexError:
        raise ValueError(f"Example index {ex_idx} out of range in bed_entries.")
    
    genome_start = bed_start + seqlet["start"]
    genome_end   = bed_start + seqlet["end"]
    strand = "-" if seqlet["is_revcomp"] else "+"
    return chrom, genome_start, genome_end, strand


def map_all_seqlets(h5_filepath, bed_files):
    """
    Read BED files, extract seqlets from the modisco HDF5 file,
    and return a list of dictionaries with mapped coordinates.
    """
    bed_entries = read_bed_files(bed_files)
    print(f"Loaded {len(bed_entries)} BED entries from {len(bed_files)} file(s).")
    seqlet_results = extract_seqlet_positions(h5_filepath)
    hits = []
    max_example_idx = 0
    for pattern_type, patterns in seqlet_results.items():
        for pattern_name, seqlets in patterns.items():
            for i, seqlet in enumerate(seqlets):
                ex_idx = seqlet["example_idx"]
                if ex_idx > max_example_idx:
                    max_example_idx = ex_idx
                chrom, gstart, gend, strand = map_seqlet_to_genome(seqlet, bed_entries)
                hits.append({
                    "pattern_type": pattern_type,
                    "pattern_name": pattern_name,
                    "seqlet_index": i,
                    "example_idx": ex_idx,
                    "chrom": chrom,
                    "genome_start": gstart,
                    "genome_end": gend,
                    "strand": strand
                })
    print(">> max_example_idx in BED:", max_example_idx)
    return hits


def convert_hits_to_dataframe(hits):
    """
    Convert a list of hit dictionaries to a pandas DataFrame with columns:
      sequence_name, start, stop, source_label, strand.
    """
    records = []
    for hit in hits:
        records.append({
            "sequence_name": hit["chrom"],
            "start": hit["genome_start"],
            "stop": hit["genome_end"],
            "strand": hit["strand"],
            "source_label": f"{hit['pattern_type']}/{hit['pattern_name']}",
        })
    return pd.DataFrame(records)


########################################
# FUNCTIONS FOR LOGO PLOTTING
########################################

def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    """
    Draw a DNA letter at position (x, y) with scaling.
    """
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
    }
    COLOR_SCHEME = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}

    path = LETTERS[letter]
    chosen_color = COLOR_SCHEME.get(letter, "black")
    if color is not None:
        chosen_color = color

    t = (mpl.transforms.Affine2D()
         .scale(1 * globscale, yscale * globscale)
         .translate(x, y)
         + ax.transData)
    # Pass the TextPath as a positional argument (as "path").
    p = PathPatch(path, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)


def plot_seq_scores(importance_scores, title,
                    figsize=(16, 2),
                    plot_y_ticks=True,
                    y_min=None, y_max=None,
                    fig_name="default"):
    """
    Plot a letter-based representation (4 x L) of importance_scores
    where importance_scores.shape = (L, 4).
    """
    scores = importance_scores.T.copy()  # shape: (4, L)
    num_positions = scores.shape[1]

    fig, ax = plt.subplots(figsize=figsize, facecolor='none')

    for j in range(num_positions):
        col = scores[:, j]
        sort_index = np.argsort(-np.abs(col))
        pos_height = 0.0
        neg_height = 0.0
        for ii in sort_index:
            nt_score = col[ii]
            nt = 'A' if ii == 0 else 'C' if ii == 1 else 'G' if ii == 2 else 'T'
            if nt_score >= 0:
                dna_letter_at(nt, j + 0.5, pos_height, yscale=nt_score, ax=ax)
                pos_height += nt_score
            else:
                dna_letter_at(nt, j + 0.5, neg_height, yscale=nt_score, ax=ax)
                neg_height += nt_score

    if not plot_y_ticks:
        ax.set_yticks([])

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_min is not None:
        ax.set_ylim(bottom=y_min)
    else:
        data_min = np.min(importance_scores)
        data_max = np.max(importance_scores)
        margin = 0.1 * max(abs(data_min), abs(data_max))
        ax.set_ylim(data_min - margin, data_max + margin)

    ax.set_xlim(0, num_positions)
    ax.set_xlabel("Position")
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=1)
    return fig, ax


def visualize_input_logo(att_grad_wt, title, fig_name='', figsize=(12, 3)):
    """
    Generate a DNA logo from importance scores.
    """
    y_min = np.min(att_grad_wt)
    y_max = np.max(att_grad_wt)
    y_max_abs = max(abs(y_min), abs(y_max))
    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs

    fig, ax = plot_seq_scores(
        att_grad_wt,
        title=title,
        y_min=y_min,
        y_max=y_max,
        figsize=figsize,
        plot_y_ticks=False,
        fig_name=fig_name
    )
    return fig, ax


########################################
# FUNCTIONS FOR MODEL OUTPUTS
########################################

def process_single_model(predictions_file):
    """
    Load predictions.npz containing arrays: x_true, x_pred, label, weight_scale.
    Returns a tuple (x_true, x_pred, label, weight_scale).
    """
    print(f"Loading predictions file: {predictions_file}")
    cache_bundle = np.load(predictions_file)
    x_true = cache_bundle['x_true']
    x_pred = cache_bundle['x_pred']
    label  = cache_bundle['label']
    weight_scale = cache_bundle['weight_scale']
    return x_true, x_pred, label, weight_scale


########################################
# (Optional) PARSING MOTIFS.HTML FOR MOTIF MAPPING
########################################

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


########################################
# MAIN FUNCTION
########################################

def main():
    parser = OptionParser()
    parser.add_option("--modisco_h5", dest="modisco_h5",
                      help="Path to TF-MoDISco HDF5 results file.")
    parser.add_option("--model_arch", dest="model_arch", default="default_arch", type="str",
                      help="Model architecture")
    parser.add_option("--trim_threshold", dest="trim_threshold", default=0.3, type="float",
                      help="Threshold for trimming seqlets")
    parser.add_option("--out_dir", dest="out_dir", default="", type="str",
                      help="Directory to save output logos")
    parser.add_option("--seq_bed", dest="seq_bed", default=None, type="str",
                      help="BED file of sequences (one line per example)")
    parser.add_option("--predictions_file", dest="predictions_file", default=None, type="str",
                      help="ML model prediction output file")
    parser.add_option("--no_motif_annotation", dest="no_motif_annotation", action="store_true", default=False,
                      help="If set, plot DNA logo without motif annotations.")
    parser.add_option("--motifs_html", dest="motifs_html", default=None, type="str",
                      help="Path to motifs.html file from modisco report output")
    parser.add_option("--qval_threshold", dest="qval_threshold", default=1, type="float",
                      help="Q-value threshold for motif mapping (if using motifs_html)")
    parser.add_option("--confidence_threshold", dest="confidence_threshold", default=0.0, type="float",
                      help="Minimum per-seqlet confidence (computed from contribution scores) to include hit in visualization")
    parser.add_option("--trim_min_length", dest="trim_min_length", default=3, type="int",
                      help="Minimum length after trimming [default: %default]")
    
    (options, args) = parser.parse_args()
    
    print("Options:")
    print("  seq_bed:", options.seq_bed)
    print("  predictions_file:", options.predictions_file)
    print("  no_motif_annotation:", options.no_motif_annotation)
    print("  motifs_html:", options.motifs_html)
    print("  qval_threshold:", options.qval_threshold)
    print("  confidence_threshold:", options.confidence_threshold)
    
    # Determine model architecture output directory.
    if options.no_motif_annotation:
        model_dir = os.path.join(options.out_dir, "motifs_no_annotation", options.model_arch)
    else:
        model_dir = os.path.join(options.out_dir, "motifs_with_annotation", options.model_arch)
    os.makedirs(model_dir, exist_ok=True)
    if options.no_motif_annotation:
        os.makedirs(os.path.join(model_dir, "dna_logo"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "dna_logo_true"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "dna_logo_ref_pred"), exist_ok=True)
    
    # Load model predictions.
    x_true, x_pred, label, weight_scale = process_single_model(options.predictions_file)
    print("Shapes:")
    print("  x_true:", x_true.shape)
    print("  x_pred:", x_pred.shape)
    print("  label:", label.shape)
    
    # Optional transformation of x_pred.
    x_pred = x_pred + 0.0001
    mean_pred = np.mean(x_pred, axis=1, keepdims=True)
    x_pred = x_pred * np.log(x_pred / mean_pred)
    
    # Read BED file.
    seqs_df = pd.read_csv(options.seq_bed, sep='\t',
                          names=['contig','start','end','gene','species','strand'])
    print("Loaded BED file with", len(seqs_df), "entries.")
    
    upstream = 450
    downstream = 50
    
    # Parse motifs.html for motif mapping, if provided.
    motif_mapping = {}
    if options.motifs_html:
        motif_mapping = parse_motifs_html(options.motifs_html, options.qval_threshold)
        mapping_file = os.path.join(model_dir, "motif_mapping.tsv")
        with open(mapping_file, "w") as f:
            f.write("modisco_pattern\tbest_match\tqval0\n")
            for k, v in motif_mapping.items():
                f.write(f"{k}\t{v['best_match']}\t{v['qval']}\n")
        print("Saved motif mapping to", mapping_file)
    else:
        print("No motifs_html provided; using original motif labels.")
    
    # Compute trimming offsets and map seqlets.
    if options.modisco_h5 is not None:
        trim_offsets = compute_trimming_offsets(options.modisco_h5, options.trim_threshold, pad=4)
        bed_files = [options.seq_bed]
        hits = map_all_seqlets(options.modisco_h5, bed_files)
        modisco_df = convert_hits_to_dataframe(hits)
        print("Total seqlet hits:", len(modisco_df))
    
    summary_rows = []
    
    # Process each region defined in the BED file.
    for idx, row in seqs_df.iterrows():
        chrom = row['contig']
        start = int(row['start'])
        end   = int(row['end'])
        strand = row['strand']
        gene = row['gene']

        if gene not in ['GLK1', 'POP4', 'FUN12', 'FLC2', 'PWP1', 'KRE33', 'RRP12', 'PIS1']:
            continue
        print(f"Processing region idx={idx}: {chrom}:{start}-{end} ({gene}, strand={strand})")
        
        tss = (start + end) // 2
        if strand == '+':
            region_start = tss - upstream
            region_end = tss + downstream
        else:
            region_start = tss - downstream
            region_end = tss + upstream
        if region_start < 0:
            region_start = 0
        print(f"Region boundaries: {region_start} - {region_end}")
        
        # Slice the prediction arrays.
        x_true_one = x_true[idx][region_start - start : region_end - start, :]
        x_pred_one = x_pred[idx][region_start - start : region_end - start, :]
        x_ref_pred_one = x_true_one * x_pred_one
        region_length = upstream + downstream
        
        dna_seq = one_hot_to_sequence(x_true[idx])
        
        if options.no_motif_annotation:
            fig, ax = visualize_input_logo(x_pred_one,
                                           title=f"DNA Logo (no annotation): {chrom}:{region_start}-{region_end} (idx={idx})",
                                           figsize=(100, 3))
            ax.set_xlim(0, region_length)
            plt.tight_layout()
            out_filename = os.path.join(model_dir, "dna_logo",
                                        f"{chrom}_{region_start}_{region_end}_idx{idx}_{gene}_{strand}.png")
            plt.savefig(out_filename, dpi=300, transparent=True)
            print("Saved logo:", out_filename)
            plt.close(fig)
            continue
        
        
        # For motif annotation mode: filter modisco seqlet hits overlapping the region.
        region_hits = modisco_df[
            (modisco_df["sequence_name"] == chrom) &
            (modisco_df["stop"] > region_start) &
            (modisco_df["start"] < region_end)
        ]
        print(f"Region hits for idx={idx}: {len(region_hits)}")
        if len(region_hits) == 0:
            continue
        
        fig, ax = visualize_input_logo(x_pred_one,
                                       title=f"DNA Logo: {chrom}:{region_start}-{region_end} (idx={idx})",
                                       figsize=(100, 3))
        
        # Remove overlapping text detection: assign all annotation text to row 0.
        annotation_info = []
        
        region_hits_sorted = region_hits.sort_values(by="start")
        for _, hitrow in region_hits_sorted.iterrows():
            # Trim coordinates based on contribution scores.
            t_start, t_stop = get_trimmed_genomic_coords(hitrow, trim_offsets)
            # Compute per-hit confidence from the pattern's contribution scores.
            pattern_id = hitrow["source_label"]
            group, pname = pattern_id.split("/")
            with h5py.File(options.modisco_h5, 'r') as f:
                if group in f and pname in f[group]:
                    pattern_group = f[group][pname]
                    if "contrib_scores" in pattern_group:
                        cs = np.array(pattern_group["contrib_scores"][:])
                    else:
                        cs = None
                else:
                    cs = None
            if cs is None:
                confidence = 0.0
            else:
                offsets = trim_offsets.get(pattern_id, None)
                if offsets is None:
                    confidence = np.sum(np.abs(cs))
                else:
                    confidence = compute_seqlet_confidence(cs, offsets['fwd_start'], offsets['fwd_end'])
            
            print(f"Hit at {t_start}-{t_stop} (confidence: {confidence:.3f})")
            if confidence < options.confidence_threshold:
                continue
            
            # Convert trimmed genomic coordinates to local coordinates relative to region_start.
            rel_start = max(t_start, region_start) - region_start
            rel_end = min(t_stop, region_end) - region_start
            width = rel_end - rel_start
            if width <= 0:
                continue
            
            # Use motif mapping if available.
            original_label = hitrow["source_label"]
            known_motif = original_label
            mapping_entry = motif_mapping.get(original_label, None)
            if mapping_entry is not None:
                if mapping_entry['qval'] is not None and mapping_entry['qval'] <= options.qval_threshold:
                    known_motif = mapping_entry['best_match']
            
            assigned_row = 0  # All annotations in the same row.
            annotation_info.append((rel_start, width, assigned_row, known_motif, "red"))
            
            summary_rows.append({
                "region_idx": idx,
                "gene": gene,
                "chrom": chrom,
                "region_start": region_start,
                "region_end": region_end,
                "motif_original": original_label,
                "known_motif": known_motif,
                "motif_genome_start": t_start,
                "motif_genome_end": t_stop,
                "motif_strand": hitrow["strand"],
                "confidence": confidence
            })
            print(f"\tAnnotated {known_motif} at {rel_start}-{rel_end} (row 0), confidence {confidence:.3f}")
        
        # Draw annotation boxes and text on the logo.
        logo_top = ax.get_ylim()[1]
        base_y = logo_top - 0.02  # All annotations in row 0.
        box_height = 0.5
        for (rel_start, width, assigned_row, known_motif, color) in annotation_info:
            y = base_y  # Fixed row (row 0)
            rect = Rectangle((rel_start, y), width, box_height,
                             linewidth=2, edgecolor=color, facecolor="none", linestyle="--")
            ax.add_patch(rect)
            ax.text(rel_start + width / 2, y + box_height, known_motif,
                    color=color, ha='center', va='bottom', fontsize=9)
            # Optionally, add a highlight rectangle.
            highlight_rect = Rectangle((rel_start, 0), width, 1.8,
                                       linewidth=1.5, edgecolor=color, facecolor="none", linestyle="--")
            ax.add_patch(highlight_rect)
        
        ax.set_xlim(0, region_length)
        plt.tight_layout()
        out_filename = os.path.join(model_dir, f"logo_{chrom}_{region_start}_{region_end}_idx{idx}_{gene}_{strand}.png")
        
        plt.savefig(out_filename, dpi=200, transparent=True)
        print("Saved annotated logo:", out_filename)
        plt.close(fig)
    
    # Write out a TSV summary of motif hits.
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_tsv_path = os.path.join(model_dir, "motif_hits_summary.tsv")
        summary_df.to_csv(summary_tsv_path, sep="\t", index=False)
        print("Saved motif hits summary TSV to", summary_tsv_path)
    else:
        print("No motif hits found to summarize.")


if __name__ == "__main__":
    main()
