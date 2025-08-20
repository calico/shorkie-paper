#!/usr/bin/env python3
"""
Script to plot a DNA logo for a specified genomic region,
and overlay TF‑MoDISco motif (seqlet) hits with annotations.

Usage:
    python plot_dna_logo_with_motifs.py \
         --modisco_h5 modisco_results.h5 \
         --bed_files bed1.bed,bed2.bed,... \
         --fasta reference.fa \
         --region "chrX:100000-100500" \
         --out_png output.png

Optional arguments (for future extension):
    --predictions_file predictions.npz

Notes:
  - The modisco HDF5 file is expected to contain groups “pos_patterns” and/or “neg_patterns”
    with motif “seqlets” (see TF‑MoDISco documentation).
  - The BED files provide the mapping from each sequence (example) to its genomic coordinates.
  - The FASTA file should be an indexed reference genome (index created by samtools faidx).
  - The region argument should be of the form: chrName:start-end
"""

from optparse import OptionParser

import os
import re
import pysam
import pyranges as pr
import numpy as np
import pandas as pd
import sys
import h5py
from Bio import motifs
from modisco.visualization import viz_sequence
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle

#########################################
# FUNCTIONS TO MAP MODISCO MOTIFS        #
#########################################

def read_bed_files(bed_files):
    """
    Reads one or more BED files and returns a list of entries.
    Each entry is a tuple:
         (chrom, start, end, label, identifier)
    """
    bed_entries = []
    for bed_file in bed_files:
        with open(bed_file, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                fields = line.strip().split()
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                label = fields[3] if len(fields) > 3 else ""
                identifier = fields[4] if len(fields) > 4 else ""
                bed_entries.append((chrom, start, end, label, identifier))
    return bed_entries

def extract_seqlet_positions(h5_filepath):
    """
    Opens the TF-MoDISco HDF5 file and extracts seqlet positional data.
    Returns a nested dictionary:
       { pattern_type: { pattern_name: list_of_seqlets } }
    Each seqlet is a dict with keys: 'example_idx', 'start', 'end', 'is_revcomp'
    """
    results = {}
    with h5py.File(h5_filepath, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type in f:
                results[pattern_type] = {}
                for pattern_name in f[pattern_type].keys():
                    pattern_group = f[pattern_type][pattern_name]
                    if "seqlets" in pattern_group:
                        seqlets_group = pattern_group["seqlets"]
                        starts = seqlets_group["start"][:]
                        ends = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        
                        seqlets_list = []
                        for i in range(len(starts)):
                            seqlet = {
                                "example_idx": int(example_idxs[i]),
                                "start": int(starts[i]),
                                "end": int(ends[i]),
                                "is_revcomp": bool(revcomps[i])
                            }
                            seqlets_list.append(seqlet)
                        results[pattern_type][pattern_name] = seqlets_list
    return results

def map_seqlet_to_genome(seqlet, bed_entries):
    """
    Given a seqlet (with keys: example_idx, start, end, is_revcomp) and the bed_entries list,
    compute the genomic coordinates.
    
    The seqlet’s position is assumed to be relative to the extracted BED region:
         genome_coordinate = BED_start + seqlet_offset.
         
    Returns:
         (chrom, genome_start, genome_end, strand)
    """
    ex_idx = seqlet["example_idx"]
    try:
        chrom, bed_start, bed_end, label, identifier = bed_entries[ex_idx]
    except IndexError:
        raise ValueError(f"Example index {ex_idx} not found in BED entries.")
    
    genome_start = bed_start + seqlet["start"]
    genome_end   = bed_start + seqlet["end"]
    strand = "-" if seqlet["is_revcomp"] else "+"
    return chrom, genome_start, genome_end, strand

def map_all_seqlets(h5_filepath, bed_files):
    """
    Reads the BED files and the modisco HDF5 file, maps all seqlets to genomic coordinates,
    and returns a list of hits. Each hit is a dict with keys:
      - pattern_type, pattern_name, seqlet_index, example_idx, chrom,
        genome_start, genome_end, strand
    """
    bed_entries = read_bed_files(bed_files)
    print(f"Loaded {len(bed_entries)} BED entries from {len(bed_files)} file(s).")
    seqlet_results = extract_seqlet_positions(h5_filepath)
    
    hits = []
    for pattern_type, patterns in seqlet_results.items():
        for pattern_name, seqlets in patterns.items():
            for i, seqlet in enumerate(seqlets):
                chrom, genome_start, genome_end, strand = map_seqlet_to_genome(seqlet, bed_entries)
                hit = {
                    "pattern_type": pattern_type,
                    "pattern_name": pattern_name,
                    "seqlet_index": i,
                    "example_idx": seqlet["example_idx"],
                    "chrom": chrom,
                    "genome_start": genome_start,
                    "genome_end": genome_end,
                    "strand": strand
                }
                hits.append(hit)
    return hits

def convert_hits_to_dataframe(hits):
    """
    Converts the list of hit dictionaries to a Pandas DataFrame with columns:
       'sequence_name', 'start', 'stop', 'source_label'
    where 'sequence_name' comes from the chromosome and 'source_label' combines
    pattern_type and pattern_name.
    """
    records = []
    for hit in hits:
        record = {
            "sequence_name": hit["chrom"],
            "start": hit["genome_start"],
            "stop": hit["genome_end"],
            "source_label": f"{hit['pattern_type']}/{hit['pattern_name']}"
        }
        records.append(record)
    return pd.DataFrame(records)



#Helper function to plot ACGT letters at a given position
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
                "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
                "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
                "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
                "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
                "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
                "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
                "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
                ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
    COLOR_SCHEME = {'G': 'orange',#'orange', 
                    'A': 'green',#'red', 
                    'C': 'blue',#'blue', 
                    'T': 'red',#'darkgreen',
                    'UP': 'green', 
                    'DN': 'red',
                    '(': 'black',
                    '.': 'black', 
                    ')': 'black'}


    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]
    if color is not None :
        chosen_color = color

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


#Plot pair of sequence logos with matched y-axis height
def visualize_input_logo(att_grad_wt, title, fig_name='', figsize=(12, 3)):
    #Get logo bounds
    y_min = np.min(att_grad_wt)
    y_max = np.max(att_grad_wt)
    y_max_abs = max(np.abs(y_min), np.abs(y_max))

    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs
    
    print("y_min = " + str(round(y_min, 8)))
    print("y_max = " + str(round(y_max, 8)))

    #Plot wt logo
    print("--- WT ---")
    fig, ax = plot_seq_scores(
        att_grad_wt, title=title, y_min=y_min, y_max=y_max,
        figsize=figsize,
        plot_y_ticks=False,
        fig_name=fig_name + '_wt',
    )
    return fig, ax


#Function to plot sequence logo
def plot_seq_scores(importance_scores, title, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, fig_name="default") :

    # Internally, we transpose so that we work with shape (4, L)
    scores = importance_scores.T.copy()  # now shape is (4, L)
    num_positions = scores.shape[1]

    fig, ax = plt.subplots(figsize=figsize, facecolor='none')
    # For each position, stack the letters by their magnitude.
    # logo_height = 1.0
    # height_base = (1.0 - logo_height) / 2.0  # Typically (1 - 1) / 2 = 0

    for j in range(num_positions):
        # Get the scores for the 4 letters at position j
        col = scores[:, j]
        # Order letters by absolute value (largest first)
        sort_index = np.argsort(-np.abs(col))
        pos_height = 0.0
        neg_height = 0.0
        for ii in range(4):
            i = sort_index[ii]
            # Map index to nucleotide (order: 0:A, 1:C, 2:G, 3:T)
            if i == 0:
                nt = 'A'
            elif i == 1:
                nt = 'C'
            elif i == 2:
                nt = 'G'
            elif i == 3:
                nt = 'T'
            nt_score = col[i]
            if nt_score >= 0:
                dna_letter_at(nt, j + 0.5, pos_height, yscale=nt_score, ax=ax)
                pos_height += nt_score
            else:
                dna_letter_at(nt, j + 0.5, neg_height, yscale=nt_score, ax=ax)
                neg_height += nt_score
        

    if plot_y_ticks:
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
    else:
        ax.set_yticks([])

    # Set axis limits
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_min is not None:
        ax.ylset_ylimim(y_min)
    else:
        ax.set_ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
        )
    # ax.set_xticks(np.arange(num_positions+1))
    ax.set_xlabel("Position")
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=1)
    return fig, ax


def parse_fasta_header(header):
    match = re.match(r'>(chr[\w]+):(\d+)-(\d+)\|(\w+)', header)
    return match.groups() if match else (None, None, None, None)

def process_fasta(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    chromosome, start, end, species = parse_fasta_header(line.strip())
                    if chromosome:
                        data.append([chromosome, int(start), int(end), species])
        return pd.DataFrame(data, columns=['Chromosome', 'Start', 'End', 'species'])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame(columns=['Chromosome', 'Start', 'End', 'species'])

def process_single_model(predictions_file):
    # File paths
    x_true_list = []
    x_pred_list = []
    label_list = []
    weight_scale_list = []

    print(f"\t>> predictions_file: {predictions_file}")
    # Load predictions and targets
    cache_bundle = np.load(predictions_file)
    x_true_list.append(cache_bundle['x_true'])
    x_pred_list.append(cache_bundle['x_pred'])
    label_list.append(cache_bundle['label'])
    weight_scale_list.append(cache_bundle['weight_scale'])


    x_true = np.concatenate(x_true_list, axis=0)
    x_pred = np.concatenate(x_pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    weight_scale = np.concatenate(weight_scale_list, axis=0)

    return x_true, x_pred, label, weight_scale

def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option("--modisco_h5", dest='modisco_h5',
                        help="Path to TF-MoDISco HDF5 results file.")
    parser.add_option('--out_dir', dest='out_dir', default='', type='str', help='Output file path [Default: %default]')
    parser.add_option('--seq_bed', dest='seq_bed', default=None, type='str', help='BED file of sequences')
    parser.add_option('--predictions_file', dest='predictions_file', default=None, type='str', help='MLM prediction output file')
   
    (options,args) = parser.parse_args()
    print("options.bed: ", options.seq_bed)
    print("options.predictions_file: ", options.predictions_file)
    
    x_true_all = []
    x_pred_all = []
    label_all = []
    weight_scale_all = []

    x_true, x_pred, label, weight_scale = process_single_model(options.predictions_file)

    print("Before x_true: ", x_true.shape)
    print("Before x_pred: ", x_pred.shape)
    print("label: ", label.shape)
    print("weight_scale: ", weight_scale.shape)

    # Now apply the transformations after averaging
    x_pred_transformed = np.copy(x_pred)
    x_pred_transformed += 0.0001  
    x_pred_transformed_mean = np.mean(x_pred_transformed, axis=1, keepdims=True)
    x_pred_transformed = x_pred_transformed * np.log(x_pred_transformed / x_pred_transformed_mean)
    x_pred = x_pred_transformed


    bed_files = [
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_train_r64.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/sequences_test.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_valid.cleaned.bed"
    ]

    # Load sequences BED file
    hits = map_all_seqlets(options.modisco_h5, bed_files)
    modisco_df = convert_hits_to_dataframe(hits)
    print(f"Total modisco seqlet hits mapped: {len(modisco_df)}")

    # Load sequences BED file
    seqs_df = pd.read_csv(options.seq_bed, sep='\t', names=['contig', 'start', 'end', 'gene', 'species', 'strand'])
    print("seqs_df: ", seqs_df)
    print("seqs_df len : ", len(seqs_df))
    upstream = 450
    downstream = 50    
    for idx, row in seqs_df.iterrows():
        # Parse BED fields:
        #   chrom, start, end, name, score, strand
        chrom  = row['contig']
        start  = int(row['start'])
        end    = int(row['end'])
        name   = row['gene']
        score  = row['species']
        strand = row['strand']
        
        tss = (start + end) // 2
        if strand == '+':
            # TSS is 'start'
            region_start = tss - upstream
            region_end   = tss + downstream
        else:
            # TSS is 'end'
            region_start = tss - downstream
            region_end   = tss + upstream
        
        # Clip at zero if needed
        if region_start < 0:
            region_start = 0
    
        # print(f"16k            : {start} - {end}")
        # print(f"selected region: {region_start} - {region_end}\n")
        
        x_true_one = x_true[idx][region_start-start:region_end-start, :]
        x_pred_one = x_pred[idx][region_start-start:region_end-start, :]

        x_ref_pred_one = x_true_one * x_pred_one

        
        # Filter modisco hits that overlap the selected region.
        # (Assuming modisco_df['sequence_name'] uses the same chromosome labels as in the region.)

        region_hits = modisco_df[
            (modisco_df["sequence_name"] == chrom) &
            (modisco_df["stop"] > region_start) &
            (modisco_df["start"] < region_end)
        ]

        if len(region_hits) > 0:
            print(f"Found {len(region_hits)} modisco hits overlapping the region.")
            print(f"Region: {chrom}:{region_start}-{region_end}")
        
            # For each modisco hit overlapping the region, add a highlight box and annotation.
            color_options = ["red"]#, "purple", "orange", "cyan", "magenta", "yellow", "black"]


            region_length = region_end - region_start

            fig, ax = visualize_input_logo(x_pred_one, title=f"DNA Logo: {chrom}:{region_start}-{region_end}", figsize=(100, 3))
            
            for idx, row in region_hits.iterrows():
                # Compute the relative positions (in the logo, position 0 corresponds to region_start).
                rel_start = max(row["start"], region_start) - region_start
                rel_end = min(row["stop"], region_end) - region_start
                width = rel_end - rel_start
                if width <= 0:
                    continue  # Skip if no overlap.
                # Choose a color based on the source label.
                color = color_options[hash(row["source_label"]) % len(color_options)]
                # Add a rectangle (using a dashed outline) spanning the full vertical extent of the logo.
                rect = Rectangle((rel_start, 0), width, 1, linewidth=2,
                                edgecolor=color, facecolor="none", linestyle="--")
                ax.add_patch(rect)
                # Annotate with motif name above the logo.
                ax.text(rel_start + width/2, 1.05, row["source_label"],
                        color=color, ha='center', va='bottom', fontsize=10, rotation=0)
            
            ax.set_xlim(0, region_length)
            # Optionally remove y-axis ticks and labels.
            ax.set_yticks([])
            ax.set_facecolor('none') 
            plt.tight_layout()
            plt.savefig(f"{options.out_dir}/logo_{chrom}_{region_start}_{region_end}_{idx}_{name}_{strand}.png", dpi=300, transparent=True)
            print(f"Saved logo plot with motif annotations to {options.out_dir}/logo_{idx}_{name}_{strand}_{chrom}_{region_start}_{region_end}.png")
            plt.show()
            plt.clf()
            break

if __name__ == "__main__":
    main()