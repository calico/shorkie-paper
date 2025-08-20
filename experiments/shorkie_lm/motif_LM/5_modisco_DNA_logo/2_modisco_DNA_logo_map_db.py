#!/usr/bin/env python3
"""
This script visualizes a DNA logo with overlaid modisco seqlet hits. In addition, it
performs “trimming” of the modisco motif (removing low‐score edges) and uses TomTom
to match the trimmed motif to a known motif database (in MEME format). The annotated
motif names on the logo will be the known motif names (if a match is found) rather than
the original modisco pattern names.
"""

from optparse import OptionParser
import os
import re
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
# (If using logomaker in the trimmed-logo plotting functions)
import logomaker

def compute_trimming_offsets(modisco_h5, trim_threshold, pad=4):
    """
    For each pattern in the modisco HDF5 file, compute trimmed boundaries based on
    the summed absolute contribution scores. Returns a dictionary mapping a pattern ID 
    (e.g. "pos_patterns/pattern_0") to a dict with trimming offsets and motif length.
    """
    offsets_dict = {}
    with h5py.File(modisco_h5, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type not in f:
                continue
            for pattern_name in f[pattern_type].keys():
                pattern = f[pattern_type][pattern_name]
                if "contrib_scores" not in pattern:
                    continue
                # Get the full forward contribution matrix
                cwm = np.array(pattern["contrib_scores"][:])
                L = cwm.shape[0]
                # Compute forward scores and trimmed indices
                score_fwd = np.sum(np.abs(cwm), axis=1)
                if np.max(score_fwd) == 0:
                    fwd_start, fwd_end = 0, L
                else:
                    trim_thresh_fwd = np.max(score_fwd) * trim_threshold
                    pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
                    if len(pass_inds_fwd) == 0:
                        fwd_start, fwd_end = 0, L
                    else:
                        fwd_start = max(np.min(pass_inds_fwd) - pad, 0)
                        fwd_end   = min(np.max(pass_inds_fwd) + pad + 1, L)
                
                # For reverse, compute a “reversed” cwm and do the same.
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
                # Convert the reverse trimming indices back to the original (forward) coordinates.
                rev_trim_start = L - rev_end_temp
                rev_trim_end   = L - rev_start_temp

                # Save the trimming offsets along with the motif length.
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
    Given a modisco hit (a row from modisco_df) and the trimming offsets dictionary,
    return (trimmed_start, trimmed_stop) in genomic coordinates.
    """
    pattern_id = row["source_label"]  # e.g. "pos_patterns/pattern_0"
    offsets = trim_offsets.get(pattern_id, None)
    # If no trimming info is found, return the original hit coordinates.
    if offsets is None:
        return row["start"], row["stop"]
    
    # The full length of the motif (should match row["stop"] - row["start"]).
    L = offsets['length']
    
    if row["strand"] == '+':
        trimmed_start = row["start"] + offsets['fwd_start']
        trimmed_stop  = row["start"] + offsets['fwd_end']
    else:
        # For reverse, the original mapping gives full hit as:
        #   row["start"] (genomic coordinate corresponding to motif position 0)
        #   row["stop"]   (row["start"] + L)
        # The trimmed region in the original (forward) orientation is then:
        trimmed_start = row["start"] + (L - offsets['rev_end'])
        trimmed_stop  = row["start"] + (L - offsets['rev_start'])
    return trimmed_start, trimmed_stop


#########################################
# FUNCTIONS FOR MAPPING MODISCO SEQLETS  #
#########################################

def read_bed_files(bed_files):
    """
    Reads one or more BED files and returns a list of entries.
    Each entry is a tuple: (chrom, start, end, label, identifier)
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
                    "start": genome_start,
                    "stop": genome_end,
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
            "start": hit["start"],
            "stop": hit["stop"],
            "source_label": f"{hit['pattern_type']}/{hit['pattern_name']}",
            "strand": hit["strand"] 
        }
        records.append(record)
    return pd.DataFrame(records)

#########################################
# HELPER FUNCTIONS FOR DNA LOGO PLOTTING  #
#########################################

def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    """
    Plot an ACGT letter as a PathPatch at a given x,y location scaled by yscale.
    """
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {
        "T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
        "UP": TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
        "DN": TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
        "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
        "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)
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

    t = mpl.transforms.Affine2D().scale(globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)
    return p

def plot_seq_scores(importance_scores, title, figsize=(16,2), plot_y_ticks=True, y_min=None, y_max=None, fig_name="default"):
    """
    Plot a sequence logo from a (L x 4) importance score array.
    Internally transposes the array to work with shape (4, L).
    """
    scores = importance_scores.T.copy()  # now shape (4, L)
    num_positions = scores.shape[1]

    fig, ax = plt.subplots(figsize=figsize, facecolor='none')
    for j in range(num_positions):
        col = scores[:, j]
        sort_index = np.argsort(-np.abs(col))
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

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
        )
    ax.set_xlabel("Position")
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=1)
    return fig, ax

def visualize_input_logo(att_grad_wt, title, fig_name='', figsize=(12,3)):
    """
    Compute y-axis bounds from the importance scores and then call plot_seq_scores.
    """
    y_min = np.min(att_grad_wt)
    y_max = np.max(att_grad_wt)
    y_max_abs = max(np.abs(y_min), np.abs(y_max))
    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.2 * y_max_abs

    print("y_min = " + str(round(y_min, 8)))
    print("y_max = " + str(round(y_max, 8)))

    fig, ax = plot_seq_scores(att_grad_wt, title=title, y_min=y_min, y_max=y_max,
                              figsize=figsize, plot_y_ticks=False, fig_name=fig_name + '_wt')
    return fig, ax

#########################################
# FUNCTIONS FOR MOTIF TRIMMING & TOMTOM   #
#########################################

def write_meme_file(ppm, bg, fname):
    """
    Write a single-motif MEME file.
    """
    with open(fname, 'w') as f:
        f.write('MEME version 4\n\n')
        f.write('ALPHABET= ACGT\n\n')
        f.write('strands: + -\n\n')
        f.write('Background letter frequencies (from unknown source):\n')
        f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(list(bg)))
        f.write('MOTIF 1 TEMP\n\n')
        f.write('letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n' % ppm.shape[0])
        for s in ppm:
            f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))

def fetch_tomtom_matches(ppm, cwm, is_writing_tomtom_matrix, output_dir,
                          pattern_name, motifs_db, background=[0.25,0.25,0.25,0.25],
                          tomtom_exec_path='tomtom', trim_threshold=0.3, trim_min_length=3):
    """
    Trim the motif (based on cwm scores), write a temporary MEME file and run TomTom
    to match the trimmed motif against a motifs database.
    Returns the TomTom results (using only the first match).
    """
    # Determine trimming positions
    score = np.sum(np.abs(cwm), axis=1)
    trim_thresh = np.max(score) * trim_threshold
    pass_inds = np.where(score >= trim_thresh)[0]
    if len(pass_inds) == 0:
        return pd.DataFrame()
    trimmed = ppm[np.min(pass_inds): np.max(pass_inds) + 1]
    if trimmed.shape[0] < trim_min_length:
        # Do not trim if too short
        trimmed = ppm
    # Write a temporary MEME file for the trimmed PWM
    fd, fname = tempfile.mkstemp()
    os.close(fd)
    write_meme_file(trimmed, background, fname)
    # Create temporary file for TomTom results
    fd2, tomtom_fname = tempfile.mkstemp()
    os.close(fd2)
    # Check that TomTom is callable
    if not shutil.which(tomtom_exec_path):
        raise ValueError(f'`tomtom` executable not found. Please install it.')
    # Run TomTom (adjust options as needed)
    cmd = f'{tomtom_exec_path} -no-ssc -oc . --verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 {fname} {motifs_db} > {tomtom_fname}'
    os.system(cmd)
    try:
        # Read only the second and sixth columns (Target ID and q-value) from TomTom output.
        tomtom_results = pd.read_csv(tomtom_fname, sep="\t", usecols=(1, 5))
    except Exception as e:
        tomtom_results = pd.DataFrame()
    # Clean up temporary files
    os.remove(fname)
    if is_writing_tomtom_matrix:
        output_subdir = os.path.join(output_dir, "tomtom")
        os.makedirs(output_subdir, exist_ok=True)
        output_filepath = os.path.join(output_subdir, f"{pattern_name}.tomtom.tsv")
        shutil.move(tomtom_fname, output_filepath)
    else:
        os.remove(tomtom_fname)
    return tomtom_results

def generate_tomtom_dataframe(modisco_h5py, output_dir, meme_motif_db,
                              is_writing_tomtom_matrix, pattern_groups,
                              top_n_matches=3, tomtom_exec="tomtom",
                              trim_threshold=0.3, trim_min_length=3):
    """
    Iterate over modisco patterns and run TomTom for each.
    Returns a DataFrame with one row per pattern and columns:
      'pattern', 'match0', 'qval0', etc.
    Note: In this example we only use the first (best) match.
    """
    tomtom_results = {}
    for i in range(top_n_matches):
        tomtom_results[f'match{i}'] = []
        tomtom_results[f'qval{i}'] = []
    patterns_list = []
    with h5py.File(modisco_h5py, 'r') as modisco_results:
        for contribution_dir_name in pattern_groups:
            if contribution_dir_name not in modisco_results.keys():
                continue
            metacluster = modisco_results[contribution_dir_name]
            # Sorting patterns by a numeric suffix if available.
            key = lambda x: int(x[0].split("_")[-1])
            for idx, (pattern_name, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                pattern_tag = f'{contribution_dir_name}.{pattern_name}'
                patterns_list.append(pattern_tag)
                ppm = np.array(pattern['sequence'][:])
                cwm = np.array(pattern["contrib_scores"][:])
                r = fetch_tomtom_matches(ppm, cwm,
                                          is_writing_tomtom_matrix=is_writing_tomtom_matrix,
                                          output_dir=output_dir,
                                          pattern_name=pattern_tag,
                                          motifs_db=meme_motif_db,
                                          tomtom_exec_path=tomtom_exec,
                                          trim_threshold=trim_threshold,
                                          trim_min_length=trim_min_length)
                # Use only the top match (if available)
                if not r.empty:
                    # r.iloc[0] contains the best match; adjust column names as needed.
                    target = r.iloc[0][r.columns[0]]
                    qval = r.iloc[0][r.columns[1]]
                else:
                    target = None
                    qval = None
                tomtom_results['match0'].append(target)
                tomtom_results['qval0'].append(qval)
                # (Fill in with None for additional columns if needed.)
                for i in range(1, top_n_matches):
                    tomtom_results[f'match{i}'].append(None)
                    tomtom_results[f'qval{i}'].append(None)
    df = pd.DataFrame({"pattern": patterns_list})
    for key, val in tomtom_results.items():
        df[key] = val
    return df

#########################################
# OTHER HELPER FUNCTIONS (e.g., model I/O)#
#########################################

def process_single_model(predictions_file):
    """
    Load the predictions file and return x_true, x_pred, label, weight_scale.
    """
    print(f"\t>> predictions_file: {predictions_file}")
    cache_bundle = np.load(predictions_file)
    x_true = cache_bundle['x_true']
    x_pred = cache_bundle['x_pred']
    label = cache_bundle['label']
    weight_scale = cache_bundle['weight_scale']
    print("x_true shape:", x_true.shape)
    print("x_pred shape:", x_pred.shape)
    return x_true, x_pred, label, weight_scale

#########################################
#         MAIN FUNCTION                 #
#########################################

def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option("--modisco_h5", dest='modisco_h5',
                      help="Path to TF-MoDISco HDF5 results file.")
    parser.add_option('--out_dir', dest='out_dir', default='', type='str',
                      help='Output directory [Default: %default]')
    parser.add_option('--seq_bed', dest='seq_bed', default=None, type='str',
                      help='BED file of sequences')
    parser.add_option('--predictions_file', dest='predictions_file', default=None, type='str',
                      help='MLM prediction output file')
    # New options for motif matching
    parser.add_option('--meme_db', dest='meme_db', default=None, type='str',
                      help='Known motif database in MEME format')
    parser.add_option('--trim_threshold', dest='trim_threshold', default=0.3, type='float',
                      help='Threshold for trimming seqlets [default: %default]')
    parser.add_option('--trim_min_length', dest='trim_min_length', default=3, type='int',
                      help='Minimum length after trimming [default: %default]')
    parser.add_option('--tomtom_exec', dest='tomtom_exec', default='tomtom', type='str',
                      help='Path to TomTom executable [default: %default]')
    
    (options, args) = parser.parse_args()
    print("Options:")
    print("  seq_bed:", options.seq_bed)
    print("  predictions_file:", options.predictions_file)
    print("  meme_db:", options.meme_db)
    
    # Load predictions and process x_pred as before
    x_true, x_pred, label, weight_scale = process_single_model(options.predictions_file)
    x_pred_transformed = np.copy(x_pred)
    x_pred_transformed += 0.0001  
    x_pred_transformed_mean = np.mean(x_pred_transformed, axis=1, keepdims=True)
    x_pred_transformed = x_pred_transformed * np.log(x_pred_transformed / x_pred_transformed_mean)
    x_pred = x_pred_transformed

    # Define BED files (you may want to parameterize these as well)
    bed_files = [
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_train_r64.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/sequences_test.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_valid.cleaned.bed"
    ]
    
    # Map modisco seqlets and convert to a DataFrame.
    hits = map_all_seqlets(options.modisco_h5, bed_files)
    modisco_df = convert_hits_to_dataframe(hits)
    print(f"Total modisco seqlet hits mapped: {len(modisco_df)}")
    
    # # If a MEME motif database is provided, run TomTom to get known motif matches.
    # motif_mapping = {}
    # if options.meme_db:
    #     print("Running TomTom motif matching with meme_db:", options.meme_db)
    #     pattern_groups = ['pos_patterns', 'neg_patterns']
    #     tomtom_df = generate_tomtom_dataframe(options.modisco_h5, options.out_dir, options.meme_db,
    #                                            is_writing_tomtom_matrix=False,
    #                                            pattern_groups=pattern_groups,
    #                                            top_n_matches=1,
    #                                            tomtom_exec=options.tomtom_exec,
    #                                            trim_threshold=options.trim_threshold,
    #                                            trim_min_length=options.trim_min_length)
    #     # Build a mapping from modisco pattern ID (e.g. "pos_patterns/pattern_0") to the best match.
    #     for i, row in tomtom_df.iterrows():
    #         # In tomtom_df the pattern is stored as "pos_patterns.pattern_0"; convert to match modisco_df.
    #         pattern_id = row['pattern']  # e.g. "pos_patterns.pattern_0"
    #         modisco_pattern_id = pattern_id.replace(".", "/")
    #         best_match = row['match0'] if pd.notnull(row['match0']) else modisco_pattern_id
    #         motif_mapping[modisco_pattern_id] = best_match
    #     print("Motif mapping dictionary:")
    #     print(motif_mapping)

    motif_mapping = {'pos_patterns/pattern_0': 'AZF1.6', 'pos_patterns/pattern_1': 'GCR2.6', 'pos_patterns/pattern_2': 'WTM1.2', 'pos_patterns/pattern_3': 'AZF1.7', 'pos_patterns/pattern_4': 'MIG3.4', 'pos_patterns/pattern_5': 'RPN4.7', 'pos_patterns/pattern_6': 'Sfp1p&consensus=cHgxRRAAAAWTTTTYHxxxt', 'pos_patterns/pattern_7': 'IME4.5', 'pos_patterns/pattern_8': 'ACA1.3', 'pos_patterns/pattern_9': 'Rfx1p&consensus=CTATTGCTGCAAC', 'pos_patterns/pattern_10': 'RDS2.4', 'pos_patterns/pattern_11': 'PHD1.13', 'pos_patterns/pattern_12': 'PHD1.13', 'pos_patterns/pattern_13': 'Nhp6ap&consensus=DxatttttATATAtaRDtxa', 'pos_patterns/pattern_14': 'Reb1p&consensus=CCGGGTAA', 'pos_patterns/pattern_15': 'PDR1.6', 'pos_patterns/pattern_16': 'FHL1.15', 'pos_patterns/pattern_17': 'ABF1.1', 'pos_patterns/pattern_18': 'PHO4.22', 'pos_patterns/pattern_19': 'ARG80.2', 'pos_patterns/pattern_20': 'CAT8.4', 'pos_patterns/pattern_21': 'SWI5.7', 'pos_patterns/pattern_22': 'PHD1.13', 'pos_patterns/pattern_23': 'PHD1.13', 'pos_patterns/pattern_24': 'FKH2.12', 'pos_patterns/pattern_25': 'DAL81.8', 'pos_patterns/pattern_26': 'RAP1.1', 'pos_patterns/pattern_27': 'HAC1.9', 'pos_patterns/pattern_28': 'CHA4.11', 'pos_patterns/pattern_29': 'Rme1p&consensus=GTACCACAAAA', 'pos_patterns/pattern_30': 'STB5.13', 'pos_patterns/pattern_31': 'Dot6p&consensus=gcGATGag', 'pos_patterns/pattern_32': 'YGR067C.3', 'pos_patterns/pattern_33': 'RIM101.1', 'pos_patterns/pattern_34': 'GAT1.13', 'pos_patterns/pattern_35': 'YAP7.5', 'pos_patterns/pattern_36': 'Reb1p&consensus=MGGGTAAB', 'pos_patterns/pattern_37': 'YKL222C.5', 'pos_patterns/pattern_38': 'UP00291_1', 'pos_patterns/pattern_39': 'STP1.4', 'pos_patterns/pattern_40': 'YAP3.3', 'pos_patterns/pattern_41': 'SFL1.4', 'pos_patterns/pattern_42': 'Cbf1p&consensus=CAcgtga', 'pos_patterns/pattern_43': 'CAD1.4', 'pos_patterns/pattern_44': 'STB1.7', 'pos_patterns/pattern_45': 'EDS1.2', 'pos_patterns/pattern_46': 'THI2.11', 'pos_patterns/pattern_47': 'AZF1.4', 'pos_patterns/pattern_48': 'IME4.5', 'pos_patterns/pattern_49': 'Hac1p&consensus=ATGGTATCAT', 'pos_patterns/pattern_50': 'STP4.3', 'pos_patterns/pattern_51': 'Tec1p&consensus=tcRRGAATGTtgxt', 'pos_patterns/pattern_52': 'YAP3.11', 'pos_patterns/pattern_53': 'Gcn4p&consensus=TTGCGTGA', 'pos_patterns/pattern_54': 'AZF1.4', 'pos_patterns/pattern_55': 'MIG3.5', 'pos_patterns/pattern_56': 'UME6.14', 'pos_patterns/pattern_57': 'SMP1.9', 'pos_patterns/pattern_58': 'MIG3.5', 'pos_patterns/pattern_59': 'RIM101.8', 'pos_patterns/pattern_60': 'BAS1.8', 'pos_patterns/pattern_61': 'Hac1p&consensus=ATGGTATCAT', 'pos_patterns/pattern_62': 'AZF1.1', 'pos_patterns/pattern_63': 'GCR2.7', 'pos_patterns/pattern_64': 'ROX1.3', 'pos_patterns/pattern_65': 'YJL206C.6', 'pos_patterns/pattern_66': 'GCN4.23', 'pos_patterns/pattern_67': 'HAP1.1', 'pos_patterns/pattern_68': 'MATALPHA1-MCM1-dimer', 'pos_patterns/pattern_69': 'YOX1.7', 'pos_patterns/pattern_70': 'Rim101p&consensus=CTTGGCR', 'pos_patterns/pattern_71': 'SNF1.1', 'pos_patterns/pattern_72': 'ECM22.7', 'pos_patterns/pattern_73': 'MCM1.1', 'pos_patterns/pattern_74': 'FKH1.3', 'pos_patterns/pattern_75': 'ASG1.3', 'pos_patterns/pattern_76': 'SWI6.5', 'pos_patterns/pattern_77': 'RGT1.7', 'pos_patterns/pattern_78': 'HAP2.10', 'pos_patterns/pattern_79': 'Ace2p&consensus=ACCAGC', 'pos_patterns/pattern_80': 'Stb3p&consensus=aattTTtCA', 'pos_patterns/pattern_81': 'PHO2.9', 'pos_patterns/pattern_82': 'PUT3.9', 'pos_patterns/pattern_83': 'UP00280_1', 'pos_patterns/pattern_84': 'PHD1.13', 'pos_patterns/pattern_85': 'UP00340_1', 'pos_patterns/pattern_86': 'ACE2.8', 'pos_patterns/pattern_87': 'Zap1p&consensus=TTCTTTATGGT', 'pos_patterns/pattern_88': 'GZF3.4', 'pos_patterns/pattern_89': 'ROX1.21', 'pos_patterns/pattern_90': 'TBS1.4', 'pos_patterns/pattern_91': 'Rfx1p&consensus=CTATTGCTGCAAC', 'pos_patterns/pattern_92': 'UP00324_1', 'pos_patterns/pattern_93': 'UP00340_1', 'pos_patterns/pattern_94': 'PHD1.13', 'pos_patterns/pattern_95': 'LEU3.6', 'pos_patterns/pattern_96': 'SWI5.7', 'pos_patterns/pattern_97': 'Dot6p&consensus=axgaxgCGATGAGStgH', 'pos_patterns/pattern_98': 'RTG3.17', 'pos_patterns/pattern_99': 'Cst6p&consensus=YRHACGTCAt', 'pos_patterns/pattern_100': 'SNF1.1', 'pos_patterns/pattern_101': 'SNF1.1', 'pos_patterns/pattern_102': 'DAL81.6', 'pos_patterns/pattern_103': 'SNF1.1', 'pos_patterns/pattern_104': 'SNF1.1', 'pos_patterns/pattern_105': 'HSF1.4', 'pos_patterns/pattern_106': 'UP00326_1', 'pos_patterns/pattern_107': 'AZF1.8', 'pos_patterns/pattern_108': 'MSN4.16', 'pos_patterns/pattern_109': 'STP4.1', 'pos_patterns/pattern_110': 'SWI6.1', 'pos_patterns/pattern_111': 'OPI1.1', 'pos_patterns/pattern_112': 'HCM1.3', 'pos_patterns/pattern_113': 'STB2.5', 'pos_patterns/pattern_114': 'PDR1.13', 'pos_patterns/pattern_115': 'GAL3.1', 'pos_patterns/pattern_116': 'MOT2.5', 'pos_patterns/pattern_117': 'YJL206C.6'}
    
    # Compute trimming offsets from the modisco HDF5 file.
    trim_offsets = compute_trimming_offsets(options.modisco_h5, options.trim_threshold, pad=4)

    # Load the sequences BED file (assumed tab-delimited with columns: contig, start, end, gene, species, strand)
    seqs_df = pd.read_csv(options.seq_bed, sep='\t', names=['contig', 'start', 'end', 'gene', 'species', 'strand'])
    print("Loaded", len(seqs_df), "sequences from", options.seq_bed)
    upstream = 450
    downstream = 50    
    for idx, row in seqs_df.iterrows():
        chrom  = row['contig']
        start  = int(row['start'])
        end    = int(row['end'])
        name   = row['gene']
        strand = row['strand']
        
        tss = (start + end) // 2
        if strand == '+':
            region_start = tss - upstream
            region_end   = tss + downstream
        else:
            region_start = tss - downstream
            region_end   = tss + upstream
        if region_start < 0:
            region_start = 0
        
        # Extract region from x_true and x_pred (adjusting indices based on the BED interval)
        x_true_one = x_true[idx][region_start - start: region_end - start, :]
        x_pred_one = x_pred[idx][region_start - start: region_end - start, :]

        # Filter modisco hits overlapping the selected region.
        region_hits = modisco_df[
            (modisco_df["sequence_name"] == chrom) &
            (modisco_df["stop"] > region_start) &
            (modisco_df["start"] < region_end)
        ]


        # counter = 0
        if len(region_hits) > 0:
            numhit = len(region_hits)
            print(f"Found {len(region_hits)} modisco hits overlapping the region {chrom}:{region_start}-{region_end}.")
            color_options = ["red"]  # You can add more colors if desired.
            region_length = region_end - region_start


            # Set a quality threshold: skip matches with q-value above this value.
            qval_threshold = 0.05  # adjust as needed

            # Sort the hits by genomic start coordinate so we can process them in order.
            region_hits_sorted = region_hits.sort_values(by="start")
            
            # Prepare a list of annotation info to place the motif boxes without overlapping.
            # We will use a greedy “interval scheduling” approach: each annotation is assigned a row (vertical offset)
            # such that if two annotations overlap in x, they get different rows.
            rows = []  # each element is a list of (rel_start, width) intervals in that row
            annotation_info = []  # list to store (rel_start, width, row_index, known_motif, color)

            # You can define a list of colors (or derive from the motif name) as before.
            # color_options = ["red", "blue", "green", "purple"]  # extend as desired





            fig, ax = visualize_input_logo(x_pred_one,
                                           title=f"DNA Logo: {chrom}:{region_start}-{region_end}",
                                           figsize=(100, 3))
            
            for ridx, rrow in region_hits.iterrows():
                # Get the trimmed genomic coordinates for this hit.
                print(rrow)
                t_start, t_stop = get_trimmed_genomic_coords(rrow, trim_offsets)
                # Compute relative positions (logo x-axis = 0 corresponds to region_start)
                rel_start = max(t_start, region_start) - region_start
                rel_end   = min(t_stop, region_end) - region_start
                width = rel_end - rel_start
                if width <= 0:
                    continue
                # Look up the known motif name (if available) from our mapping dictionary.
                original_label = rrow["source_label"]

                known_motif = motif_mapping.get(original_label, original_label)
                color = color_options[hash(known_motif) % len(color_options)]





            #     # Determine the row (vertical offset) for this annotation.
            #     # We check the current rows to see if this [rel_start, rel_end] overlaps with any
            #     # annotation already placed in that row.
            #     assigned_row = None
            #     for i, row in enumerate(rows):
            #         overlap = False
            #         for (s, w) in row:
            #             # Two intervals [s, s+w] and [rel_start, rel_end] overlap if:
            #             #    not (current hit is completely to the left or right)
            #             if not (rel_end <= s or rel_start >= s + w):
            #                 overlap = True
            #                 break
            #         if not overlap:
            #             assigned_row = i
            #             row.append((rel_start, width))
            #             break
            #     if assigned_row is None:
            #         # No existing row can hold this annotation without overlap; create a new row.
            #         rows.append([(rel_start, width)])
            #         assigned_row = len(rows) - 1

            #     # Determine a color (here we use a hash on the motif name, but you can adjust as desired)
            #     color = color_options[hash(known_motif) % len(color_options)]
            #     # Save the annotation information (we’ll draw all annotations after processing the hits).
            #     annotation_info.append((rel_start, width, assigned_row, known_motif, color))

            # # Now, determine where to draw these annotation boxes.
            # # We'll draw them above the DNA logo.
            # logo_top = ax.get_ylim()[1]  # get the top of the current DNA logo (y-axis)
            # base_y = logo_top + 0.5       # add a margin; adjust as needed
            # box_height = 0.5              # height of each annotation box (adjust as needed)
            # row_spacing = 0.1             # extra spacing between rows

            # # Draw the boxes and annotation texts.
            # for (rel_start, width, row_index, known_motif, color) in annotation_info:
            #     y = base_y + row_index * (box_height + row_spacing)
            #     # Draw the rectangle (box) for the motif annotation.
            #     rect = Rectangle((rel_start, y), width, box_height, linewidth=2,
            #                     edgecolor=color, facecolor="none", linestyle="--")
            #     ax.add_patch(rect)
            #     # Draw the motif name above the rectangle.
            #     ax.text(rel_start + width/2, y + box_height,
            #             known_motif, color=color, ha='center', va='bottom', fontsize=10)







                # Draw a rectangle spanning the vertical extent of the logo over only the trimmed region.
                rect = Rectangle((rel_start, 0), width, 1.7, linewidth=2,
                                 edgecolor=color, facecolor="none", linestyle="--")
                

                ax.add_patch(rect)
                # Annotate the rectangle with the motif name.
                ax.text(rel_start + width/2, 1.7, known_motif,
                        color=color, ha='center', va='bottom', fontsize=10, rotation=0)
            
            ax.set_xlim(0, region_length)
            ax.set_yticks([])
            ax.set_facecolor('none')
            plt.tight_layout()
            outfile = os.path.join(options.out_dir, f"{numhit}_logo_{chrom}_{region_start}_{region_end}_{idx}_{name}_{strand}.png")
            plt.savefig(outfile, dpi=300, transparent=True)
            print(f"Saved logo plot with trimmed motif annotations to {outfile}")
            plt.show()
            plt.clf()

            # counter += 1
            # if counter >= 1:
            #     break   # Remove or adjust if you wish to process more sequences.

if __name__ == "__main__":
    main()
