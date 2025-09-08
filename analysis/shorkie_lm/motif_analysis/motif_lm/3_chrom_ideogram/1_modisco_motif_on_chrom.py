#!/usr/bin/env python3
"""
This script takes a TF-MoDISco results HDF5 file, one or more BED files,
and a GTF annotation file, maps seqlets to genomic coordinates, and
plots the motif (seqlet) positions along with gene regions on a yeast
chromosome ideogram.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py

############################
# Yeast Chromosome Sizes   #
############################
# Adjust for your specific yeast reference version.
YEAST_CHR_SIZES = {
    "chrI":   230218,
    "chrII":  813184,
    "chrIII": 316620,
    "chrIV":  1531933,
    "chrV":   576874,
    "chrVI":  270161,
    "chrVII": 1090940,
    "chrVIII": 562643,
    "chrIX":  439888,
    "chrX":   745751,
    "chrXI":  666816,
    "chrXII": 1078177,
    "chrXIII": 924431,
    "chrXIV": 784333,
    "chrXV":  1091291,
    "chrXVI": 948066
}

# Explicit chromosome order for plotting:
CHR_ORDER = [
    "chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI",
    "chrVII", "chrVIII", "chrIX", "chrX", "chrXI", "chrXII",
    "chrXIII", "chrXIV", "chrXV", "chrXVI"
]

def append_chr_to_roman(chromosome):
    """Convert a chromosome label like 'I' to 'chrI' if needed."""
    roman_mapping = {
        'I': 'chrI', 'II': 'chrII', 'III': 'chrIII', 'IV': 'chrIV',
        'V': 'chrV', 'VI': 'chrVI', 'VII': 'chrVII', 'VIII': 'chrVIII',
        'IX': 'chrIX', 'X': 'chrX', 'XI': 'chrXI', 'XII': 'chrXII',
        'XIII': 'chrXIII', 'XIV': 'chrXIV', 'XV': 'chrXV', 'XVI': 'chrXVI',
    }
    return roman_mapping.get(chromosome, chromosome)

############################
# GTF Parsing Functions    #
############################
def read_gtf(gtf_file, feature_type="gene"):
    """
    Reads a GTF file into a Pandas DataFrame and keeps only rows
    where the 'feature' equals the specified type (default 'gene').
    Returns a DataFrame with columns:
      ['seqname', 'start', 'end', 'strand', 'feature', 'gene_name']
    """
    col_names = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ]
    df = pd.read_csv(gtf_file, sep='\t', comment='#', names=col_names, low_memory=False)
    
    # Filter to keep only the desired feature (e.g., gene)
    df = df[df["feature"] == feature_type]
    
    # Parse the gene name (or gene id) from the attribute column
    df["gene_name"] = df["attribute"].apply(parse_gene_name)
    return df

def parse_gene_name(attr_string):
    """
    Extracts 'gene_name' (or falls back to 'gene_id') from a GTF attribute string.
    Example attribute: gene_id "YAL069W"; gene_version "1"; gene_name "CAF16";
    """
    fields = attr_string.strip().split(";")
    for field in fields:
        field = field.strip()
        if field.startswith("gene_name"):
            parts = field.split()
            if len(parts) >= 2:
                return parts[1].replace('"', '')
        elif field.startswith("gene_id"):
            parts = field.split()
            if len(parts) >= 2:
                return parts[1].replace('"', '')
    return "unknown_gene"

############################
# Plotting Functions       #
############################
def plot_yeast_ideogram_with_genes(motif_df, gene_df, out_png="yeast_ideogram.png"):
    """
    Plots the yeast chromosomes as thick horizontal bars, overlays gene regions,
    and plots motif hits (seqlets) as points. The motif_df is expected to have columns:
      'sequence_name' (chromosome), 'start', 'stop', and 'source_label'
    """
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Map each chromosome to a y-position for plotting.
    y_positions = {chr_name: i for i, chr_name in enumerate(CHR_ORDER)}
    bar_height = 0.6
    
    # Draw chromosome bars.
    for chr_name in CHR_ORDER:
        if chr_name not in YEAST_CHR_SIZES:
            continue
        chr_len = YEAST_CHR_SIZES[chr_name]
        y_val = y_positions[chr_name]
        ax.barh(
            y=y_val,
            width=chr_len,
            left=0,
            height=bar_height,
            color='lightgray',
            edgecolor='black'
        )
    
    # Draw gene regions as thinner bars.
    for idx, row in gene_df.iterrows():
        chrom = row["seqname"]
        chrom = append_chr_to_roman(chrom)
        if chrom not in y_positions:
            continue
        y_val = y_positions[chrom]
        start = row["start"]
        end = row["end"]
        # Color code by strand.
        color = "blue" if row["strand"] == "+" else "green"
        gene_width = end - start
        gene_bar_height = bar_height * 0.4
        y_offset = 0.1 if row["strand"] == "+" else -0.1
        ax.barh(
            y=y_val + y_offset,
            width=gene_width,
            left=start,
            height=gene_bar_height,
            color=color,
            alpha=0.7
        )
    
    # Plot motif hits (seqlets) as points.
    for idx, row in motif_df.iterrows():
        chrom = row["sequence_name"]
        if chrom not in y_positions:
            continue
        y_val = y_positions[chrom]
        start = int(row["start"])
        stop = int(row["stop"])
        midpoint = (start + stop) / 2
        
        # Choose a color based on the source label.
        label = row["source_label"]
        # (You can modify the color mapping as desired.)
        color_options = ["red", "purple", "orange", "cyan", "magenta", "yellow", "black"]
        color = color_options[hash(label) % len(color_options)]
        
        ax.plot(midpoint, y_val, marker='o', markersize=0.5, color=color, alpha=0.8)
    
    # Formatting the plot.
    ax.set_yticks([y_positions[ch] for ch in CHR_ORDER])
    ax.set_yticklabels(CHR_ORDER)
    ax.set_xlabel("Genomic coordinate (bp)")
    ax.set_ylabel("Chromosome")
    ax.set_title("TF-MoDISco Seqlet Hits & Gene Regions on Yeast Chromosomes")
    
    max_chr_len = max(YEAST_CHR_SIZES.values())
    ax.set_xlim(-0.01 * max_chr_len, 1.01 * max_chr_len)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")

############################
# Modisco Processing Functions  #
############################
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
                # Assuming BED columns: chrom, start, end, label, identifier
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
    
    The assumption is that the seqlet positions are relative to the sequence
    extracted from the BED file; thus:
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

############################
# Main Function            #
############################
def main():
    """
    Usage:
      python plot_modisco_ideogram.py modisco_results.h5 annotation.gtf output.png
      
    Arguments:
      modisco_results.h5 : Path to the TF-MoDISco HDF5 results file.
      annotation.gtf   : GTF file for gene annotations.
      output.png       : Filename for the output plot.
    """
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} modisco_results.h5 bed1.bed [bed2.bed ...] annotation.gtf output.png")
        sys.exit(1)
    
    modisco_h5 = sys.argv[1]
    annotation_gtf = sys.argv[-2]
    out_png = sys.argv[-1]
    # bed_files = sys.argv[2:-2]
    
    bed_files = [
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_train_r64.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/sequences_test.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_valid.cleaned.bed"
    ]

    if len(bed_files) < 1:
        print("Error: At least one BED file must be provided.")
        sys.exit(1)

    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n5000.h5"
    
    print("Processing TF-MoDISco results...")
    hits = map_all_seqlets(modisco_h5, bed_files)
    print(f"Total seqlet hits mapped: {len(hits)}")
    
    # Convert hits to a DataFrame that the plotting function can use.
    modisco_df = convert_hits_to_dataframe(hits)
    
    # Load gene annotations from the provided GTF file.
    gene_df = read_gtf(annotation_gtf, feature_type="gene")
    
    # Create the ideogram plot.
    plot_yeast_ideogram_with_genes(modisco_df, gene_df, out_png=out_png)

if __name__ == "__main__":
    main()
