#!/usr/bin/env python3
import os
import pandas as pd
from collections import defaultdict

# Function to parse the attributes column from a GTF line
def parse_attributes(attribute_str):
    """
    Parse a GTF attribute string into a dictionary.
    Example:
        'gene_id "geneA"; transcript_id "transA";'
    Returns a dict like:
        {'gene_id': '"geneA"', 'transcript_id': '"transA"'}
    """
    attributes = {}
    parts = attribute_str.strip().split(';')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        key_value = part.split(' ', 1)
        if len(key_value) == 2:
            key, value = key_value
            attributes[key] = value
    return attributes

# Helper function to check whether two intervals overlap
def intervals_overlap(start1, end1, start2, end2):
    return not (end1 < start2 or start1 > end2)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Check multi-exon transcript overlaps in sequence folds.")
    parser.add_argument("--root_dir", default="../../..", help="Root directory pointing to Yeast_ML")
    args = parser.parse_args()
    
    # File paths (update these paths as needed)
    sequence_fold_bed = f"{args.root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_model/data/sequences.bed"
    gtf_file = f"{args.root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf"

    # Load the fold regions from the BED file into a DataFrame
    # Expecting columns: chrom, start, end, fold
    df_fold = pd.read_csv(sequence_fold_bed, sep="\t", header=None, names=["chrom", "start", "end", "fold"])
    
    # Parse the GTF file to build a dictionary of transcript regions and count exons.
    # For each transcript (identified by 'Parent' or 'transcript_id'),
    # we record the minimum exon start, maximum exon end, and the exon count.
    transcripts = {}  # key: transcript_id, value: dict with chrom, start, end, exon_count
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 9:
                continue
            chrom = cols[0]
            feature_type = cols[2]
            try:
                start = int(cols[3])
                end = int(cols[4])
            except ValueError:
                continue
            attributes_str = cols[8]
            
            # We only care about exon features
            if feature_type.lower() == 'exon':
                attributes = parse_attributes(attributes_str)
                # Try to get the transcript id from either 'Parent' or 'transcript_id'
                if 'Parent' in attributes:
                    transcript_id = attributes['Parent']
                elif 'transcript_id' in attributes:
                    transcript_id = attributes['transcript_id']
                else:
                    continue
                
                if transcript_id not in transcripts:
                    transcripts[transcript_id] = {
                        "chrom": "chr" + chrom,
                        "start": start,
                        "end": end,
                        "exon_count": 1
                    }
                else:
                    transcripts[transcript_id]["start"] = min(transcripts[transcript_id]["start"], start)
                    transcripts[transcript_id]["end"] = max(transcripts[transcript_id]["end"], end)
                    transcripts[transcript_id]["exon_count"] += 1

    # For each transcript, check which fold regions overlap its boundaries.
    results = []
    for transcript_id, info in transcripts.items():
        chrom = info["chrom"]
        t_start = info["start"]
        t_end = info["end"]
        exon_count = info["exon_count"]
        
        # Get fold regions on the same chromosome
        df_chrom = df_fold[df_fold["chrom"] == chrom]
        # Filter those that overlap the transcript region
        overlapping_rows = df_chrom[df_chrom.apply(
            lambda row: intervals_overlap(t_start, t_end, row["start"], row["end"]),
            axis=1
        )]
        
        if not overlapping_rows.empty:
            # Build comma-separated lists of folds and overlap ranges
            overlap_folds = []
            overlap_ranges = []
            for idx, row in overlapping_rows.iterrows():
                overlap_start = max(t_start, row["start"])
                overlap_end = min(t_end, row["end"])
                overlap_folds.append(str(row["fold"]))
                overlap_ranges.append(f"{overlap_start}-{overlap_end}")
            
            fold_str = ",".join(overlap_folds)
            range_str = ",".join(overlap_ranges)
        else:
            fold_str = "None"
            range_str = "None"
        
        results.append({
            "transcript_id": transcript_id,
            "chrom": chrom,
            "transcript_start": t_start,
            "transcript_end": t_end,
            "exon_count": exon_count,
            "overlapping_fold": fold_str,
            "overlapping_range": range_str
        })
    
    # Create a DataFrame for the results and output to the terminal and a CSV file
    df_results = pd.DataFrame(results)
    print(df_results)
    
    output_dir = "results" 
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/0_transcript_fold_overlaps.csv"
    df_results.to_csv(output_file, index=False, sep="\t")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
