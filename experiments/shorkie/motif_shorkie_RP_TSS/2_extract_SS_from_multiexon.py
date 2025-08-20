#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict

def parse_attributes(attr_str):
    """
    Parse the attributes column of a GTF line.
    Returns a dictionary mapping keys to values.
    """
    attrs = {}
    for attr in attr_str.strip().strip(';').split(';'):
        if attr.strip() == "":
            continue
        # Attributes are usually formatted as: key "value"
        parts = attr.strip().split(' ', 1)
        if len(parts) == 2:
            key = parts[0]
            # Remove any quotes from the value
            value = parts[1].replace('"', '').strip()
            attrs[key] = value
    return attrs

def main():
    parser = argparse.ArgumentParser(description="Extract gene regions (extended by +/-50bp) for genes with at least one transcript having >1 exon from a GTF annotation.")
    parser.add_argument("-i", "--input", required=True, help="Input GTF file")
    parser.add_argument("-o", "--output", required=True, help="Output BED file")
    args = parser.parse_args()

    # Dictionaries to hold counts and mappings.
    transcript_exon_counts = defaultdict(int)
    transcript_to_gene = {}
    gene_to_transcripts = defaultdict(set)
    
    # Store gene regions from gene feature lines.
    gene_info = {}
    # If gene feature not available, use union of exons.
    # Store for each gene: list of tuples (chrom, start, end, strand)
    gene_exons = defaultdict(list)

    try:
        with open(args.input, "r") as fin:
            for line in fin:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue
                chrom, source, feature, start, end, score, strand, frame, attributes = fields
                attr_dict = parse_attributes(attributes)
                
                if feature == "gene":
                    gene_id = attr_dict.get("gene_id")
                    if gene_id:
                        # Store gene region from gene feature (GTF is 1-based inclusive)
                        gene_info[gene_id] = (chrom, int(start), int(end), strand)
                
                if feature == "exon":
                    gene_id = attr_dict.get("gene_id")
                    transcript_id = attr_dict.get("transcript_id")
                    if transcript_id and gene_id:
                        transcript_exon_counts[transcript_id] += 1
                        transcript_to_gene[transcript_id] = gene_id
                        gene_to_transcripts[gene_id].add(transcript_id)
                        # Also record the exon for gene union computation if gene feature is missing.
                        gene_exons[gene_id].append((chrom, int(start), int(end), strand))
    except FileNotFoundError:
        sys.exit(f"Error: Cannot open input file {args.input}")

    # Prepare output BED file lines.
    # BED format: chrom, start (0-based), end (end coordinate), gene_id, score, strand.
    # We extend the gene region by 50bp upstream and downstream.
    bed_lines = []
    for gene_id, transcripts in gene_to_transcripts.items():
        # Check if at least one transcript has more than one exon.
        has_multi_exon = any(transcript_exon_counts[tid] > 1 for tid in transcripts)
        if not has_multi_exon:
            continue

        # Determine gene coordinates: use gene_info if available; otherwise, compute from exons.
        if gene_id in gene_info:
            chrom, gene_start, gene_end, strand = gene_info[gene_id]
        else:
            # Compute union of all exon coordinates for this gene.
            if gene_exons[gene_id]:
                chrom = gene_exons[gene_id][0][0]
                gene_start = min(exon[1] for exon in gene_exons[gene_id])
                gene_end = max(exon[2] for exon in gene_exons[gene_id])
                # If strands are not consistent, default to '.'
                strand_set = {exon[3] for exon in gene_exons[gene_id]}
                strand = strand_set.pop() if len(strand_set) == 1 else "."
            else:
                continue  # Should not happen, but skip if no exon recorded

        # Extend region by 50bp. Convert to 0-based coordinates for BED (subtract one from start).
        bed_start = max(0, gene_start - 50 - 1)  # ensuring non-negative start
        bed_end = gene_end + 50
        if chr == "Mito":
            continue
        # Using a placeholder score of 0.
        bed_lines.append(f"chr{chrom}\t{bed_start}\t{bed_end}\t{gene_id}\t0\t{strand}")

    try:
        with open(args.output, "w") as fout:
            for line in bed_lines:
                fout.write(line + "\n")
        print(f"BED file written to: {args.output}")
    except IOError as e:
        sys.exit(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main()
