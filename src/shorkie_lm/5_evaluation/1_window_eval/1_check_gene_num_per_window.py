import pandas as pd
from intervaltree import IntervalTree
import argparse

def read_gff(file_path):
    """Reads GFF file and extracts gene intervals."""
    gff = pd.read_csv(file_path, sep='\t', comment='#', header=None,
                      names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'])
    genes = gff[gff['type'] == 'gene'].copy()
    genes['start'] = genes['start'].astype(int)
    genes['end'] = genes['end'].astype(int)
    
    # Assuming coding regions are marked as 'CDS' in the GFF file
    coding_regions = gff[gff['type'] == 'CDS'].copy()
    coding_regions['start'] = coding_regions['start'].astype(int)
    coding_regions['end'] = coding_regions['end'].astype(int)
    
    return genes, coding_regions

def read_bed(file_path):
    """Reads BED file and extracts intervals."""
    bed = pd.read_csv(file_path, sep='\t', header=None, 
                      names=['chrom', 'start', 'end', 'name', 'other'])
    bed_intervals = bed[['chrom', 'start', 'end']].copy()
    bed_intervals['chrom'] = bed_intervals['chrom'].str.replace('^chr', '', regex=True)  # Remove 'chr' if it starts the string
    bed_intervals['start'] = bed_intervals['start'].astype(int)
    bed_intervals['end'] = bed_intervals['end'].astype(int)
    return bed_intervals

def build_interval_tree(regions):
    """Builds an interval tree for each chromosome."""
    interval_trees = {}
    for _, row in regions.iterrows():
        chrom = str(row['seqid'])
        start = row['start']
        end = row['end']
        if start != end:  # Filter out null intervals
            if chrom not in interval_trees:
                interval_trees[chrom] = IntervalTree()
            interval_trees[chrom].addi(start, end, (start, end))
        else:
            print(f"Skipping null interval: {chrom} {start}-{end}")
    return interval_trees

def calculate_nucleotide_counts(interval_tree, interval_start, interval_end):
    """Calculates the number of nucleotides in coding and non-coding regions."""
    nucleotides_coding = 0
    nucleotides_non_coding = 0
    overlaps = interval_tree[interval_start:interval_end]
    for overlap in overlaps:
        start = max(interval_start, overlap.begin)
        end = min(interval_end, overlap.end)
        length = end - start
        nucleotides_coding += length
    nucleotides_non_coding = (interval_end - interval_start) - nucleotides_coding
    return nucleotides_coding, nucleotides_non_coding

def count_overlaps_and_nucleotides(gene_trees, coding_trees, bed_intervals):
    """Counts overlaps and calculates nucleotide counts for each BED interval."""
    overlap_counts = []
    for _, row in bed_intervals.iterrows():
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        total_nucleotides = end - start
        if chrom in gene_trees:
            gene_tree = gene_trees[chrom]
            overlaps = gene_tree[start:end]
            complete_within = [iv for iv in overlaps if iv.begin >= start and iv.end <= end]
            boundary_overlap = [iv for iv in overlaps if iv.begin < start or iv.end > end]

            if chrom in coding_trees:
                coding_tree = coding_trees[chrom]
                nucleotides_coding, nucleotides_non_coding = calculate_nucleotide_counts(coding_tree, start, end)        
            else:
                nucleotides_coding = 0
                nucleotides_non_coding = total_nucleotides
            overlap_counts.append((chrom, start, end, len(overlaps), len(complete_within), len(boundary_overlap), 
                                   nucleotides_coding, nucleotides_non_coding))
        else:
            overlap_counts.append((chrom, start, end, 0, 0, 0, 0, total_nucleotides))
    # print("overlap_counts: ", overlap_counts)
    for i in range(len(overlap_counts)):
        # print(len(overlap_counts[i]))
        if len(overlap_counts[i]) == 10:
            print(overlap_counts[i])
    return pd.DataFrame(overlap_counts, columns=['chrom', 'start', 'end', 'total_overlap', 'complete_within', 
                                                 'boundary_overlap', 'nucleotides_coding', 'nucleotides_non_coding'])



def main(gff_file, bed_file, output):
    genes, coding_regions = read_gff(gff_file)
    bed_intervals = read_bed(bed_file)
    print("bed_intervals: ", set(bed_intervals['chrom']))
    
    # Debugging: Check the first few rows of the dataframes
    print("GFF Genes:")
    print(genes.head())
    print("GFF Coding Regions:")
    print(coding_regions.head())
    print("BED Intervals:")
    print(bed_intervals.head())

    gene_trees = build_interval_tree(genes)
    coding_trees = build_interval_tree(coding_regions)
    print("gene_trees: ", gene_trees.keys())
    print("coding_trees: ", coding_trees.keys())
    overlap_counts = count_overlaps_and_nucleotides(gene_trees, coding_trees, bed_intervals)

    overlap_counts.to_csv(output, index=False)
    print("Overlap counts saved to overlap_counts.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count overlaps between GFF and BED intervals.')
    parser.add_argument('--gff_file', type=str, help='Path to the GFF file')
    parser.add_argument('--bed_file', type=str, help='Path to the BED file')
    parser.add_argument('--output', type=str, help='Path to the output file (default: overlap_counts.csv)', default='overlap_counts.csv')
    
    args = parser.parse_args()
    
    main(args.gff_file, args.bed_file, args.output)
