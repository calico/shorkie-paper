import pandas as pd
import argparse
import matplotlib.pyplot as plt

def viz(input, output):
    # Read the input data
    df = pd.read_csv(input, sep=',', header=0)
    
    # Group by chromosome and sort by start position within each group
    grouped = df.groupby('chrom')
    
    # Determine the number of chromosomes
    num_chroms = len(grouped)
    
    # Create subplots with an appropriate layout
    fig, axes = plt.subplots(nrows=num_chroms, ncols=1, figsize=(28, 6*num_chroms))
    
    # If there's only one chromosome, axes will not be an array, so we convert it to a list
    if num_chroms == 1:
        axes = [axes]
    
    for ax, (chrom, group) in zip(axes, grouped):
        group_sorted = group.sort_values(by='start')
        
        ax.bar(range(len(group_sorted)), group_sorted['total_overlap'], alpha=0.7, label='Number of genes overlapped')
        ax.bar(range(len(group_sorted)), group_sorted['complete_within'], alpha=0.7, label='Number of genes completely enveloped')
        
        ax.set_title(f'Overlap Features for Chromosome {chrom}')
        ax.set_xlabel('Entries (sorted by start position)')
        ax.set_ylabel('Feature Count')
        ax.set_xticks(ticks=range(len(group_sorted)))
        ax.set_xticklabels(group_sorted['start'], rotation=90)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count overlaps between GFF and BED intervals.')
    parser.add_argument('--input', type=str, help='Path to the input file', default='GCA_000146045_2_ovp.txt')
    parser.add_argument('--output', type=str, help='Path to the output file (default: overlap_counts.png)', default='overlap_counts.png')
    args = parser.parse_args()
    viz(args.input, args.output)