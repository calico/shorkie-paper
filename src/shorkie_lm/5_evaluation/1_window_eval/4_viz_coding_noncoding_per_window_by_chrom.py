import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os, sys

coding_ratio = []

def viz(input, output):
    # Read the input data
    df = pd.read_csv(input, sep=',', header=0)
    
    # Calculate the percentages of coding and non-coding nucleotides
    df['total_nucleotides'] = df['nucleotides_coding'] + df['nucleotides_non_coding']
    df['coding_percentage'] = df['nucleotides_coding'] / df['total_nucleotides']
    df['non_coding_percentage'] = df['nucleotides_non_coding'] / df['total_nucleotides']
    
    # Calculate the overall coding percentage
    total_coding_nucleotides = df['nucleotides_coding'].sum()
    total_nucleotides = df['total_nucleotides'].sum()
    overall_coding_percentage = (total_coding_nucleotides / total_nucleotides) * 100
    coding_ratio.append(overall_coding_percentage)
    
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
        
        # Plot stacked bar plots
        ax.bar(range(len(group_sorted)), group_sorted['coding_percentage'], label='Coding', alpha=0.7)
        ax.bar(range(len(group_sorted)), group_sorted['non_coding_percentage'], bottom=group_sorted['coding_percentage'], label='Non-Coding', alpha=0.7)
        
        ax.set_title(f'Coding/Non-Coding Percentage for Chromosome {chrom}')
        ax.set_xlabel('Entries (sorted by start position)')
        ax.set_ylabel('Percentage')
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