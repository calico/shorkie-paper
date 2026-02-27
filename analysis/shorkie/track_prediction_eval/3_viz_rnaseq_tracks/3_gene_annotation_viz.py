import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def parse_gtf_attributes(attr_str):
    """Parse GTF attribute string into a dictionary."""
    attributes = {}
    parts = attr_str.strip(';').split('; ')
    for part in parts:
        if not part: continue
        key, val = part.split(' ', 1)
        val = val.strip('"')
        attributes[key] = val
    return attributes

def parse_gtf(gtf_file, region_chr, region_start, region_end):
    """Parse GTF file and extract features in the specified genomic region."""
    columns = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
    gtf = pd.read_csv(gtf_file, sep='\t', comment='#', names=columns, usecols=range(9))
    
    # Filter and parse attributes
    filtered = gtf[
        (gtf['seqid'] == region_chr) &
        (gtf['end'] >= region_start) &
        (gtf['start'] <= region_end) &
        (gtf['type'].isin(['gene', 'exon']))
    ].copy()
    filtered['attributes'] = filtered['attributes'].apply(parse_gtf_attributes)
    
    return filtered

def plot_genes(gtf_data, region_chr, region_start, region_end, output_file):
    """Plot genes and exons in IGV-like style using GTF data."""
    fig, ax = plt.subplots(figsize=(10, 4))
    y_level = 0
    gene_tracker = {}  # Track gene positions to avoid overlaps

    for _, row in gtf_data.iterrows():
        if row['type'] == 'gene':
            # Get gene identifier
            gene_id = row['attributes'].get('gene_id', 'unknown_gene')
            gene_name = row['attributes'].get('gene_name', gene_id)
            
            # Manage track levels for overlapping genes
            y_level = 1
            while any(abs(row['start'] - g[0]) < 100 and abs(row['end'] - g[1]) < 100 
                      for g in gene_tracker.get(y_level, [])):
                y_level += 1
            
            # Draw gene strand-aware arrow
            color = 'skyblue'
            if row['strand'] == '+':
                arrow = Polygon([(row['start'], y_level), 
                               (row['end'] - 10, y_level), 
                               (row['end'], y_level + 0.5), 
                               (row['end'] - 10, y_level), 
                               (row['start'], y_level)], closed=True, fc=color)
            else:
                arrow = Polygon([(row['start'], y_level + 0.5), 
                               (row['start'] + 10, y_level), 
                               (row['end'], y_level), 
                               (row['start'] + 10, y_level), 
                               (row['start'], y_level + 0.5)], closed=True, fc=color)
            ax.add_patch(arrow)
            ax.text(row['start'], y_level + 0.7, gene_name, fontsize=8)
            gene_tracker.setdefault(y_level, []).append((row['start'], row['end']))

        elif row['type'] == 'exon':
            # Draw exon under its parent gene
            exon = Rectangle((row['start'], y_level - 0.2), 
                             row['end'] - row['start'], 0.4, fc='darkblue')
            ax.add_patch(exon)

    # Configure plot
    ax.set_xlim(region_start, region_end)
    ax.set_ylim(0, y_level + 1)
    ax.set_xlabel(f'Genomic Position on {region_chr}', fontsize=10)
    ax.set_title(f'Gene Visualization: {region_chr}:{region_start}-{region_end}', fontsize=12)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize genes in a genomic region using GTF data.")
    parser.add_argument("--region", help="Genomic region (e.g., chr1:1000-5000)", required=True)
    parser.add_argument("--output", help="Output image file (e.g., plot.png)", default="gene_plot.png")
    parser.add_argument("--root_dir", help="Root directory for Yeast_ML", default="../../..")
    args = parser.parse_args()

    root_dir = args.root_dir
    gtf_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf'

    region_chr, pos = args.region.split(':')
    region_start, region_end = map(int, pos.split('-'))
    
    gtf_data = parse_gtf(gtf_file, region_chr, region_start, region_end)
    plot_genes(gtf_data, region_chr, region_start, region_end, args.output)
    print(f"Plot saved to {args.output}")