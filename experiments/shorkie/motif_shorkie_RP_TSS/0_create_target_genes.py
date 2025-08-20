#!/usr/bin/env python3
import sys
import re

def parse_gtf_attributes(attr_str):
    """
    Parse the GTF attribute string and return a dictionary of key/value pairs.
    Example attribute format: key "value"; key2 "value2";
    """
    attrs = {}
    for attribute in attr_str.strip().split(';'):
        if attribute.strip() == "":
            continue
        parts = attribute.strip().split(' ', 1)
        if len(parts) == 2:
            key = parts[0]
            # Remove quotes and extra whitespace
            value = parts[1].replace('"', '').strip()
            attrs[key] = value
    return attrs

def extract_targets(regulation_file):
    """
    Read the regulation file (downloaded from SGD) and extract the target gene names.
    Assumes that comment lines start with "!" and that the first non-comment line is a header.
    The 'Target' column is used (assumed to be named "Target").
    """
    targets = set()
    with open(regulation_file, 'r') as fin:
        header = None
        target_index = 2  # default to column 3 (0-indexed) if header is not found
        for line in fin:
            if line.startswith("!"):
                continue
            line = line.strip()
            if not line:
                continue
            # Use the first non-comment line as header
            if header is None:
                header = line.split('\t')
                try:
                    target_index = header.index("Target")
                except ValueError:
                    # fallback to column index 2 if "Target" not found
                    target_index = 2
                continue
            parts = line.split('\t')
            if len(parts) <= target_index:
                continue
            gene = parts[target_index].strip()
            if gene:
                targets.add(gene)
    return targets

def process_gtf(gtf_file, targets, output_file):
    """
    Read the GTF file and write out a BED file for all gene entries whose gene name is in 'targets'.
    For each gene, compute a 500bp region centered on the TSS:
      - For plus strand: 450 bp upstream and 50 bp downstream of the TSS.
      - For minus strand: 50 bp upstream and 450 bp downstream of the TSS.
    Note that TSS is defined as:
      + strand: TSS = gtf_start - 1  (0-indexed)
      - strand: TSS = gtf_end - 1    (0-indexed)
    Output format (BED-style): 
      chrom, region_start, region_end, gene_name, ".", strand
    """
    with open(gtf_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if line.startswith("#"):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attributes = parts
            # Process only "gene" features
            if feature != "gene":
                continue
            attr_dict = parse_gtf_attributes(attributes)
            # Use gene_name if available; otherwise use gene_id
            gene_name = attr_dict.get("gene_name", attr_dict.get("gene_id", None))
            if gene_name is None:
                continue
            if gene_name not in targets:
                continue
            try:
                gtf_start = int(start)
                gtf_end   = int(end)
            except ValueError:
                continue

            if strand == "+":
                # For plus strand, TSS is at gtf_start (convert to 0-indexed)
                TSS = gtf_start - 1
                region_start = TSS - 450
                region_end   = TSS + 50
            elif strand == "-":
                # For minus strand, TSS is at gtf_end (convert to 0-indexed)
                TSS = gtf_end - 1
                # For minus strand, "upstream" relative to transcription is toward higher coordinates
                region_start = TSS - 50
                region_end   = TSS + 450
            else:
                continue

            # Ensure region_start is not negative.
            if region_start < 0:
                region_start = 0

            fout.write(f"chr{chrom}\t{region_start}\t{region_end}\t{gene_name}\t.\t{strand}\n")

def main(regulation_file, gtf_file, output_file):
    # Extract target gene names from the regulation file
    targets = extract_targets(regulation_file)
    print(f"Found {len(targets)} target genes.")
    # Process the GTF file to generate a BED file with 500bp regions around the TSS for these targets.
    process_gtf(gtf_file, targets, output_file)
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    # Update file paths as necessary
    # regulator = "SWI4"
    # regulator = "MET4"
    # regulator = "RPN4"
    regulator = "MSN4"

    regulation_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/{regulator}_targets/{regulator}_targets.txt"
    input_gtf = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf"
    output_bed = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/{regulator}_targets.bed"
    main(regulation_file, input_gtf, output_bed)
