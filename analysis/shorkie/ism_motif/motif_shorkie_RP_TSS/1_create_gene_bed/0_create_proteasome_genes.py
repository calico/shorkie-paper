#!/usr/bin/env python3
import sys
import re

def parse_attributes(attr_str):
    """
    Parse the GTF attribute string and return a dictionary of key/value pairs.
    """
    attrs = {}
    # Split on semicolon and strip whitespace
    for attribute in attr_str.strip().split(';'):
        if attribute.strip() == "":
            continue
        # Expect format key "value"
        parts = attribute.strip().split(' ', 1)
        if len(parts) == 2:
            key = parts[0]
            # Remove quotes and extra spaces
            value = parts[1].replace('"', '').strip()
            attrs[key] = value
    return attrs

def main(gtf_file, output_file):
    # List of proteasome subunit genes: 11 CP and 17 RP genes.
    cp_genes = ['PRE1', 'PRE2', 'PRE3', 'PRE4', 'PRE5', 'PRE6', 'PRE8', 'PRE9', 'PRE10', 'PUP1', 'PUP2']
    rp_genes = ['RPT1', 'RPT2', 'RPT3', 'RPT4', 'RPT5', 'RPT6', 'RPN1', 'RPN2', 'RPN3', 'RPN5', 'RPN6', 'RPN7', 'RPN8', 'RPN9', 'RPN10', 'RPN11', 'RPN12']
    gene_set = set(cp_genes + rp_genes)

    with open(gtf_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            # Skip header/comment lines
            if line.startswith("#"):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            chrom, source, feature, start, end, score, strand, frame, attributes = fields
            # Process only gene entries (adjust if needed)
            if feature != "gene":
                continue

            attr_dict = parse_attributes(attributes)
            # Try "gene_name" first; fallback to "gene_id" if not available
            gene_name = attr_dict.get("gene_name", attr_dict.get("gene_id", None))
            if gene_name is None:
                continue

            if gene_name not in gene_set:
                continue

            # Convert start and end to integers
            try:
                gtf_start = int(start)
                gtf_end = int(end)
            except ValueError:
                continue

            # Determine the TSS based on strand:
            # For + strand, TSS is at gtf_start (converted to 0-indexed by subtracting 1).
            # For - strand, TSS is at gtf_end (converted to 0-indexed by subtracting 1).
            if strand == "+":
                TSS = gtf_start - 1
                region_start = TSS - 450
                region_end   = TSS + 50
            elif strand == "-":
                TSS = gtf_end - 1
                # For minus strand, "upstream" (relative to transcription) is in the direction of increasing coordinates.
                region_start = TSS - 50
                region_end   = TSS + 450
            else:
                # Skip entries with unexpected strand information.
                continue

            # Make sure region_start is not negative.
            if region_start < 0:
                region_start = 0

            # Write out the BED entry: chrom, start, end, gene_name, score, strand.
            fout.write(f"{chrom}\t{region_start}\t{region_end}\t{gene_name}\t.\t{strand}\n")

if __name__ == "__main__":
    # Example usage; update paths as necessary.
    input_gtf = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.gtf"
    output_bed = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/Proteasome_genes.bed"
    main(input_gtf, output_bed)
