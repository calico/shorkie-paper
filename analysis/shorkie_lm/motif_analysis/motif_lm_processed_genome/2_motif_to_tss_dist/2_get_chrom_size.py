#!/usr/bin/env python3
import os
import sys
import glob

def get_chromosome_sizes(fasta_file):
    sizes = {}
    with open(fasta_file, 'r') as file:
        chrom_name = None
        chrom_length = 0
        for line in file:
            line = line.strip()
            if not line:
                continue
            # New header indicates a new chromosome
            if line.startswith('>'):
                # If a previous chromosome was being processed, store its length.
                if chrom_name is not None:
                    sizes[chrom_name] = chrom_length
                # Extract the chromosome name (first word after '>')
                chrom_name = line[1:].split()[0]
                chrom_length = 0
            else:
                # Accumulate the sequence length (ignoring whitespace)
                chrom_length += len(line)
        # Save the last chromosome read from the file.
        if chrom_name is not None:
            sizes[chrom_name] = chrom_length
    return sizes

def main():
    # exp_species = "strains_select"   # or "schizosaccharomycetales"
    exp_species = "schizosaccharomycetales"

    directory = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{exp_species}_gtf/fasta"
    outfile = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/2_motif_to_tss_dist/results/{exp_species}_motif_hits/genome.chrom.sizes"
    
    # Search for files ending with .fa or .fasta (case insensitive)
    fasta_files = glob.glob(os.path.join(directory, "*.cleaned.fasta")) + glob.glob(os.path.join(directory, "*.cleaned.fa"))
    if not fasta_files:
        sys.exit("No FASTA files found in the specified directory.")

    with open(outfile, 'w') as out:
        # Process each FASTA file.
        for fasta_file in fasta_files:
            # Get the base file name without extension.
            base = os.path.basename(fasta_file)
            name, _ = os.path.splitext(base)
            # I want to remove ".cleaned" from the name
            name = name.replace(".cleaned", "")
            
            sizes = get_chromosome_sizes(fasta_file)
            # Output each chromosome's info: "filename_chromosome <tab> size"
            for chrom, size in sizes.items():
                print(f"{name}:{chrom}\t{size}")
                out.write(f"{name}:{chrom}\t{size}\n")

if __name__ == "__main__":
    main()
