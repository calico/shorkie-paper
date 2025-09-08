#!/usr/bin/env python3
import pandas as pd
from Bio import SeqIO
import sys
import os

# Input files (adjust these file names/paths as needed)
tsv_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL_kita_etal/fix/selected_eQTL/intersected_CIS.tsv"
fasta_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask"
gtf_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf"
output_tsv = "results/output_pos_sequences.tsv"


os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
# Chromosome mapping dictionary: maps names in TSV (or numeric strings) to the names used in your GTF/FASTA.
chrom_mapping = {
    "chromosome1": "chrI",
    "chromosome2": "chrII",
    "chromosome3": "chrIII",
    "chromosome4": "chrIV",
    "chromosome5": "chrV",
    "chromosome6": "chrVI",
    "chromosome7": "chrVII",
    "chromosome8": "chrVIII",
    "chromosome9": "chrIX",
    "chromosome10": "chrX",
    "chromosome11": "chrXI",
    "chromosome12": "chrXII",
    "chromosome13": "chrXIII",
    "chromosome14": "chrXIV",
    "chromosome15": "chrXV",
    "chromosome16": "chrXVI",
    "1": "chrI",
    "2": "chrII",
    "3": "chrIII",
    "4": "chrIV",
    "5": "chrV",
    "6": "chrVI",
    "7": "chrVII",
    "8": "chrVIII",
    "9": "chrIX",
    "10": "chrX",
    "11": "chrXI",
    "12": "chrXII",
    "13": "chrXIII",
    "14": "chrXIV",
    "15": "chrXV",
    "16": "chrXVI",
    1: "chrI",
    2: "chrII",
    3: "chrIII",
    4: "chrIV",
    5: "chrV",
    6: "chrVI",
    7: "chrVII",
    8: "chrVIII",
    9: "chrIX",
    10: "chrX",
    11: "chrXI",
    12: "chrXII",
    13: "chrXIII",
    14: "chrXIV",
    15: "chrXV",
    16: "chrXVI",
}

####################################
# 1. Read the eQTL TSV file
####################################
eqtls = pd.read_csv(tsv_file, sep="\t")

####################################
# 2. Load the yeast reference genome
####################################
# Build a dictionary: key = chromosome id, value = sequence (a Bio.Seq object)
genome_dict = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    genome_dict[record.id] = record.seq

####################################
# 3. Parse the yeast GTF file to extract gene TSS information
####################################
# We create a dictionary with key: (chromosome, gene_name) --> (TSS, strand)
gene_info = {}
with open(gtf_file) as gtf:
    for line in gtf:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 9:
            continue
        seqname, source, feature, start, end, score, strand, frame, attributes = fields
        if feature != "gene":
            continue
        start = int(start)
        end = int(end)
        attr_dict = {}
        for attr in attributes.split(";"):
            attr = attr.strip()
            if not attr:
                continue
            if " " in attr:
                key, value = attr.split(" ", 1)
                value = value.replace("\"", "").strip()
                attr_dict[key] = value
        gene_name = attr_dict.get("gene_id") or attr_dict.get("gene_name")
        if gene_name is None:
            continue

        # Determine the TSS (for '+' strand, use start; for '-' strand, use end)
        tss = start if strand == "+" else end

        # Map the chromosome name if needed
        mapped_seqname = chrom_mapping.get(seqname, seqname)
        gene_info[(mapped_seqname, gene_name)] = (tss, strand)

print(f"Loaded {len(gene_info)} gene TSS entries from GTF.")
####################################
# 4. Define fixed flanking sequences to be appended
####################################
upstream_flank = "TGCATTTTTTTCACATC"
downstream_flank = "GGTTACGGCTGTT"

####################################
# 5. Process each eQTL and build the output rows
####################################
output_rows = []
for idx, row in eqtls.iterrows():
    # Map the chromosome names for the eQTL and gene.
    raw_chrom = row["Chr"]  # e.g. "chromosome16"
    chrom = chrom_mapping.get(raw_chrom, raw_chrom)
    try:
        eqtl_pos = int(row["ChrPos"])  # 1-indexed; assumed to be the leftmost base of the ref allele.
    except ValueError:
        print(f"Invalid ChrPos at row {idx}. Skipping.", file=sys.stderr)
        continue

    # Process reference allele (which might be more than 1 nucleotide)
    ref_allele = str(row["Reference"]).strip()
    L_ref = len(ref_allele)
    
    # For the alternate allele, split by comma and take the first one.
    alt_field = str(row["Alternate"]).strip()
    alt_allele = alt_field.split(",")[0].strip()
    L_alt = len(alt_allele)
    
    # For gene chromosome, use mapping as well.
    raw_gene_chr = str(row["Chr"])
    gene_chr = chrom_mapping.get(raw_gene_chr, raw_gene_chr)
    gene = row["#Gene"]
    print(f"raw_gene_chr: {raw_gene_chr}; gene_chr: {gene_chr}; gene: {gene}")

    # Lookup gene TSS using (gene_chr, gene)
    tss, strand = None, None
    if (gene_chr, gene) in gene_info:
        tss, strand = gene_info[(gene_chr, gene)]
    else:
        print(f"Gene {gene} on {raw_gene_chr} (mapped to {gene_chr}) not found in GTF. Marking group as outside_window.", file=sys.stderr)
        strand = None

    # Determine group membership (within 500bp upstream of TSS) if gene info is available
    if tss is not None and strand is not None:
    #     if strand == "+":
    #         # group = "upstream_500" if (eqtl_pos < tss and (tss - eqtl_pos) <= 500) else "outside_window"
    #         group = "upstream_500" if (eqtl_pos < tss and (tss - eqtl_pos) <= 500) else "outside_window"
    #         group 
    #     elif strand == "-":
    #         group = "upstream_500" if (eqtl_pos > tss and (eqtl_pos - tss) <= 500) else "outside_window"
    # else:
    #     group = "outside_window"  # default

        group = abs(eqtl_pos - tss)  # for testing
    else:
        group = -1

    # Check that the chromosome exists in the reference genome
    if chrom not in genome_dict:
        print(f"Chromosome {raw_chrom} (mapped to {chrom}) not found in reference genome. Skipping eQTL at row {idx}.", file=sys.stderr)
        continue
    chrom_seq = genome_dict[chrom]

    # Calculate the center of the eQTL in the genome.
    # We assume ChrPos is the leftmost position (1-indexed) of the reference allele.
    # Compute the center using the reference allele length.
    eqtl_center = eqtl_pos + (L_ref // 2)  # still in 1-indexed coordinates
    center_index = eqtl_center - 1         # convert to 0-indexed

    # Define the 80bp extraction window around the center.
    extraction_start = center_index - 40
    extraction_end = center_index + 40
    if extraction_start < 0 or extraction_end > len(chrom_seq):
        print(f"eQTL at {chrom}:{eqtl_pos} (center {eqtl_center}) is too close to the boundary for extraction. Skipping.", file=sys.stderr)
        continue

    extracted_seq = str(chrom_seq[extraction_start:extraction_end]).upper()
    if len(extracted_seq) != 80:
        print(f"Extracted sequence length for {chrom}:{eqtl_pos} is not 80 bp. Skipping.", file=sys.stderr)
        continue

    # Determine where the reference allele should appear in the extracted sequence.
    allele_start_in_extracted = 40 - (L_ref // 2)
    allele_from_genome = extracted_seq[allele_start_in_extracted : allele_start_in_extracted + L_ref]
    if allele_from_genome.upper() != ref_allele.upper():
        print(f"Reference nucleotide mismatch at {chrom}:{eqtl_pos}: genome has '{allele_from_genome}' but TSV has '{ref_allele}'.", file=sys.stderr)

    # Create the final reference sequence by appending the fixed flanks.
    final_ref_seq = (upstream_flank + extracted_seq + downstream_flank).upper()

    # Create the alternate sequence by replacing the reference allele substring with the alternate allele.
    alt_extracted_seq = extracted_seq[:allele_start_in_extracted] + alt_allele.upper() + extracted_seq[allele_start_in_extracted + L_ref:]
    final_alt_seq = (upstream_flank + alt_extracted_seq + downstream_flank).upper()

    output_rows.append({
        "Gene": row["#Gene"],
        "Chr": chrom,
        "ChrPos": eqtl_pos,
        "tss_dist": group,
        "final_ref_seq": final_ref_seq,
        "final_alt_seq": final_alt_seq
    })

####################################
# 6. Write output as a TSV file
####################################
output_df = pd.DataFrame(output_rows)
output_df.to_csv(output_tsv, sep="\t", index=False)

print("Sequence extraction and TSV file writing completed.")
