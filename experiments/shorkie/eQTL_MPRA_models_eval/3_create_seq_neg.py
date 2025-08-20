#!/usr/bin/env python3
import pandas as pd
from Bio import SeqIO
import sys
import os

negsets = [1, 2, 3, 4]
os.makedirs("results", exist_ok=True) 

for negset in negsets:
    # Input files (adjust these file names/paths as needed)
    tsv_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/eQTL_pseudo_negatives/results/negative_eqtls_set{negset}.tsv"
    fasta_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask"
    gtf_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf"
    output_tsv = f"results/output_neg_sequences_{negset}.tsv"

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
    # 1. Read the new TSV file with neg columns
    ####################################
    eqtls = pd.read_csv(tsv_file, sep="\t")

    ####################################
    # 2. Load the yeast reference genome
    ####################################
    genome_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome_dict[record.id] = record.seq

    ####################################
    # 3. Parse the yeast GTF file to extract gene TSS information
    ####################################
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

    ####################################
    # 4. Define fixed flanking sequences to be appended
    ####################################
    upstream_flank = "TGCATTTTTTTCACATC"
    downstream_flank = "GGTTACGGCTGTT"

    ####################################
    # 5. Process each neg eQTL and build the output rows
    ####################################
    output_rows = []
    for idx, row in eqtls.iterrows():
        raw_chrom = row["neg_chrom"]  # e.g. "chromosome14"
        chrom = chrom_mapping.get(raw_chrom, raw_chrom)
        
        try:
            eqtl_pos = int(row["neg_pos"])  # 1-indexed position (leftmost base of the ref allele)
        except ValueError:
            print(f"Invalid neg_pos at row {idx}. Skipping.", file=sys.stderr)
            continue

        ref_allele = str(row["neg_ref"]).strip()
        L_ref = len(ref_allele)
        
        alt_field = str(row["neg_alt"]).strip()
        alt_allele = alt_field.split(",")[0].strip()
        L_alt = len(alt_allele)
        
        gene = row["neg_gene"]
        gene_chr = chrom_mapping.get(raw_chrom, raw_chrom)
        
        tss, strand = None, None
        if (gene_chr, gene) in gene_info:
            tss, strand = gene_info[(gene_chr, gene)]
        else:
            print(f"Gene {gene} on {raw_chrom} (mapped to {gene_chr}) not found in GTF. Setting tss_dist to -1.", file=sys.stderr)

        # Replace the original group classification with tss_dist (absolute distance)
        if tss is not None:
            tss_dist = abs(eqtl_pos - tss)
        else:
            tss_dist = -1

        if chrom not in genome_dict:
            print(f"Chromosome {raw_chrom} (mapped to {chrom}) not found in reference genome. Skipping neg eQTL at row {idx}.", file=sys.stderr)
            continue
        chrom_seq = genome_dict[chrom]

        eqtl_center = eqtl_pos + (L_ref // 2)
        center_index = eqtl_center - 1

        extraction_start = center_index - 40
        extraction_end = center_index + 40
        if extraction_start < 0 or extraction_end > len(chrom_seq):
            print(f"Neg eQTL at {chrom}:{eqtl_pos} (center {eqtl_center}) is too close to the boundary for extraction. Skipping.", file=sys.stderr)
            continue

        extracted_seq = str(chrom_seq[extraction_start:extraction_end]).upper()
        if len(extracted_seq) != 80:
            print(f"Extracted sequence length for {chrom}:{eqtl_pos} is not 80 bp. Skipping.", file=sys.stderr)
            continue

        allele_start_in_extracted = 40 - (L_ref // 2)
        allele_from_genome = extracted_seq[allele_start_in_extracted : allele_start_in_extracted + L_ref]
        if allele_from_genome.upper() != ref_allele.upper():
            print(f"Reference nucleotide mismatch at {chrom}:{eqtl_pos}: genome has '{allele_from_genome}' but TSV has '{ref_allele}'.", file=sys.stderr)

        final_ref_seq = (upstream_flank + extracted_seq + downstream_flank).upper()
        alt_extracted_seq = extracted_seq[:allele_start_in_extracted] + alt_allele.upper() + extracted_seq[allele_start_in_extracted + L_ref:]
        final_alt_seq = (upstream_flank + alt_extracted_seq + downstream_flank).upper()

        Gene = row["neg_gene"]
        output_rows.append({
            "Gene": Gene,
            "Chr": chrom,
            "ChrPos": eqtl_pos,
            "tss_dist": tss_dist,
            "final_ref_seq": final_ref_seq,
            "final_alt_seq": final_alt_seq
        })


    ####################################
    # 6. Write output as a TSV file
    ####################################
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_tsv, sep="\t", index=False)

    print("Negative sequence extraction and TSV file writing completed for negset", negset)
