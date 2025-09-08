#!/usr/bin/env python3

import pandas as pd
import sys
import os

def map_chromosome_to_roman(chromosome):
    roman_mapping = {
        'chromosome1': 'I',
        'chromosome2': 'II',
        'chromosome3': 'III',
        'chromosome4': 'IV',
        'chromosome5': 'V',
        'chromosome6': 'VI',
        'chromosome7': 'VII',
        'chromosome8': 'VIII',
        'chromosome9': 'IX',
        'chromosome10': 'X',
        'chromosome11': 'XI',
        'chromosome12': 'XII',
        'chromosome13': 'XIII',
        'chromosome14': 'XIV',
        'chromosome15': 'XV',
        'chromosome16': 'XVI',
    }
    return roman_mapping.get(chromosome, chromosome)

def validate_chromosomes(df, chrom_col, valid_chromosomes):
    unique_chroms = df[chrom_col].unique()
    invalid_chroms = set(unique_chroms) - set(valid_chromosomes)
    if invalid_chroms:
        print(f"ERROR: Found invalid chromosomes in column '{chrom_col}': {invalid_chroms}")
        sys.exit(1)

def chromosome_sort_key(chromosome):
    # Mapping for Roman numerals to integers
    roman_to_int = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6,
        'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12,
        'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16
    }
    original_chromosome = chromosome  # Keep original for debugging
    if chromosome.startswith('chromosome'):
        chrom = chromosome.replace('chromosome', '').strip()
    else:
        chrom = chromosome.strip()

    chrom = chrom.upper()
    if chrom.isdigit():
        return int(chrom)
    elif chrom in roman_to_int:
        return 1000 + roman_to_int[chrom]  # Offset for Roman numerals
    elif chrom == 'X':
        return 2000
    elif chrom == 'Y':
        return 2001
    elif chrom == 'MT':
        return 2002
    else:
        print(f"WARNING: Unknown chromosome '{original_chromosome}' encountered. Placing it at the end.")
        return 3000

def sort_eQTLs(df, chrom_col, pos_col):
    df['chrom_order'] = df[chrom_col].apply(chromosome_sort_key)
    df[pos_col] = pd.to_numeric(df[pos_col], errors='coerce')
    initial_len = len(df)
    df = df.dropna(subset=[pos_col])
    if len(df) < initial_len:
        print(f"WARNING: Dropped {initial_len - len(df)} rows due to invalid positions.")
    df = df.sort_values(by=['chrom_order', pos_col])
    df = df.drop(columns=['chrom_order'])
    return df

def write_vcf(df, output_vcf, chrom_col, pos_col, ref_col, alt_col, gene_col, vcf_type):
    with open(output_vcf, 'w') as vcf:
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write(f"##INFO=<ID=GENE,Number=1,Type=String,Description=\"Associated gene for {vcf_type} eQTL\">\n")
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        for idx, row in df.iterrows():
            chrom = str(row[chrom_col])
            try:
                pos = int(float(row[pos_col]))
            except ValueError:
                print(f"WARNING: Skipping row with invalid position at index {idx}")
                continue
            ref = str(row[ref_col])
            alt = str(row[alt_col])
            gene = str(row[gene_col])
            if not chrom or not pos or not ref or not alt or not gene:
                print(f"WARNING: Skipping row with missing data at index {idx}")
                continue
            info = f"GENE={gene}"
            if pos < 1:
                print(f"WARNING: Skipping row with invalid position {pos} at index {idx}")
                continue
            vcf_line = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n"
            vcf.write(vcf_line)

def process_file(input_file):
    print(f"\nProcessing input file: {input_file}")
    if not os.path.isfile(input_file):
        print(f"ERROR: Input file '{input_file}' does not exist.")
        return

    try:
        df = pd.read_csv(input_file, sep='\t', dtype=str)
        print(f"Successfully read input file with {len(df)} rows.")
    except Exception as e:
        print(f"ERROR: Failed to read input file '{input_file}'. Details: {e}")
        return

    # Expected columns from previous output:
    # pos_chrom, pos_pos, pos_ref, pos_alt, pos_gene,
    # neg_chrom, neg_pos, neg_ref, neg_alt, neg_gene
    required_columns = [
        'pos_chrom', 'pos_pos', 'pos_ref', 'pos_alt', 'pos_gene',
        'neg_chrom', 'neg_pos', 'neg_ref', 'neg_alt', 'neg_gene'
    ]
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"ERROR: Missing required columns in input file '{input_file}': {missing_cols}")
        return

    print(f"Number of rows before filtering: {len(df)}")
    df = df[(df['pos_chrom'] != 'Mito') & (df['neg_chrom'] != 'Mito')]
    print(f"Number of rows after filtering: {len(df)}")

    # Explicit valid chromosomes (note the input uses 'chromosomeN' format)
    valid_chromosomes = ["chromosome" + str(i) for i in range(1, 17)] + ['X', 'Y', 'MT']

    print("Validating chromosome names for positive eQTLs...")
    validate_chromosomes(df, 'pos_chrom', valid_chromosomes)
    print("Validating chromosome names for negative eQTLs...")
    validate_chromosomes(df, 'neg_chrom', valid_chromosomes)

    # Process positive eQTLs
    positives = df[['pos_chrom', 'pos_pos', 'pos_ref', 'pos_alt', 'pos_gene']].drop_duplicates()
    positives = sort_eQTLs(positives, 'pos_chrom', 'pos_pos')
    positives['pos_chrom'] = positives['pos_chrom'].apply(map_chromosome_to_roman)
    print(f"Extracted {len(positives)} unique positive eQTLs.")

    # Process negative eQTLs
    negatives = df[['neg_chrom', 'neg_pos', 'neg_ref', 'neg_alt', 'neg_gene']].drop_duplicates()
    negatives = sort_eQTLs(negatives, 'neg_chrom', 'neg_pos')
    negatives['neg_chrom'] = negatives['neg_chrom'].apply(map_chromosome_to_roman)
    print(f"Extracted {len(negatives)} unique negative eQTLs.")

    base = os.path.splitext(os.path.basename(input_file))[0]
    positive_vcf = f"results/{base}.positive.vcf"
    negative_vcf = f"results/{base}.negative.vcf"

    print(f"Writing Positive eQTLs VCF to '{positive_vcf}'...")
    write_vcf(
        df=positives,
        output_vcf=positive_vcf,
        chrom_col='pos_chrom',
        pos_col='pos_pos',
        ref_col='pos_ref',
        alt_col='pos_alt',
        gene_col='pos_gene',
        vcf_type='POSITIVE'
    )
    print(f"Successfully wrote Positive VCF: '{positive_vcf}'.")

    print(f"Writing Negative eQTLs VCF to '{negative_vcf}'...")
    write_vcf(
        df=negatives,
        output_vcf=negative_vcf,
        chrom_col='neg_chrom',
        pos_col='neg_pos',
        ref_col='neg_ref',
        alt_col='neg_alt',
        gene_col='neg_gene',
        vcf_type='NEGATIVE'
    )
    print(f"Successfully wrote Negative VCF: '{negative_vcf}'.")

def main():
    os.makedirs("results", exist_ok=True)
    # Hard-coded input files from the previous step output.
    input_files = []
    for i in range(1, 11):
        input_files.append(f"results/negative_eqtls_set{i}.tsv")
    
    # Process each negative set file.
    for input_file in input_files:
        process_file(input_file)

if __name__ == "__main__":
    main()
