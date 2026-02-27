#!/usr/bin/env python3
import os
import glob
import argparse
import re
import pandas as pd

# -------------------------------
# Chromosome Mapping
# -------------------------------

def map_chromosome_to_roman(chromosome):
    roman_mapping = {
        'chromosome1':'chrI','chromosome2':'chrII','chromosome3':'chrIII',
        'chromosome4':'chrIV','chromosome5':'chrV','chromosome6':'chrVI',
        'chromosome7':'chrVII','chromosome8':'chrVIII','chromosome9':'chrIX',
        'chromosome10':'chrX','chromosome11':'chrXI','chromosome12':'chrXII',
        'chromosome13':'chrXIII','chromosome14':'chrXIV','chromosome15':'chrXV','chromosome16':'chrXVI',
        'I':'chrI','II':'chrII','III':'chrIII','IV':'chrIV','V':'chrV','VI':'chrVI','VII':'chrVII',
        'VIII':'chrVIII','IX':'chrIX','X':'chrX','XI':'chrXI','XII':'chrXII','XIII':'chrXIII','XIV':'chrXIV','XV':'chrXV','XVI':'chrXVI'
    }
    return roman_mapping.get(chromosome, chromosome)

# -------------------------------
# Parse GTF for TSS
# -------------------------------

def parse_gtf_for_tss(gtf_file):
    tss_data = {}
    with open(gtf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 9 or cols[2].lower() != 'gene':
                continue
            chrom = cols[0]
            start, end = int(cols[3]), int(cols[4])
            strand = cols[6]
            attr = cols[8]
            m = re.search(r'gene_id\s+"([^"]+)"', attr)
            if not m:
                continue
            gid = m.group(1)
            tss_pos = start if strand == '+' else end
            tss_data[gid] = {'chrom': chrom, 'tss': tss_pos}
    return tss_data

# -------------------------------
# Distance Calculation
# -------------------------------
def calculate_distance(chrom, pos, gene, tss_data):
    info = tss_data.get(gene)
    if not info:
        return None
    gene_chrom = map_chromosome_to_roman(info['chrom'])
    if gene_chrom != chrom:
        return None
    return abs(pos - info['tss'])

# -------------------------------
# Parse Shorkie Files
# -------------------------------
def parse_shorkie_file(filepath, tss_data, label_type="Positive"):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, sep='\t')
    
    records = []
    for _, row in df.iterrows():
        gene = row['Gene']
        chrom = row['Chrom']
        pos = int(row['Pos'])
        
        distance = calculate_distance(chrom, pos, gene, tss_data)
        
        record = {
            'Position_Gene': f"{chrom}:{pos}_{gene}",
            'distance': distance,
            'label': 1 if label_type == "Positive" else 0,
            'label_type': label_type,
            'Chr': chrom,
            'ChrPos': pos,
            'gene': gene,
        }
        
        if 'LLR' in row:
            record['LLR'] = row['LLR']
        if 'logSED_agg' in row:
            record['logSED_agg'] = row['logSED_agg']
            
        records.append(record)
    
    out_df = pd.DataFrame(records)
    print(f"Read {len(out_df)} records from {filepath}")
    return out_df

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse shorkie results into unified TSVs.")
    parser.add_argument("--root_dir", required=True, help="Root path containing revision_experiments/eQTL/...")
    parser.add_argument("--output_dir", required=True, help="Directory to save output TSVs")
    parser.add_argument("--gtf_file", required=True, help="Path to GTF file for TSS extraction")
    args = parser.parse_args()

    # root = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    gtf = args.gtf_file
    tss_data = parse_gtf_for_tss(gtf)

    exps = [
        "caudal_etal_Shorkie", "caudal_etal_Shorkie_LM", "caudal_etal_Shorkie_Random_Init",
        "kita_etal_Shorkie", "kita_etal_Shorkie_LM", "kita_etal_Shorkie_Random_Init",
        "Renganaath_etal_Shorkie", "Renganaath_etal_Shorkie_LM", "Renganaath_etal_Shorkie_Random_Init"
    ]

    out_base = args.output_dir

    for exp in exps:
        exp_path = os.path.join(args.root_dir, 'revision_experiments', 'eQTL', exp)
        is_lm = "Shorkie_LM" in exp
        is_random_init = "Random_Init" in exp

        # Decide on the prefix for positive files based on exp
        if "caudal" in exp.lower():
            pos_prefix = "caudal"
        else:
            pos_prefix = "kita"
        
        for negset in range(1, 5):
            outdir = os.path.join(out_base, f"negset_{negset}")
            os.makedirs(outdir, exist_ok=True)

            if is_lm:
                # Shorkie LM has positive/variant_scores.tsv and negative/variant_scores_set{negset}.tsv
                pos_file = os.path.join(exp_path, "positive", "variant_scores.tsv")
                neg_file = os.path.join(exp_path, "negative", f"variant_scores_set{negset}.tsv")
                
            elif is_random_init:
                pos_file = os.path.join(exp_path, "positive", "results", "variant_scores_logSED.tsv")
                if exp == "kita_etal_Shorkie_Random_Init":
                    neg_file = os.path.join(exp_path, "negative", "results", f"variant_scores_logSED_set{negset}.tsv")
                else:
                    neg_file = os.path.join(exp_path, "negative", "results", f"variant_scores_set{negset}.tsv")
            else:
                if exp == "Renganaath_etal_Shorkie":
                    pos_file = os.path.join(exp_path, "positive", "results", "variant_scores_logSED.tsv")
                else:
                    pos_file = os.path.join(exp_path, "positive", "results", f"{pos_prefix}_positive_shorkie_scores.csv")
                neg_file = os.path.join(exp_path, "negative", "results", f"variant_scores_set{negset}.tsv")

            pos_df = parse_shorkie_file(pos_file, tss_data, label_type="Positive")
            neg_df = parse_shorkie_file(neg_file, tss_data, label_type="Negative")

            combined = pd.concat([pos_df, neg_df], ignore_index=True)

            out_file = os.path.join(outdir, f"{exp}_scores.tsv")
            combined.to_csv(out_file, sep='\t', index=False)
            print(f"Wrote {len(combined)} rows (with distances) to {out_file}")

if __name__ == '__main__':
    main()
