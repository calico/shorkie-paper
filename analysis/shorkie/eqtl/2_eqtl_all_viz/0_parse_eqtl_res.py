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
# Filename Parsing
# -------------------------------
def parse_filename(path):
    """
    Expect basename like:
      avg_logSED_scores_YAL016C-B_chrI_124373.txt
    """
    base = os.path.basename(path).replace('.txt','')
    parts = base.split('_')
    # parts = ['avg','logSED','scores','YAL016C-B','chrI','124373']
    gene   = parts[2]
    chrom  = parts[3]
    pos    = int(parts[4])
    return gene, chrom, pos

# -------------------------------
# Parse eQTL Score Files
# -------------------------------
def parse_eqtl_files(indir, tss_data, label_type="Positive"):
    records = []
    pattern = os.path.join(indir, "logSED_scores_*.txt")
    for fn in glob.glob(pattern):
        gene, chrom, pos = parse_filename(fn)
        with open(fn) as f:
            parts = f.readline().strip().split()
        logSED_agg = float(parts[1])
        logSED_avg = float(parts[2])

        distance = calculate_distance(chrom, pos, gene, tss_data)
        records.append({
            'Position_Gene': f"{chrom}:{pos}_{gene}",
            'logSED_agg': logSED_agg,
            'logSED_avg': logSED_avg,
            'distance': distance,
            'label': 1 if label_type == "Positive" else 0,
            'label_type': label_type,
            'Chr': chrom,
            'ChrPos': pos,
            'gene': gene,
        })
    df = pd.DataFrame(records)
    print(f"Read {len(df)} records from {pattern}")
    return df

# -------------------------------
# Main
# -------------------------------

def main():
    root = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
    gtf = os.path.join(root, 'data', 'eQTL', 'neg_eQTLS', 'GCA_000146045_2.59.gtf')
    exp_root = f"{root}/experiments/SUM_data_process/"
    tss_data = parse_gtf_for_tss(gtf)

    experiments = ["kita_etal", "caudal_etal"]
    for exp in experiments:
        for negset in range(1, 5):

            outdir = f"results/negset_{negset}"
            os.makedirs(outdir, exist_ok=True)

            pos_indir = os.path.join(
                exp_root, f"eQTL_{exp}", "positive", "results", "test"
            )
            neg_indir = os.path.join(
                exp_root, f"eQTL_{exp}", "negative", "results", f"test_{negset}"
            )

            pos_df = parse_eqtl_files(pos_indir, tss_data, label_type="Positive")
            neg_df = parse_eqtl_files(neg_indir, tss_data, label_type="Negative")
            combined = pd.concat([pos_df, neg_df], ignore_index=True)

            out_file = os.path.join(outdir, f"{exp}_scores.tsv")
            combined.to_csv(out_file, sep='\t', index=False)
            print(f"Wrote {len(combined)} rows (with distances) to {out_file}")

if __name__ == '__main__':
    main()
