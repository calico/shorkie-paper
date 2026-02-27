#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
from baskerville import seqnn, dna
from tqdm import tqdm

# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="Score negative variants using Shorkie_LM")
    parser.add_argument("--variants_tsv", required=True, help="Path to negative variants TSV")
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--params_file", required=True)
    parser.add_argument("--fasta_file", required=True)
    parser.add_argument("--output_file", required=True, help="Output file path")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of variants for testing")
    return parser.parse_args()

def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # Extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna.dna_1hot(seq_dna)
    return seq_1hot

def process_sequence_with_mask(genome_open, chrm, start, end, seq_len=16384, ref_allele=None):
    seq_len_actual = end - start
    
    # Pad sequence to input window size
    # We want the variant (at 'start') to be exactly at 'center_idx' (seq_len // 2)
    center_idx = seq_len // 2
    
    # Calculate padding needed on the left to place 'start' at 'center_idx'
    pad_left = center_idx
    
    # Calculate final window coordinates
    window_start = start - pad_left
    window_end = window_start + seq_len

    seq_1hot = make_seq_1hot(genome_open, chrm, window_start, window_end, seq_len)
    seq_1hot = seq_1hot.astype("float32")
    
    # Verify Reference Allele BEFORE masking
    if ref_allele:
        # Get the one-hot vector at the center position
        center_vec = seq_1hot[center_idx, :4] # Ensure we only look at first 4 chars just in case
        
        # Simple argmax to get the base index
        if np.sum(center_vec) == 0:
             # N or zero-ed out?
             extracted_base = "N"
        else:
             base_idx = np.argmax(center_vec)
             # Map back to char
             inv_nuc_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
             extracted_base = inv_nuc_map.get(base_idx, "N")
        
        if extracted_base != ref_allele:
            raise ValueError(f"Reference Mismatch at {chrm}:{start}! Expected {ref_allele}, found {extracted_base} at center index {center_idx}.")

    # Mask center position
    seq_1hot[center_idx, :] = 0.0
    
    num_species = 165
    padding = np.zeros((seq_len, num_species + 1), dtype="float32")
    x_new = np.concatenate([seq_1hot, padding], axis=-1)
    
    # Set the 114th column to 1 
    x_new[:, 114] = 1.0
    
    return x_new

def main():
    args = parse_args()
    
    # Load Params
    with open(args.params_file) as pf:
        params = json.load(pf)
    params_model = params["model"]
    num_species = 165
    params_model["num_features"] = num_species + 5
    
    # Load Model
    print(f"Loading model from {args.model_file}...")
    model = seqnn.SeqNN(params_model)
    model.restore(args.model_file, trunk=False, by_name=False)
    keras_model = model.model
    
    # Load Variants
    print(f"Loading variants from {args.variants_tsv}...")
    df = pd.read_csv(args.variants_tsv, sep="\t", dtype=str)
    
    genome_open = pysam.Fastafile(args.fasta_file)
    
    results = []
    
    print(f"Processing variants...")
    total = len(df)
    if args.limit:
        total = min(total, args.limit)
            
    for idx, row in tqdm(df.iterrows(), total=total):
        if args.limit and len(results) >= args.limit:
            break
            
        try:
             # Columns: neg_chrom, neg_pos, neg_ref, neg_alt, neg_gene
            gene = row["neg_gene"]
            chrom = row["neg_chrom"] 
            
            # Map chromosome names
            chrom_map = {
                "chromosome1": "chrI", "chromosome2": "chrII", "chromosome3": "chrIII", "chromosome4": "chrIV",
                "chromosome5": "chrV", "chromosome6": "chrVI", "chromosome7": "chrVII", "chromosome8": "chrVIII",
                "chromosome9": "chrIX", "chromosome10": "chrX", "chromosome11": "chrXI", "chromosome12": "chrXII",
                "chromosome13": "chrXIII", "chromosome14": "chrXIV", "chromosome15": "chrXV", "chromosome16": "chrXVI"
            }
            if chrom in chrom_map:
                chrom_str = chrom_map[chrom]
            elif chrom.replace("chromosome", "") in [str(i) for i in range(1, 17)]:
                 chrom_str = chrom 
            else:
                 chrom_str = chrom 
                 
            pos = int(row["neg_pos"])
            
            # Use actual_ref fetched from FASTA as the reference allele
            pos_0based = pos - 1
            
            actual_ref = genome_open.fetch(chrom_str, pos_0based, pos_0based + 1).upper()
            
            # Use columns from TSV
            ref_tsv = str(row["neg_ref"]).strip()
            alt_tsv = str(row["neg_alt"]).strip()
            
            # Check for indel in Ref
            if not pd.isna(ref_tsv) and len(ref_tsv) != 1:
                 print(f"SKIPPED: Ref length != 1 ({ref_tsv}) (Variant: {chrom_str}:{pos}_{gene}) - likely an Indel", file=sys.stderr)
                 continue
            
            if pd.isna(alt_tsv):
                print(f"SKIPPED: Alt is NA (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                continue
                
            alleles = [a.strip() for a in alt_tsv.split(",") if a.strip()]
            
            for alt in alleles:
                if len(alt) != 1: 
                    # Only score SNPs
                    print(f"SKIPPED: Alt length != 1 ({alt}) (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                    continue
                if len(actual_ref) != 1: 
                    print(f"SKIPPED: Ref length != 1 ({actual_ref}) (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                    continue

                if ref_tsv != actual_ref:
                    print(f"WARNING: Reference allele mismatch at {chrom_str}:{pos}_{gene}. CSV Ref: {ref_tsv} vs Genome extracted: {actual_ref}. Using Fasta.", file=sys.stderr)
                else:
                    print(f"DEBUG: Reference allele match at {chrom_str}:{pos}_{gene}. Ref: {actual_ref}", file=sys.stderr)
                    
                x_input = process_sequence_with_mask(genome_open, chrom_str, pos_0based, pos_0based+1, args.seq_len, ref_allele=actual_ref)
                x_input = x_input[np.newaxis, ...] 
                
                # Use call instead of predict for speed in loop
                # Output is a tensor, convert to numpy
                preds = keras_model(x_input, training=False)
                probs = preds.numpy()[0, args.seq_len // 2, :]
                
                nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                
                if actual_ref not in nuc_map or alt not in nuc_map:
                    print(f"SKIPPED: Allele not in map ({actual_ref}->{alt}) (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                    continue
                    
                p_ref = probs[nuc_map[actual_ref]]
                p_alt = probs[nuc_map[alt]]
                
                epsilon = 1e-12
                llr = np.log((p_alt + epsilon) / (p_ref + epsilon))
                score_diff = p_alt - p_ref
                
                results.append({
                    "Gene": gene,
                    "Chrom": chrom_str,
                    "Pos": pos,
                    "Ref_FASTA": actual_ref,
                    "Ref_TSV": ref_tsv,
                    "Alt": alt,
                    "Prob_Ref": p_ref,
                    "Prob_Alt": p_alt,
                    "LLR": llr,
                    "Score_Diff": score_diff,
                     # Optional: Keep original pos info to trace back?
                    "Pos_Gene": row["pos_gene"],
                    "Pos_Chrom": row["pos_chrom"],
                    "Pos_Pos": row["pos_pos"]
                })
                
        except Exception as e:
            print(f"SKIPPED: Error processing row {idx}: {e}", file=sys.stderr)
            continue

    # Create results directory if it doesn't exist
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Saved results to {args.output_file}")
    
    genome_open.close()

if __name__ == "__main__":
    main()
