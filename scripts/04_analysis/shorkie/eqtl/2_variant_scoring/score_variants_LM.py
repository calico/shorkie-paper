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
    parser = argparse.ArgumentParser(description="Score variants using Shorkie_LM")
    parser.add_argument("--variants_csv", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--params_file", required=True)
    parser.add_argument("--fasta_file", required=True)
    parser.add_argument("--output_file", default="variant_scores.tsv")
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
    print(f"Loading variants from {args.variants_csv}...")
    # NOTE: Renganaath input is CSV, not TSV
    df = pd.read_csv(args.variants_csv, sep=",", dtype=str)
    
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
            # Columns: gene, chr, pos, ref, alt
            gene = row["gene"]
            chrom = row["chr"] 
            
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
            
            pos = int(row["pos"])
            
            # Use actual_ref fetched from FASTA as the reference allele
            pos_0based = pos - 1
            
            actual_ref = genome_open.fetch(chrom_str, pos_0based, pos_0based + 1).upper()
            
            # Verify against input CSV ref allele (sanity check)
            input_ref = str(row["ref"]).strip()
            if not pd.isna(input_ref) and input_ref.upper() != actual_ref:
                print(f"WARNING: Reference allele mismatch at {chrom_str}:{pos}_{gene}. CSV Ref: {input_ref} vs Genome extracted: {actual_ref}. Using Fasta.", file=sys.stderr)
            else:
                 print(f"DEBUG: Reference allele match at {chrom_str}:{pos}_{gene}. Ref: {actual_ref}", file=sys.stderr)
            
            alt = str(row["alt"]).strip()
            
            if pd.isna(alt):
                print(f"SKIPPED: Alt is NA (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                continue
                
            alleles = [a.strip() for a in alt.split(",") if a.strip()]
            
            for alt_allele in alleles:
                if len(alt_allele) != 1: 
                    print(f"SKIPPED: Alt length != 1 ({alt_allele}) (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                    continue
                if len(actual_ref) != 1: 
                    print(f"SKIPPED: Ref length != 1 ({actual_ref}) (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                    continue

                if len(input_ref) != 1:
                     print(f"SKIPPED: Input Ref length != 1 ({input_ref}) (Variant: {chrom_str}:{pos}_{gene}) - likely an Indel", file=sys.stderr)
                     continue
                    
                x_input = process_sequence_with_mask(genome_open, chrom_str, pos_0based, pos_0based+1, args.seq_len, ref_allele=actual_ref)
                x_input = x_input[np.newaxis, ...] 
                
                # Use call instead of predict for speed in loop
                # Output is a tensor, convert to numpy
                preds = keras_model(x_input, training=False)
                probs = preds.numpy()[0, args.seq_len // 2, :]
                
                nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                
                if actual_ref not in nuc_map or alt_allele not in nuc_map:
                    print(f"SKIPPED: Allele not in map ({actual_ref}->{alt_allele}) (Variant: {chrom_str}:{pos}_{gene})", file=sys.stderr)
                    continue
                    
                p_ref = probs[nuc_map[actual_ref]]
                p_alt = probs[nuc_map[alt_allele]]
                
                epsilon = 1e-12
                llr = np.log((p_alt + epsilon) / (p_ref + epsilon))
                score_diff = p_alt - p_ref
                
                results.append({
                    "Gene": gene,
                    "Chrom": chrom_str,
                    "Pos": pos,
                    "Ref": actual_ref,
                    "Alt": alt_allele,
                    "Prob_Ref": p_ref,
                    "Prob_Alt": p_alt,
                    "LLR": llr,
                    "Score_Diff": score_diff
                })
                
        except Exception as e:
            print(f"SKIPPED: Error processing row {idx} (Variant: {chrom_str}:{pos}_{gene}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Saved results to {args.output_file}")
    
    genome_open.close()

if __name__ == "__main__":
    main()
