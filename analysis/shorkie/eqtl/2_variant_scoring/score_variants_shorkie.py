#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
from baskerville import seqnn
from baskerville import gene as bgene
from tqdm import tqdm
from yeast_helpers_selfsupervised import *

# Ensure we can import from local directory if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="Score variants using Shorkie (logSED) for Kita et al.")
    parser.add_argument("--variants_csv", required=True)
    
    # Model arguments matching viz_rnaseq_cov_ISM.py
    parser.add_argument("--params_file", required=True)
    parser.add_argument("--targets_file", required=True)
    parser.add_argument("--gtf_file", required=True)
    parser.add_argument("--fasta_file", required=True)
    
    parser.add_argument("--output_file", default="results/kita_positive_shorkie_scores.csv")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of variants for testing")
    parser.add_argument("--num_folds", type=int, default=8, help="Number of trained folds to ensemble.")
    
    return parser.parse_args()

def predict_tracks_ensemble(model_list, sequence):
    preds = [predict_tracks([model], sequence) for model in model_list]
    return np.mean(np.array(preds), axis=0)

def main():
    args = parse_args()
    
    # Load Params
    with open(args.params_file) as pf:
        params = json.load(pf)
    params_model = params["model"]
    num_species = 165
    params_model["num_features"] = num_species + 5
    
    # Load Targets
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    target_index = targets_df.index
    print("Targets track number:", len(target_index))

    # Load Models (Ensemble)
    models = []
    print(f"Loading {args.num_folds} models...")
    for fold in range(args.num_folds):
        fold_param = f"f{fold}c0"
        model_file = os.path.join(
            os.path.dirname(args.params_file),
            "train",
            fold_param,
            "train",
            "model_best.h5",
        )
        
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, trunk=False, by_name=False)        
        seqnn_model.build_slice(target_index)
        seqnn_model.build_ensemble(True, [0])
        models.append(seqnn_model)
    
    seqnn_model = models[0]
    
    # Load Genomic Resources
    print(f"Loading FASTA and GTF...")
    fasta_open = pysam.Fastafile(args.fasta_file)
    transcriptome = bgene.Transcriptome(args.gtf_file)
    
    # Load Variants
    print(f"Loading variants from {args.variants_csv}...")
    df = pd.read_csv(args.variants_csv, sep="\t", dtype=str)
    
    results = []
    
    print(f"Processing variants...")
    total = len(df)
    if args.limit:
        total = min(total, args.limit)
            
    for idx, row in tqdm(df.iterrows(), total=total):
        if args.limit and len(results) >= args.limit:
            break
            
        try:
            # Map columns for Kita et al.
            gene = row["#Gene"]
            pos = int(row["position"])
            ref_csv = str(row["Reference"]).strip()
            alt_csv = str(row["Alternate"]).strip()
            
            # Find Gene in Transcriptome to get chromosome
            gene_keys = [gkey for gkey in transcriptome.genes.keys() if gene in gkey]
            
            if len(gene_keys) == 0:
                print(f"SKIPPED: Gene {gene} not found in transcriptome (Variant: {pos})", file=sys.stderr)
                continue
            
            gene_obj = transcriptome.genes[gene_keys[0]]
            
            chrom = gene_obj.chrom
            if not chrom.startswith("chr"): 
                 chrom = "chr" + chrom
            variant_chrom = chrom
            
            # Determine optimal gene slice bounding that fits both variant and gene within Shorkie bounds
            gene_center = gene_obj.midpoint()
            
            seq_out_offset_bp = seqnn_model.model_strides[0] * seqnn_model.target_crops[0]
            seq_out_len_bp = seqnn_model.model_strides[0] * seqnn_model.target_lengths[0]
            
            # The variant `pos` must fall strictly within `[start, start + seq_len)`
            # The gene_center must optimally fall within the output slice `[start + offset, start + offset + len)`
            min_start = max(pos - args.seq_len + 1, gene_center - seq_out_offset_bp - seq_out_len_bp + 1)
            max_start = min(pos - 1, gene_center - seq_out_offset_bp)
            
            if min_start <= max_start:
                start = int((min_start + max_start) // 2)
            else:
                # Variant fundamentally cannot map alongside gene inside architectural limits
                start = int(gene_center - args.seq_len // 2)
                
            end = start + args.seq_len
            
            seq_out_start = int(start + seq_out_offset_bp)
            seq_out_len = int(seq_out_len_bp)
            
            gene_slice = gene_obj.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)
            
            # Prepare Sequences
            sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)
            
            if pd.isna(alt_csv):
                print(f"SKIPPED: Alt is NaN (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                continue
                
            alleles = [a.strip() for a in alt_csv.split(",") if a.strip()]
            
            for alt in alleles:
                if len(alt) != 1:
                    print(f"SKIPPED: Not a SNP (Alt: {alt}) (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                    continue
                
                # Skip if alt equals ref (no mutation)
                if alt == ref_csv:
                    print(f"SKIPPED: Alt == Ref ({alt}) (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                    continue
                
                # Check Ref Match
                center_idx = pos - start - 1
                
                # Verify Ref via genome extracting
                val = sequence_one_hot_wt[center_idx, :4].numpy()
                val_idx = np.argmax(val)
                nt_map = {0:'A', 1:'C', 2:'G', 3:'T'}
                extracted_ref = nt_map.get(val_idx, 'N')
                
                if np.sum(val) == 0:
                    extracted_ref = 'N'

                if ref_csv != extracted_ref:
                    print(f"DEBUG: Genome extracted ref ({extracted_ref}) != input ref ({ref_csv}) at {variant_chrom}:{pos}_{gene}.", file=sys.stderr)
                else:
                    print(f"DEBUG: Ref match verified at {variant_chrom}:{pos}_{gene}. Ref: {extracted_ref}", file=sys.stderr)

                # Apply mutation
                alt_ix = -1
                if alt == 'A': alt_ix = 0
                elif alt == 'C': alt_ix = 1
                elif alt == 'G': alt_ix = 2
                elif alt == 'T': alt_ix = 3
                
                if alt_ix == -1: 
                    print(f"SKIPPED: Invalid Alt Allele '{alt}' (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                    continue
                
                seq_mut_curr = np.copy(sequence_one_hot_wt)
                if 0 <= center_idx < args.seq_len:
                    seq_mut_curr[center_idx, :4] = 0.
                    seq_mut_curr[center_idx, alt_ix] = 1.
                
                # Predict
                y_wt = predict_tracks_ensemble(models, sequence_one_hot_wt)
                y_mut = predict_tracks_ensemble(models, seq_mut_curr)
                
                # Calculate logSED (Aggregated)
                cov_wt_all = np.mean(y_wt, axis=(0,1,3))
                cov_mt_all = np.mean(y_mut, axis=(0,1,3))
                
                sum_ref_all = cov_wt_all[gene_slice].sum()
                sum_alt_all = cov_mt_all[gene_slice].sum()
                
                logSED_agg  = np.log2(sum_alt_all + 1) - np.log2(sum_ref_all + 1)
                
                # Per Track (Mean)
                cov_wt = np.mean(y_wt, axis=(0,1))
                cov_mt = np.mean(y_mut, axis=(0,1))
                
                sr_tracks = cov_wt[gene_slice, :].sum(axis=0)
                sa_tracks = cov_mt[gene_slice, :].sum(axis=0)
                
                logSED_tracks = np.log2(sa_tracks + 1) - np.log2(sr_tracks + 1)
                logSED_mean_pertrack = np.mean(logSED_tracks)

                results.append({
                    "Gene": gene,
                    "Chrom": chrom,
                    "Pos": pos,
                    "Ref": ref_csv,
                    "Alt": alt,
                    "logSED_agg": logSED_agg,
                    "logSED_mean_pertrack": logSED_mean_pertrack
                })

        except Exception as e:
            print(f"SKIPPED: Error processing row {idx} (Variant: {gene}:{pos}): {e}", file=sys.stderr)
            continue

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Saved results to {args.output_file}")
    
    fasta_open.close()

if __name__ == "__main__":
    main()
