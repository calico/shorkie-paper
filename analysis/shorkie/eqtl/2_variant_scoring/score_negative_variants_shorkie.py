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
import gc

# Add path to import yeast_helpers_selfsupervised from positive directory
sys.path.append("/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/revision_experiments/eQTL/kita_etal_Shorkie/positive")
from yeast_helpers_selfsupervised import *

def parse_args():
    parser = argparse.ArgumentParser(description="Score negative variants using Shorkie (logSED) for Kita et al.")
    parser.add_argument("--variants_tsv", required=True, help="Path to negative variants TSV")
    
    # Model arguments matching viz_rnaseq_cov_ISM.py
    parser.add_argument("--params_file", required=True)
    parser.add_argument("--targets_file", required=True)
    parser.add_argument("--gtf_file", required=True)
    parser.add_argument("--fasta_file", required=True)
    
    parser.add_argument("--output_file", required=True, help="Output file path")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of variants for testing")
    parser.add_argument("--num_folds", type=int, default=8, help="Number of trained folds to ensemble.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for prediction")
    
    return parser.parse_args()

def predict_batch_ensemble(models, x_batch):
    batch_preds = []
    
    for i, model in enumerate(models):
        p = model(x_batch)
        p = np.array(p, dtype="float32")
        
        if i == 0:
            accum = p
        else:
            accum += p
            
        del p
        gc.collect()
        
    accum /= len(models)
    return accum

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
    
    seqnn_model0 = models[0]
    
    # Load Genomic Resources
    print(f"Loading FASTA and GTF...")
    fasta_open = pysam.Fastafile(args.fasta_file)
    transcriptome = bgene.Transcriptome(args.gtf_file)
    
    # Load Variants
    print(f"Loading variants from {args.variants_tsv}...")
    df = pd.read_csv(args.variants_tsv, sep="\t", dtype=str)
    
    results = []
    
    print(f"Processing variants...", flush=True)
    total = len(df)
    if args.limit:
        total = min(total, args.limit)
    
    batch_size = args.batch_size
    
    for start_idx in tqdm(range(0, total, batch_size)):
        end_idx = min(start_idx + batch_size, total)
        current_batch_df = df.iloc[start_idx:end_idx]
        
        batch_inputs_wt = []
        batch_inputs_mut = []
        batch_meta = []
        
        for i, (idx, row) in enumerate(current_batch_df.iterrows()):
            try:
                gene = row["neg_gene"]
                variant_chrom = row["neg_chrom"]
                pos = int(row["neg_pos"])
                ref_tsv = str(row["neg_ref"]).strip()
                alt_tsv = str(row["neg_alt"]).strip()

                # CHROM MAPPING
                chrom_map = {
                    "chromosome1": "chrI", "chromosome2": "chrII", "chromosome3": "chrIII", "chromosome4": "chrIV",
                    "chromosome5": "chrV", "chromosome6": "chrVI", "chromosome7": "chrVII", "chromosome8": "chrVIII",
                    "chromosome9": "chrIX", "chromosome10": "chrX", "chromosome11": "chrXI", "chromosome12": "chrXII",
                    "chromosome13": "chrXIII", "chromosome14": "chrXIV", "chromosome15": "chrXV", "chromosome16": "chrXVI"
                }
                chrom_str = chrom_map.get(variant_chrom, variant_chrom)
                
                # TRANSCRIPTOME
                gene_keys = [gkey for gkey in transcriptome.genes.keys() if gene in gkey]
                if len(gene_keys) == 0: 
                    print(f"SKIPPED: Gene {gene} not found in transcriptome (Variant: {variant_chrom}:{pos})", file=sys.stderr)
                    continue
                gene_obj = transcriptome.genes[gene_keys[0]]
                
                # Determine optimal gene slice bounding that fits both variant and gene within Shorkie bounds
                gene_center = gene_obj.midpoint()
                
                seq_out_offset_bp = seqnn_model0.model_strides[0] * seqnn_model0.target_crops[0]
                seq_out_len_bp = seqnn_model0.model_strides[0] * seqnn_model0.target_lengths[0]
                
                min_start = max(pos - args.seq_len + 1, gene_center - seq_out_offset_bp - seq_out_len_bp + 1)
                max_start = min(pos - 1, gene_center - seq_out_offset_bp)
                
                if min_start <= max_start:
                    start = int((min_start + max_start) // 2)
                else:
                    start = int(gene_center - args.seq_len // 2)
                    
                end = start + args.seq_len
                
                seq_out_start = int(start + seq_out_offset_bp)
                seq_out_len = int(seq_out_len_bp)
                
                stride = int(seqnn_model0.model_strides[0])
                gene_slice = gene_obj.output_slice(seq_out_start, seq_out_len, stride, False)
                if hasattr(gene_slice, 'astype'):
                    gene_slice = gene_slice.astype(int)
                elif isinstance(gene_slice, list):
                    gene_slice = np.array(gene_slice, dtype=int)
                
                # SEQUENCE
                x_wt_var = process_sequence(fasta_open, chrom_str, start, end, args.seq_len)
                x_wt = x_wt_var.numpy()
                
                # Check Ref Match
                center_idx = pos - start - 1
                
                val = x_wt[center_idx, :4]
                val_idx = np.argmax(val)
                nt_map = {0:'A', 1:'C', 2:'G', 3:'T'}
                extracted_ref = nt_map.get(val_idx, 'N')
                
                if np.sum(val) == 0:
                    extracted_ref = 'N'
                
                if ref_tsv != extracted_ref:
                    print(f"DEBUG: Genome extracted ref ({extracted_ref}) != input ref ({ref_tsv}) at {variant_chrom}:{pos}_{gene}.", file=sys.stderr)
                else:
                    print(f"DEBUG: Ref match verified at {variant_chrom}:{pos}_{gene}. Ref: {extracted_ref}", file=sys.stderr)
                
                if pd.isna(alt_tsv): 
                    print(f"SKIPPED: Alt is NaN (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                    continue
                alleles = [a.strip() for a in alt_tsv.split(",") if a.strip()]
                
                for alt in alleles:
                    if len(alt) != 1: 
                        print(f"SKIPPED: Not a SNP (Alt: {alt}) (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                        continue
                    
                    # Skip if alt equals ref (no mutation)
                    if alt == ref_tsv:
                        print(f"SKIPPED: Alt == Ref ({alt}) (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                        continue
                    
                    center_idx = pos - start - 1
                    alt_ix = {'A':0, 'C':1, 'G':2, 'T':3}.get(alt, -1)
                    if alt_ix == -1: 
                        print(f"SKIPPED: Invalid Alt Allele '{alt}' (Variant: {variant_chrom}:{pos}_{gene})", file=sys.stderr)
                        continue
                    
                    x_mut = np.copy(x_wt)
                    if 0 <= center_idx < args.seq_len:
                        x_mut[center_idx, :4] = 0.
                        x_mut[center_idx, alt_ix] = 1.
                    
                    batch_inputs_wt.append(x_wt)
                    batch_inputs_mut.append(x_mut)
                    
                    batch_meta.append({
                        "Gene": gene,
                        "Chrom": chrom_str,
                        "Pos": pos,
                        "Ref": ref_tsv,
                        "Alt": alt,
                        "Gene_Slice": gene_slice,
                        "Pos_Gene": row.get("pos_gene"),
                        "Pos_Chrom": row.get("pos_chrom"),
                        "Pos_Pos": row.get("pos_pos")
                    })
                    
            except Exception as e:
                print(f"SKIPPED: Error prepping row {idx}: {e}", file=sys.stderr)
                continue

        if not batch_inputs_wt:
            continue
            
        # Stack
        X_wt = np.array(batch_inputs_wt)
        X_mut = np.array(batch_inputs_mut)
        
        # Predict
        try:
            Y_wt = predict_batch_ensemble(models, X_wt)
            Y_mut = predict_batch_ensemble(models, X_mut)
        except Exception as e:
            print(f"Prediction failed for batch: {e}", file=sys.stderr, flush=True)
            del X_wt, X_mut
            gc.collect()
            continue
            
        # Compute Scores
        for i, meta in enumerate(batch_meta):
            y_w = Y_wt[i]
            y_m = Y_mut[i]
            g_slice = meta["Gene_Slice"]
            
            cov_wt_all = np.mean(y_w, axis=1)
            cov_mt_all = np.mean(y_m, axis=1)
            
            sum_ref_all = cov_wt_all[g_slice].sum()
            sum_alt_all = cov_mt_all[g_slice].sum()
            
            logSED_agg = np.log2(sum_alt_all + 1) - np.log2(sum_ref_all + 1)
            
            sr_tracks = y_w[g_slice, :].sum(axis=0)
            sa_tracks = y_m[g_slice, :].sum(axis=0)
            
            logSED_tracks = np.log2(sa_tracks + 1) - np.log2(sr_tracks + 1)
            logSED_mean_pertrack = np.mean(logSED_tracks)
            
            res_dict = {
                "Gene": meta["Gene"],
                "Chrom": meta["Chrom"],
                "Pos": meta["Pos"],
                "Ref": meta["Ref"],
                "Alt": meta["Alt"],
                "logSED_agg": logSED_agg,
                "logSED_mean_pertrack": logSED_mean_pertrack
            }
            if meta["Pos_Gene"]: res_dict["Pos_Gene"] = meta["Pos_Gene"]
            if meta["Pos_Chrom"]: res_dict["Pos_Chrom"] = meta["Pos_Chrom"]
            if meta["Pos_Pos"]: res_dict["Pos_Pos"] = meta["Pos_Pos"]
            
            results.append(res_dict)
            
        del X_wt, X_mut, Y_wt, Y_mut, batch_inputs_wt, batch_inputs_mut
        gc.collect()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Saved results to {args.output_file} (Rows: {len(res_df)})", flush=True)
    
    fasta_open.close()

if __name__ == "__main__":
    main()
