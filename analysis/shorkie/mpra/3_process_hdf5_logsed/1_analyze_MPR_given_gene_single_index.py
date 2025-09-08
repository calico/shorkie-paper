#!/usr/bin/env python
import h5py
import numpy as np
import sys
import os
import pandas as pd

def main(filenames, gene_name, target_gene, t0_indices_file, output_dir=None):
    """
    Process HDF5 files across folds, extract data for the target gene,
    average logSED scores, then sample the target dimension by T0 indices.
    """
    # Load T0 indices
    print(f"Loading T0 indices from {t0_indices_file}")
    idx_df = pd.read_csv(t0_indices_file, sep='\t')
    t0_positions = idx_df['position_0based'].tolist()
    print("t0_positions: ", t0_positions)

    # Prepare output directory
    if output_dir is None:
        output_dir = f"{gene_name}_{target_gene}_outputs"
    os.makedirs(output_dir, exist_ok=True)

    logSED_list = []
    common_si = None
    common_ci = None
    gene_seq_ids = None
    common_ctx_ids = None

    # Iterate over fold files
    for i, filename in enumerate(filenames):
        print(f"in file: {i} {filename}")
        try:
            with h5py.File(filename, "r") as f:
                gene_ds = f["gene"][:]
                si = f["si"][:]
                ci = f["ci"][:]
                logSED = f["logSED"][:]
                ctx_id_ds = f["ctx_id"][:]
                seq_id_ds = f["seq_id"][:]

                # Decode byte strings
                genes = np.array([g.decode() if isinstance(g, bytes) else g for g in gene_ds])
                ctx_ids = np.array([c.decode() if isinstance(c, bytes) else c for c in ctx_id_ds])
                seq_ids = np.array([s.decode() if isinstance(s, bytes) else s for s in seq_id_ds])

                # Find target gene rows
                matching_indices = np.where(genes == target_gene)[0]
                if matching_indices.size == 0:
                    print(f"Gene {target_gene} not found in file: {filename}")
                    continue

                # Save consistent arrays from first file
                if common_si is None:
                    common_si = si[matching_indices]
                    common_ci = ci[matching_indices]
                    gene_seq_ids = seq_ids[common_si]
                    common_ctx_ids = ctx_ids

                # Extract logSED for this gene
                gene_logSED = logSED[matching_indices, :]  # (num_matches, num_targets)
                logSED_list.append(gene_logSED)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    if not logSED_list:
        print("No data found for gene across provided files.")
        return

    # Average across folds
    avg_logSED = np.mean(np.stack(logSED_list, axis=0), axis=0)  # (num_matches, num_targets)
    # Sample by T0 positions
    avg_logSED = avg_logSED[:, t0_positions]  # (num_matches, num_t0)

    # Group by context and save
    num_contexts = len(common_ctx_ids)
    for context_idx in range(num_contexts):
        mask = (common_ci == context_idx)
        count = np.count_nonzero(mask)
        if count == 0:
            print(f"No entries for gene {target_gene} in context index {context_idx} ({common_ctx_ids[context_idx]}).")
            continue

        group_si = common_si[mask]
        group_ci = common_ci[mask]
        group_logSED = avg_logSED[mask, :]
        group_seq_ids = gene_seq_ids[mask]
        group_ctx_id = common_ctx_ids[context_idx]

        print(f"Context {context_idx} ({group_ctx_id}) has {count} entries. logSED sampled shape: {group_logSED.shape}")
        out_file = os.path.join(output_dir, f"{target_gene}_context_{context_idx}_{group_ctx_id}.npz")
        np.savez_compressed(out_file,
                            si=group_si,
                            seq_ids=group_seq_ids,
                            ci=group_ci,
                            logSED=group_logSED,
                            ctx_id=group_ctx_id)
        print(f"Saved data to {out_file}\n")

if __name__ == "__main__":
    # Configuration
    folds = [str(i) for i in range(8)]
    seq_types = ["yeast_seqs", "high_exp_seqs", "low_exp_seqs", "challenging_seqs", "all_random_seqs"]
    strands = ['+', '-']
    # Path to T0 indices file
    t0_indices_file = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/t0_indices.tsv'

    root_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded"

    for strand in strands:
        if strand == '+':
            genes = ["GPM3", "SLI1", "VPS52", "YMR160W", "MRPS28", "YCT1", "RDL2", "PHS1", "RTC3", "MSN4"]
            gene_name_2_id = {
                "GPM3": "YOL056W", 
                "SLI1": "YGR212W", 
                "VPS52": "YDR484W", 
                "YMR160W": "YMR160W", 
                "MRPS28": "YDR337W", 
                "YCT1": "YLL055W", 
                "RDL2": "YOR286W", 
                "PHS1": "YJL097W", 
                "RTC3": "YHR087W", 
                "MSN4": "YKL062W"
            }
        elif strand == '-':
            genes = ["COA4", "ERI1", "RSM25", "ERD1", "MRM2", "SNT2", "CSI2", "RPE1", "PKC1", "AIM11", "MAE1", "MRPL1"]
            gene_name_2_id = {
                "COA4": "YLR218C", 
                "ERI1": "YPL096C-A", 
                "RSM25": "YIL093C", 
                "ERD1": "YDR414C", 
                "MRM2": "YGL136C", 
                "SNT2": "YGL131C", 
                "CSI2": "YOL007C",
                "RPE1": "YJL121C", 
                "PKC1": "YBL105C", 
                "AIM11": "YER093C-A", 
                "MAE1": "YKL029C", 
                "MRPL1": "YDR116C"
            }

        for seq_type in seq_types:
            for gene_name in genes:
                target_gene = gene_name_2_id[gene_name]
                # Build filenames
                filenames = []
                for fold in folds:
                    base = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train"
                    if strand == '+':
                        path = f"{base}/f{fold}c0/MPRA/{seq_type}/{gene_name}_pos/sed.h5"
                        output_dir = f"{root_dir}/all_seq_types/{seq_type}/{gene_name}_{target_gene}_pos_outputs"
                    else:
                        path = f"{base}/f{fold}c0/MPRA/{seq_type}/{gene_name}_neg/sed.h5"
                        output_dir = f"{root_dir}/all_seq_types/{seq_type}/{gene_name}_{target_gene}_neg_outputs"
                    filenames.append(path)

                main(filenames, gene_name, target_gene, t0_indices_file, output_dir)
