#!/usr/bin/env python
import h5py
import numpy as np
import pandas as pd
import sys
import os

def process_fold(filename, target_gene, t0_positions):
    """
    Load HDF5, extract entries for target_gene, and sample columns by T0 positions.
    Returns dict keyed by context index.
    """
    results = {}
    with h5py.File(filename, "r") as f:
        gene_ds = f["gene"][:]         # (num_scores,)
        si = f["si"][:]
        ci = f["ci"][:]
        logSED = f["logSED"][:]        # (num_scores, num_targets)
        logSED_ALT = f["logSED_ALT_ORIG"][:]
        logSED_REF = f["logSED_REF_ORIG"][:]

        ctx_id_ds = f["ctx_id"][:]
        seq_id_ds = f["seq_id"][:]

        # decode bytes
        genes = np.array([g.decode() if isinstance(g, bytes) else g for g in gene_ds])
        ctx_ids = np.array([c.decode() if isinstance(c, bytes) else c for c in ctx_id_ds])
        seq_ids = np.array([s.decode() if isinstance(s, bytes) else s for s in seq_id_ds])

        # find rows for target
        idx = np.where(genes == target_gene)[0]
        if idx.size == 0:
            print(f"Gene {target_gene} not found in {filename}")
            return results

        # apply T0 sampling on columns
        sampled_logSED = logSED[idx][:, t0_positions]
        sampled_ALT   = logSED_ALT[idx][:, t0_positions]
        sampled_REF   = logSED_REF[idx][:, t0_positions]

        gene_si = si[idx]
        gene_ci = ci[idx]
        gene_seq_ids = seq_ids[gene_si]

        # group by context
        for ci_val in np.unique(gene_ci):
            mask = gene_ci == ci_val
            results[ci_val] = {
                'si': gene_si[mask],
                'ci': gene_ci[mask],
                'logSED': sampled_logSED[mask, :],
                'logSED_ALT_ORIG': sampled_ALT[mask, :],
                'logSED_REF_ORIG': sampled_REF[mask, :],
                'seq_ids': gene_seq_ids[mask],
                'ctx_id': ctx_ids[ci_val]
            }
    return results


def average_results(results_list):
    """
    Average numeric arrays across fold results for each context.
    """
    averaged = {}
    contexts = sorted(set().union(*[res.keys() for res in results_list]))
    for ctx in contexts:
        entries = [res[ctx] for res in results_list if ctx in res]
        if not entries:
            continue
        first = entries[0]
        # stack and mean
        logSED_stack = np.stack([e['logSED'] for e in entries], axis=0)
        alt_stack   = np.stack([e['logSED_ALT_ORIG'] for e in entries], axis=0)
        ref_stack   = np.stack([e['logSED_REF_ORIG'] for e in entries], axis=0)
        averaged[ctx] = {
            'si': first['si'],
            'ci': first['ci'],
            'logSED': np.mean(logSED_stack, axis=0),
            'logSED_ALT_ORIG': np.mean(alt_stack, axis=0),
            'logSED_REF_ORIG': np.mean(ref_stack, axis=0),
            'seq_ids': first['seq_ids'],
            'ctx_id': first['ctx_id']
        }
    return averaged


def main_from_files(filenames, gene_name, target_gene, t0_file, output_dir=None):
    if output_dir is None:
        output_dir = f"{gene_name}_{target_gene}_averaged"
    os.makedirs(output_dir, exist_ok=True)

    # load T0 positions
    df_idx = pd.read_csv(t0_file, sep='\t')
    t0_positions = df_idx['position_0based'].tolist()

    all_results = []
    for fn in filenames:
        print(f"Processing: {fn}")
        res = process_fold(fn, target_gene, t0_positions)
        all_results.append(res)

    averaged = average_results(all_results)
    for ctx, data in averaged.items():
        count = len(data['si'])
        print(f"Context {ctx} ({data['ctx_id']}): {count} entries, shape {data['logSED'].shape}")
        out_npz = os.path.join(output_dir, f"{target_gene}_ctx{ctx}_{data['ctx_id']}_avg.npz")
        np.savez_compressed(out_npz,
            si=data['si'], ci=data['ci'], seq_ids=data['seq_ids'], ctx_id=data['ctx_id'],
            logSED=data['logSED'], logSED_ALT_ORIG=data['logSED_ALT_ORIG'], logSED_REF_ORIG=data['logSED_REF_ORIG'])
        print(f"Saved to {out_npz}\n")


if __name__ == "__main__":
    # Define the three sequence types.
    seq_types = ["all_SNVs_seqs", "motif_perturbation", "motif_tiling_seqs"]
    # Define the strands.
    strands = ['+', '-']
    # Define the fold numbers (0 to 7).
    folds = list(range(8))
    t0_indices_file = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/t0_indices.tsv'
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
        root_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded"
        for seq_type in seq_types:
            for gene_name in genes:
                target_gene = gene_name_2_id[gene_name]
                filenames = []
                if strand == '+':
                    for fold in folds:
                        h5_filename = (
                            f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/"
                            f"seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/"
                            f"16bp/self_supervised_unet_small_bert_drop/train/f{fold}c0/MPRA/"
                            f"{seq_type}/{gene_name}_pos/sed.h5"
                        )
                        filenames.append(h5_filename)
                    output_dir = f"{root_dir}/all_seq_types/{seq_type}/{gene_name}_{target_gene}_pos_outputs"
                elif strand == '-':
                    for fold in folds:
                        h5_filename = (
                            f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/"
                            f"seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/"
                            f"16bp/self_supervised_unet_small_bert_drop/train/f{fold}c0/MPRA/"
                            f"{seq_type}/{gene_name}_neg/sed.h5"
                        )
                        filenames.append(h5_filename)
                    output_dir = f"{root_dir}/all_seq_types/{seq_type}/{gene_name}_{target_gene}_neg_outputs"
                print(f"Processing gene {gene_name} ({target_gene}) across folds: {folds}")
                main_from_files(filenames, gene_name, target_gene, t0_indices_file, output_dir)
