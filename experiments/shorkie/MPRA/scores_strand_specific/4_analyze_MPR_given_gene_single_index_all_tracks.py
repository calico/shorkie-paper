#!/usr/bin/env python
import h5py
import numpy as np
import sys
import os

def main(filenames, gene_name, target_gene, output_dir=None):
    """
    Process a list of HDF5 files (each corresponding to a fold), extract data for the target gene,
    and average the logSED scores across folds.
    """
    # If no output directory is provided, use one based on the target gene.
    if output_dir is None:
        output_dir = f"{gene_name}_{target_gene}_outputs"
    # Create the output directory if it does not exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logSED_list = []
    common_si = None
    common_ci = None
    gene_seq_ids = None
    common_ctx_ids = None

    for i, filename in enumerate(filenames):
        print("in file:", i, filename)
        try:
            with h5py.File(filename, "r") as f:
                # Load datasets.
                gene_ds = f["gene"][:]      # shape: (num_scores,)
                si = f["si"][:]             # shape: (num_scores,)
                ci = f["ci"][:]             # shape: (num_scores,)
                logSED = f["logSED"][:]     # shape: (num_scores, num_targets)

                # Identifier datasets.
                ctx_id_ds = f["ctx_id"][:]  # shape: (number_of_contexts,)
                seq_id_ds = f["seq_id"][:]  # shape: (number_of_sequences,)

                print("\t* gene_ds shape:", gene_ds.shape)
                print("\t* si shape:", si.shape)
                print("\t* ci shape:", ci.shape)
                print("\t* logSED shape:", logSED.shape)
                print("\t* ctx_id_ds shape:", ctx_id_ds.shape)
                print("\t* seq_id_ds shape:", seq_id_ds.shape)

                # Convert byte strings to regular strings if needed.
                genes = np.array([g.decode("utf-8") if isinstance(g, bytes) else g 
                                for g in gene_ds])
                ctx_ids = np.array([c.decode("utf-8") if isinstance(c, bytes) else c 
                                    for c in ctx_id_ds])
                seq_ids = np.array([s.decode("utf-8") if isinstance(s, bytes) else s 
                                    for s in seq_id_ds])

                # --- Extract rows for the target gene ---
                matching_indices = np.where(genes == target_gene)[0]
                if matching_indices.size == 0:
                    print(f"Gene {target_gene} not found in file: {filename}")
                    continue

                # For the first file, save indices and related arrays (assuming these are consistent across folds).
                if common_si is None:
                    common_si = si[matching_indices]
                    common_ci = ci[matching_indices]
                    gene_seq_ids = seq_ids[common_si]
                    common_ctx_ids = ctx_ids

                # Extract logSED scores for this gene and append.
                gene_logSED = logSED[matching_indices, :]  # shape: (num_matches, num_targets)
                print("\t* gene_logSED shape:", gene_logSED.shape)
                logSED_list.append(gene_logSED)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    if not logSED_list:
        print("No data found for gene across provided files.")
        return
    print("logSED_list length:", len(logSED_list))
    # Average logSED scores across folds.
    avg_logSED = np.mean(np.stack(logSED_list, axis=0), axis=0)

    # --- Group the entries by their inserted context (ci) ---
    num_contexts = len(common_ctx_ids)
    for context_idx in range(num_contexts):
        # Find indices among target gene entries with this context index.
        group_mask = (common_ci == context_idx)
        group_count = np.count_nonzero(group_mask)
        if group_count == 0:
            print(f"No entries for gene {target_gene} in context index {context_idx} ({common_ctx_ids[context_idx]}).")
            continue

        # Extract data for this context.
        group_si      = common_si[group_mask]
        group_ci      = common_ci[group_mask]  # All equal to context_idx.
        group_logSED  = avg_logSED[group_mask, :]  # averaged scores
        group_seq_ids = gene_seq_ids[group_mask]
        group_ctx_id  = common_ctx_ids[context_idx]  # a single string

        print(f"Context index {context_idx} ({group_ctx_id}) has {group_count} entries for gene {target_gene}.")
        print(f"  Sequence IDs: {len(group_seq_ids)}")
        print(f"  Context IDs: {len(group_ci)}")
        print(f"  logSED shape: {group_logSED.shape}")
        print(f"  ctx_id: {group_ctx_id}")

        # Prepare a filename.
        out_filename = os.path.join(output_dir, f"{target_gene}_context_{context_idx}_{group_ctx_id}.npz")

        # Save the data for this inserted context into an NPZ file.
        np.savez_compressed(out_filename,
                            si=group_si,
                            seq_ids=group_seq_ids,
                            ci=group_ci,
                            logSED=group_logSED,
                            ctx_id=group_ctx_id)
        print(f"Saved data to {out_filename}\n")


if __name__ == "__main__":
    # In this example we loop over a set of gene symbols and sequence types.
    # Provide a list of folds (as strings matching the directory naming convention).
    folds = ["0", "1", "2", "3", "4", "5", "6", "7"]  # <-- update this list as needed

    # seq_types options.
    seq_types = ["yeast_seqs", "high_exp_seqs", "low_exp_seqs", "challenging_seqs", "all_random_seqs"]
    # seq_types = ["challenging_seqs", "all_random_seqs"]
    strands = ['+', '-']

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
                # Look up the proper target gene ID for the given gene symbol.
                target_gene = gene_name_2_id[gene_name]
                # Build a list of HDF5 filenames for the current gene across folds.
                filenames = []
                for fold in folds:
                    if strand == '+':
                        h5_filename = (f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/"
                                     f"seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/"
                                     f"16bp/self_supervised_unet_small_bert_drop/train/f{fold}c0/MPRA/"
                                     f"{seq_type}/{gene_name}_pos/sed.h5")
                        output_dir = f"{root_dir}/all_seq_types/{seq_type}/{gene_name}_{target_gene}_pos_outputs"
                    elif strand == '-':
                        h5_filename = (f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/"
                                     f"seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/"
                                     f"16bp/self_supervised_unet_small_bert_drop/train/f{fold}c0/MPRA/"
                                     f"{seq_type}/{gene_name}_neg/sed.h5")
                        output_dir = f"{root_dir}/all_seq_types/{seq_type}/{gene_name}_{target_gene}_neg_outputs"
                    filenames.append(h5_filename)

                # Process the list of files for the given gene.
                main(filenames, gene_name, target_gene, output_dir)
