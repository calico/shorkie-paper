#!/usr/bin/env python

import os
import sys
import gc
import json
import h5py
import numpy as np
import pandas as pd
import pysam
import pyranges as pr
import tensorflow as tf
from optparse import OptionParser

########################
# GLOBAL CONSTANT
########################
NUM_SPECIES = 165  # Single place to define the number of extra species channels

########################
# HELPER FUNCTIONS
########################

def reverse_complement(seq):
    """
    Return the reverse-complement of a DNA sequence.
    """
    complement_map = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement_map)[::-1]

def one_hot_encode(seq):
    """
    Convert a DNA sequence into one-hot encoding.
    A, C, G, T are encoded; any other character (e.g. N) becomes a zero vector.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 
               'a': 0, 'c': 1, 'g': 2, 't': 3}
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    return one_hot

def compute_complement(pr_obj, chrom_sizes):
    """
    Compute the complement intervals (i.e. intergenic regions) for a PyRanges object.
    Merges overlapping intervals, then computes the gaps on each chromosome.
    Returns a DataFrame with columns: Chromosome, Start, End.
    """
    merged_pr = pr_obj.merge()
    merged_df = merged_pr.as_df()
    complement_list = []
    for chrom, size in chrom_sizes.items():
        chrom_df = merged_df[merged_df.Chromosome == chrom].sort_values("Start")
        current_start = 0
        if chrom_df.shape[0] == 0:
            complement_list.append({"Chromosome": chrom, "Start": 0, "End": size})
        else:
            for _, row in chrom_df.iterrows():
                if row["Start"] > current_start:
                    complement_list.append({"Chromosome": chrom,
                                            "Start": current_start,
                                            "End": row["Start"]})
                current_start = max(current_start, row["End"])
            if current_start < size:
                complement_list.append({"Chromosome": chrom,
                                        "Start": current_start,
                                        "End": size})
    return pd.DataFrame(complement_list)

def process_seqs(chrom, start, end, strand, seq_len, fasta_open,
                 variable_length=False):
    """
    Fetch the sequence for [start, end] on the given strand.
    
    If variable_length=False:
      - Trim the sequence if it's longer than seq_len.
      - Pad with 'N's if it's shorter than seq_len.
    If variable_length=True:
      - Use the interval as is, without trimming or padding.

    Then:
      1. Convert to forward orientation if strand == '+'
         or reverse-complement if strand == '-'.
      2. One-hot encode into shape (L, 4).
      3. Concatenate (L, NUM_SPECIES+1) zeros => final shape (L, 4 + NUM_SPECIES + 1).
      4. Set channel #114 to 1 across all positions (per your code's convention).
    """
    # 1) Clip to chromosome boundaries
    chrom_size = fasta_open.get_reference_length(chrom)
    start_clipped = max(0, start)
    end_clipped = min(end, chrom_size)
    actual_len = end_clipped - start_clipped

    # 2) Handle fixed-length or variable-length
    if not variable_length:
        desired_len = seq_len
        if actual_len > desired_len:
            # Trim the sequence from the center
            excess = actual_len - desired_len
            start_clipped += excess // 2
            end_clipped = start_clipped + desired_len
            # Clip again in case of boundary issues
            start_clipped = max(0, start_clipped)
            end_clipped = min(end_clipped, chrom_size)
        elif actual_len < desired_len:
            # We'll handle padding with 'N' below
            pass

    # 3) Fetch the forward-strand sequence from the genome
    seq_str = fasta_open.fetch(chrom, start_clipped, end_clipped)

    # 4) Pad with 'N' if it's still shorter than desired_len
    final_len = end_clipped - start_clipped
    if not variable_length and final_len < seq_len:
        diff = seq_len - final_len
        left_pad = diff // 2
        right_pad = diff - left_pad
        seq_str = ("N" * left_pad) + seq_str + ("N" * right_pad)
        final_len = len(seq_str)

    # 5) Reverse-complement if strand == '-'
    if strand == '-':
        seq_str = reverse_complement(seq_str)

    # 6) One-hot encode
    seq_1hot = one_hot_encode(seq_str)  # shape (L, 4)

    # 7) Add extra channels
    extra_zeros = np.zeros((seq_1hot.shape[0], NUM_SPECIES + 1), dtype=np.float32)
    x_new = np.concatenate([seq_1hot, extra_zeros], axis=-1)  # (L, 4 + NUM_SPECIES + 1)

    # 8) Convert to TF tensor and set channel 114
    x_new = tf.Variable(x_new)
    x_new[:, 114].assign(tf.ones([tf.shape(x_new)[0]], dtype=tf.float32))

    return x_new  # shape (L, 170) if NUM_SPECIES=165

def extract_hidden_reps(seqnn_model, x_batch, layer_names, pool_method='mean'):
    """
    Extract embeddings from multiple hidden layers specified by layer_names,
    for all items in x_batch.

    Args:
      seqnn_model: The trained model.
      x_batch: Tensor of shape [batch_size, seq_len, channels].
      layer_names: List of layer names to extract from the model.
      pool_method: 'mean', 'max', or 'cls' to pool across the seq_len dimension.

    Returns:
      A dict of {layer_name -> numpy array of shape (batch_size, hidden_dim)}.
    """
    embeddings_dict = {layer: [] for layer in layer_names}

    # Build sub-models one time for each layer
    sub_models = {}
    for layer in layer_names:
        lyr = seqnn_model.model.get_layer(layer)
        rep_model = tf.keras.Model(inputs=seqnn_model.model.inputs, 
                                   outputs=lyr.output)
        sub_models[layer] = rep_model

    # For each layer, do a forward pass once
    for layer in layer_names:
        rep_model = sub_models[layer]
        hidden_states = rep_model.predict(x_batch)  # shape could be (B, L, D) or (B, D)

        # We'll unify pooling logic: if the output is 3D, apply pooling;
        # if 2D, we assume the layer is already pooled.
        if hidden_states.ndim == 3:
            # hidden_states: (batch_size, seq_len, hidden_dim)
            if pool_method == 'mean':
                emb = hidden_states.mean(axis=1)
            elif pool_method == 'max':
                emb = hidden_states.max(axis=1)
            elif pool_method == 'cls':
                emb = hidden_states[:, 0, :]  # take the first position
            else:
                raise ValueError(f"Unknown pooling: {pool_method}")
        elif hidden_states.ndim == 2:
            # shape (batch_size, hidden_dim) => already pooled
            emb = hidden_states
        else:
            raise ValueError(f"Unexpected shape for layer {layer}: {hidden_states.shape}")

        embeddings_dict[layer] = emb

    return embeddings_dict

########################
# MAIN FUNCTION
########################

def main():
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("--fasta_file", dest="fasta_file", default="genome.fasta",
                      help="Path to FASTA file of the genome.")
    parser.add_option("--gtf_file", dest="gtf_file", default="annotations.gtf",
                      help="Path to GTF file with gene/region annotations.")
    parser.add_option("--out_file", dest="out_file", default="embeddings.h5",
                      help="Where to store the output embeddings.")
    parser.add_option("--model_file", dest="model_file", default="model_best.h5",
                      help="Path to the trained language model weights.")
    parser.add_option("--params_file", dest="params_file", default="params.json",
                      help="Path to the JSON file with model params.")
    parser.add_option("--seq_len", dest="seq_len", default=16384, type=int,
                      help="Desired sequence length (if not variable-length).")
    parser.add_option("--pool_method", dest="pool_method", default="mean",
                      help="Pooling method: 'mean', 'max', or 'cls'.")
    parser.add_option("--chrom_filter", dest="chrom_filter", default=None,
                      help="If provided, only process this chromosome (e.g. 'chrI').")
    parser.add_option("--max_count", dest="max_count", default=20, type=int,
                      help="Max intervals to process (<=0 => process all).")
    parser.add_option("--variable_length", dest="variable_length", default=False,
                      action="store_true",
                      help="If set, do not pad/trim sequences to --seq_len.")
    parser.add_option("--batch_size", dest="batch_size", default=8, type=int,
                      help="Number of intervals to process at once (for efficiency).")

    (opts, args) = parser.parse_args()

    # Silence TF warnings and force CPU usage
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    ########################################
    # 1) Load the model
    ########################################
    from baskerville import seqnn

    with open(opts.params_file) as f:
        params = json.load(f)
        params_model = params["model"]
        params_train = params["train"]

    # Adjust model shape to match input channels
    # 4 DNA channels + 165 species channels + 1 special channel = 170 total
    params_model["num_features"] = NUM_SPECIES + 5
    params_model["seq_length"] = opts.seq_len

    # This index is part of your code's convention; only relevant if used in the architecture
    params_train['r64_idx'] = 109

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(opts.model_file, trunk=False)
    seqnn_model.model.summary()

    ########################################
    # 2) Read genome & GTF
    ########################################
    fasta = pysam.Fastafile(opts.fasta_file)
    gtf_pr = pr.read_gtf(opts.gtf_file)
    gtf_df = gtf_pr.as_df()
    # Skip chrMito specifically
    gtf_df = gtf_df[gtf_df["Chromosome"] != "chrMito"].copy()

    # Build chromosome size dict
    chrom_sizes = {ref: length for ref, length in zip(fasta.references, fasta.lengths)}

    # ------------------------------
    # (a) Extract features of interest
    # ------------------------------
    features_of_interest = ['gene', 'CDS', 'start_codon', 'stop_codon']
    selected_df = gtf_df[gtf_df.Feature.isin(features_of_interest)].copy()

    # ------------------------------
    # (b) Create promoter intervals from start_codon
    # ------------------------------
    promoter_list = []
    start_codon_df = selected_df[selected_df.Feature == 'start_codon'].copy()

    for idx, row in start_codon_df.iterrows():
        chrom = row['Chromosome']
        strand = row['Strand']
        gene_id = row.get('gene_id', f'feature_{idx}')

        # If 'start_codon' is a 3bp region, row['Start'] is left-edge, row['End'] is right-edge
        # For the plus strand, the "start coordinate" is row['Start']
        # For the minus strand, the "start coordinate" is row['End']
        if strand == '+':
            prom_start = row['Start'] - 500
            prom_end = row['Start']  # up to but not including the start_codon
        else:
            prom_start = row['End']
            prom_end = row['End'] + 500

        # Bound-check (clip to [0, chrom_size])
        prom_start = max(0, prom_start)
        prom_end = min(chrom_sizes[chrom], prom_end)

        promoter_list.append({
            "Chromosome": chrom,
            "Start": prom_start,
            "End": prom_end,
            "Strand": strand,
            "Feature": "promoter",
            "gene_id": gene_id
        })

    promoter_df = pd.DataFrame(promoter_list)

    # ------------------------------
    # (c) Combine selected features + promoter
    # ------------------------------
    combined_feature_df = pd.concat([selected_df, promoter_df], ignore_index=True)

    # ------------------------------
    # (d) Compute intergenic intervals
    # ------------------------------
    pr_combined_feat = pr.PyRanges(combined_feature_df)
    intergenic_df = compute_complement(pr_combined_feat, chrom_sizes)
    intergenic_df['Feature'] = 'intergenic'
    intergenic_df['gene_id'] = 'intergenic'

    # Merge everything into a single DataFrame
    combined_df = pd.concat([combined_feature_df, intergenic_df], ignore_index=True)

    # If user wants to restrict to a single chromosome
    if opts.chrom_filter is not None:
        combined_df = combined_df[combined_df.Chromosome == opts.chrom_filter].reset_index(drop=True)
    
    print("Unique features in combined_df:", combined_df['Feature'].unique())
    print(f"{len(combined_df)} total intervals after combination.")

    # Example: optionally drop any features you don't want
    #   For demonstration, let's drop 'CDS', 'start_codon', 'stop_codon'
    #   while *keeping* 'promoter' and 'gene' and 'intergenic'.
    combined_df = combined_df[
        ~combined_df.Feature.isin(['CDS', 'start_codon', 'stop_codon'])
    ].reset_index(drop=True)

    # Shuffle intervals
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    combined_df.to_csv("combined_intervals.csv", index=False)

    ########################################
    # 3) Prepare output
    ########################################
    out_h5 = h5py.File(opts.out_file, "w")

    selected_layers = [
        'max_pooling1d_6',
        'multihead_attention',
        'dense',
        'dense_1',
        'multihead_attention_7',
        'dense_14',
        'dense_15',
        'dense_16',
        'dense_28',
        'dense_29'
    ]
    print("\nExtracting from layers:")
    for layer_name in selected_layers:
        print("  ", layer_name)

    # We’ll store embeddings as lists, to be stacked later
    layer_storage = {lyr: [] for lyr in selected_layers}
    meta_storage = []

    ########################################
    # 4) Batched iteration over intervals
    ########################################
    batch_x = []
    batch_meta = []
    processed_count = 0
    total_intervals = len(combined_df)

    for i, row in combined_df.iterrows():
        chrom = row['Chromosome']
        start = row['Start']
        end   = row['End']
        strand = row.get('Strand', '+')
        feature = row.get('Feature', 'unknown')
        gene_id = row.get('gene_id', f'feature_{i}')

        # 4a) Build the input tensor for this interval
        x_new = process_seqs(
            chrom=chrom,
            start=start,
            end=end,
            strand=strand,
            seq_len=opts.seq_len,
            fasta_open=fasta,
            variable_length=opts.variable_length
        )
        # x_new shape: (L, 170)
        batch_x.append(x_new.numpy())  # Convert to numpy for easy stacking
        batch_meta.append([chrom, start, end, strand, feature, gene_id])

        processed_count += 1

        # 4b) Once we have "batch_size" intervals or are at the end, run a forward pass
        if (len(batch_x) == opts.batch_size) or (processed_count == total_intervals) or \
           (0 < opts.max_count == processed_count):

            if not opts.variable_length:
                # All sequences have length == seq_len => stack them
                x_batch = np.stack(batch_x, axis=0)  # shape (B, seq_len, 170)
                x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)

                # Extract hidden embeddings for the entire batch
                batch_embeddings = extract_hidden_reps(
                    seqnn_model,
                    x_batch_tf,
                    selected_layers,
                    pool_method=opts.pool_method
                )
                # batch_embeddings[layer].shape => (B, hidden_dim)

                # Save them
                for layer_name in selected_layers:
                    layer_storage[layer_name].append(batch_embeddings[layer_name])
                meta_storage.extend(batch_meta)

            else:
                # variable_length=True => process each interval individually
                for bx, meta in zip(batch_x, batch_meta):
                    single_tf = tf.convert_to_tensor(bx[None, ...], dtype=tf.float32)  # shape (1, L, 170)
                    single_embeddings = extract_hidden_reps(
                        seqnn_model,
                        single_tf,
                        selected_layers,
                        pool_method=opts.pool_method
                    )
                    for layer_name in selected_layers:
                        # single_embeddings[layer_name].shape => (1, hidden_dim)
                        layer_storage[layer_name].append(single_embeddings[layer_name])
                    meta_storage.append(meta)

            # Clear the batch buffers
            batch_x = []
            batch_meta = []

            # If max_count is used, check if we hit the limit
            if 0 < opts.max_count == processed_count:
                print(f"Reached max_count={opts.max_count}. Stopping iteration.")
                break

        if (processed_count % 100 == 0):
            print(f"Processed {processed_count} intervals...")

    ########################################
    # 5) Write to HDF5
    ########################################
    meta_array = np.array(meta_storage, dtype='S')
    out_h5.create_dataset("metadata", data=meta_array)

    # For each layer, we have a list of arrays
    # If we processed in batch mode (variable_length=False), each element in the list
    # is an array of shape (B, hidden_dim). We need to concatenate them.
    for layer_name in selected_layers:
        chunk_list = layer_storage[layer_name]
        # Each chunk is shape (batch_size, hidden_dim).
        stacked_emb = np.concatenate(chunk_list, axis=0)  # shape (num_intervals, hidden_dim)
        out_h5.create_dataset(f"embeddings_{layer_name}", data=stacked_emb)

    out_h5.close()
    print(f"\nDone. Embeddings saved to {opts.out_file}")

if __name__ == "__main__":
    main()
