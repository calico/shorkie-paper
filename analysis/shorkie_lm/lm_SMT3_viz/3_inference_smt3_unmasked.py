#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
from baskerville import seqnn

# Parameters (Hardcoded based on exploration)
MODEL_DIR = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train"
MODEL_FILE = os.path.join(MODEL_DIR, "model_best.h5")
PARAMS_FILE = os.path.dirname(MODEL_DIR) + "/params.json" # Parent dir usually
FASTA_FILE = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask"
OUTPUT_DIR = "inference_smt3_output"
SPECIES_INDEX = 9 # GCA_000146045_2 index (0-based) from sorted list
NUM_SPECIES = 165
SEQ_LENGTH = 16384 # From statistics.json

# SMT3 Windows (from grep of sequences_train.cleaned.bed)
# Format: chrom, start, end, label, species
SMT3_WINDOWS = [
    ("chrIV", 1454592, 1470976, "train", "GCA_000146045_2"),
    ("chrIV", 1458688, 1475072, "train", "GCA_000146045_2"),
    ("chrIV", 1462784, 1479168, "train", "GCA_000146045_2"),
    ("chrIV", 1466880, 1483264, "train", "GCA_000146045_2"),
]

def dna_1hot(seq):
    seq_len = len(seq)
    seq_code = np.zeros((seq_len, 4), dtype='float16')
    seq = seq.upper()
    for i in range(seq_len):
        nt = seq[i]
        if nt == 'A':
            seq_code[i, 0] = 1
        elif nt == 'C':
            seq_code[i, 1] = 1
        elif nt == 'G':
            seq_code[i, 2] = 1
        elif nt == 'T':
            seq_code[i, 3] = 1
        # N or others remain 0
    return seq_code

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading params from {PARAMS_FILE}")
    with open(PARAMS_FILE) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    
    params_model["num_features"] = 4
    if params_train["loss"] == 'mlm':
        params_model["num_features"] = NUM_SPECIES + 5

    print(f"Initializing model...")
    seqnn_model = seqnn.SeqNN(params_model)
    print(f"Restoring model from {MODEL_FILE}")
    seqnn_model.restore(MODEL_FILE, 0)
    
    fasta_open = pysam.Fastafile(FASTA_FILE)
    
    x_trues = []
    x_preds = []
    labels = [] # Species labels (indices)
    
    print(f"Processing {len(SMT3_WINDOWS)} windows likely covering SMT3 (Unmasked Inference)...")
    
    for i, (chrom, start, end, label_str, species_str) in enumerate(SMT3_WINDOWS):
        print(f"Processing Window {i}: {chrom}:{start}-{end}")
        
        seq = fasta_open.fetch(chrom, start, end)
        if len(seq) != SEQ_LENGTH:
            print(f"Warning: Sequence length {len(seq)} != {SEQ_LENGTH}. Padding or cropping.")
            # Simple pad if short
            if len(seq) < SEQ_LENGTH:
                 seq = seq + "N" * (SEQ_LENGTH - len(seq))
            else:
                 seq = seq[:SEQ_LENGTH]
                 
        # One hot
        x = dna_1hot(seq) # (L, 4)
        
        # Prepare Label (Species One Hot)
        # label_vec expected to be (num_species,)
        label_vec = np.zeros(NUM_SPECIES, dtype='float32')
        label_vec[SPECIES_INDEX] = 1.0
        
        # Construct Input Pattern (Masked LM style)
        # Input shape: (1, L, 4 + 1 + num_species) 
        # 4 bases + 1 mask channel + num_species
        
        # Add batch dim
        x_batch = x[np.newaxis, ...] # (1, L, 4)
        label_batch = label_vec[np.newaxis, ...] # (1, num_species)
        
        # Concatenate: x, zeros(mask), tiled_label
        # (1, L, 4)
        # (1, L, 1) -> Mask channel (REMAINS 0 for Unmasked Inference)
        # (1, L, S) -> Species
        
        x_inp = np.concatenate([
            x_batch,
            np.zeros((1, SEQ_LENGTH, 1)),
            np.tile(label_batch[:, np.newaxis, :], (1, SEQ_LENGTH, 1))
        ], axis=-1)
        
        # Predict on Unmasked Input
        # Pass the full sequence with mask bit 0. 
        # The model will output probabilities for each position based on the attention context.
        print("Running unmasked inference...")
        x_pred = seqnn_model.model.predict(x=[x_inp], batch_size=1, verbose=False)[..., :4].astype('float16')
                    
        # Append results
        x_trues.append(x)
        x_preds.append(x_pred[0]) # Remove batch dim
        labels.append(SPECIES_INDEX)
        
    # Save
    x_true_arr = np.array(x_trues, dtype='float16')
    x_pred_arr = np.array(x_preds, dtype='float16')
    label_arr = np.array(labels, dtype='int32')
    
    out_file = os.path.join(OUTPUT_DIR, "preds_smt3_unmasked.npz")
    print(f"Saving results to {out_file}")
    np.savez_compressed(
        out_file,
        x_true=x_true_arr,
        x_pred=x_pred_arr,
        label=label_arr
    )
    print("Done.")

if __name__ == "__main__":
    main()
