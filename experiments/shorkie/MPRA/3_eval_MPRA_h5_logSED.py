import h5py
import numpy as np

# Path to your HDF5 file
h5_path = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train//f0c0/MPRA/all_random_seqs/AIM11_neg/sed.h5"

# Open the file and read the ALT, REF, and logSED datasets
with h5py.File(h5_path, "r") as f:
    alt_data = f["ALT"][:]      # shape: (num_SNPs, num_targets)
    ref_data = f["REF"][:]      # shape: (num_SNPs, num_targets)
    logsed_file = f["logSED"][:]  # shape: (num_SNPs, num_targets)

# --------------------------------------------------------------------------
# Recompute the logSED ( = log2(ALT+1) − log2(REF+1) ) for each SNP and target
# NOTE: In your original function, ALT and REF are already the sums
#       across the sequence length dimension, so we just take
#       ALT[i, :] and REF[i, :] as the "alt_preds_sum" and "ref_preds_sum"
# --------------------------------------------------------------------------
logsed_computed = np.log2(alt_data + 1) - np.log2(ref_data + 1)

print("logsed_file: ", logsed_file)
print("logsed_computed: ", logsed_computed)
# Now compare your newly computed logSED to what is stored in the file
# For example, we can look at the maximum absolute difference
max_diff = np.max(np.abs(logsed_computed - logsed_file))
print("Maximum difference between computed logSED and file's logSED:", max_diff)

rel_error = np.abs(logsed_computed - logsed_file) / np.maximum(np.abs(logsed_file), 1e-8)
print("Mean relative error:", np.mean(rel_error))
print("Max relative error:", np.max(rel_error))


# Optionally, do whatever further analysis/plots you need...
# e.g. check if they're essentially the same or if there are any numerical shifts.
