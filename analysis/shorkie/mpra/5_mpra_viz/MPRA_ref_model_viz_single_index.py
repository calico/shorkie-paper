import numpy as np
import pandas as pd
import csv
import os
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt

######################################
# Directories
######################################
root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/'
BASE_OUTPUT_DIR = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/viz'

######################################
# 1. Load Ground Truth and CSV Data  
######################################
# Load ground truth expression values.
gt_filename = os.path.join(root_dir, 'filtered_test_data_with_MAUDE_expression.txt')
with open(gt_filename) as f:
    reader = csv.reader(f, delimiter="\t")
    lines = list(reader)
GROUND_TRUTH_EXP = np.array([float(line[1]) for line in lines])
print("GROUND_TRUTH_EXP: ", GROUND_TRUTH_EXP)  # Print first 5 values for verification
print("number of nans in GROUND_TRUTH_EXP:", np.sum(np.isnan(GROUND_TRUTH_EXP)))
print("Total ground truth sequences:", len(GROUND_TRUTH_EXP))
print("Max expression:", np.max(GROUND_TRUTH_EXP))
print("Min expression:", np.min(GROUND_TRUTH_EXP))

# Load CSV files for each sequence type (using the 'pos' column).

# High Expression Sequences
df_high = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/high_exp_seqs.csv'))
final_high = df_high['pos'].values  

# Low Expression Sequences
df_low = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/low_exp_seqs.csv'))
final_low = df_low['pos'].values

# Yeast Sequences
df_yeast = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/yeast_seqs.csv'))
final_yeast = df_yeast['pos'].values

# Challenging Sequences (using sampling approach)
df_challenging = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/challenging_seqs.csv'))
df_challenging_sample = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/fix/challenging_seqs_sample_ids.tsv'), sep="\t")
# Subtract 1 from the sampled row indices (assuming TSV indices are 1-indexed)
sample_indices_challenging = df_challenging_sample['original_row_id'].values - 1
sampled_df_challenging = df_challenging.iloc[sample_indices_challenging]
final_challenging = sampled_df_challenging['pos'].values

# All Random Sequences (using sampling approach)
df_all_random = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/all_random_seqs.csv'))
df_all_random_sample = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/fix/all_random_seqs_sample_ids.tsv'), sep="\t")
sample_indices_all_random = df_all_random_sample['original_row_id'].values - 1
sampled_df_all_random = df_all_random.iloc[sample_indices_all_random]
final_all_random = sampled_df_all_random['pos'].values

print("Counts:")
print("High exp sequences:", len(final_high))
print("Low exp sequences:", len(final_low))
print("Yeast sequences:", len(final_yeast))
print("Challenging sequences:", len(final_challenging))
print("All random sequences:", len(final_all_random))

# Create a dictionary mapping sequence type to its CSV indices.
csv_indices_dict = {
    "high_exp_seqs": final_high,
    "low_exp_seqs": final_low,
    "yeast_seqs": final_yeast,
    "challenging_seqs": final_challenging,
    "all_random_seqs": final_all_random
}

######################################
# 2. Load Reference Model Output
######################################
# Assume the reference model output file has the same format as the ground truth file.

ref_filename = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/random-promoter-dream-challenge-2022/data/DREAM-RNN_output.txt"
with open(ref_filename) as f:
    reader = csv.reader(f, delimiter="\t")
    ref_lines = list(reader)
REFERENCE_EXP = np.array([float(line[1]) for line in ref_lines])
print("Total DREAM-RNN model predictions:", len(REFERENCE_EXP))
######################################
# 3. Function to Plot Scatter Plot with Regression
######################################
def plot_scatter_with_regression(gt, pred, title, filename, corr_pearson, corr_spearman, label_name):
    plt.figure(figsize=(5,5))
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    plt.scatter(gt, pred, color='green', label=label_name, s=15, alpha=0.6)
    try:
        slope, intercept = np.polyfit(gt, pred, 1)
        x_vals = np.linspace(np.min(gt), np.max(gt), 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, 'r-', label=f"Regression (Pearson: {corr_pearson:.3f}, Spearman: {corr_spearman:.3f})")
    except Exception as e:
        print("Regression computation error:", e)
    plt.xlabel("Average expression levels (YFP fluorescence)")
    plt.ylabel("DREAM-RNN model prediction")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved scatter plot to {filename}")

######################################
# 4. Loop Over Sequence Types & Plot Scatter Plots
######################################
seq_types = ["yeast_seqs", "high_exp_seqs", "low_exp_seqs", "challenging_seqs", "all_random_seqs"]

for seq_type in seq_types:
    # Get indices for the current group from the dictionary.
    indices = np.array(csv_indices_dict[seq_type]).astype(int)
    gt_subset = GROUND_TRUTH_EXP[indices]
    ref_subset = REFERENCE_EXP[indices]
    
    # remove NaNs and Infs
    mask = np.isfinite(gt_subset) & np.isfinite(ref_subset)
    gt_subset  = gt_subset[mask]
    ref_subset = ref_subset[mask]
    if len(gt_subset) < 2:
        print(f"Not enough valid points for {seq_type}, skipping.")
        continue

    # Compute correlation coefficients.
    if np.std(gt_subset) == 0 or np.std(ref_subset) == 0:
        pearson_corr = 0.0
        spearman_corr = 0.0
    else:
        pearson_corr, _ = pearsonr(gt_subset, ref_subset)
        spearman_corr, _ = spearmanr(gt_subset, ref_subset)
    
    # Determine label name based on the sequence type.
    if seq_type == "yeast_seqs":
        label_name = "Yeast Sequence"
    elif seq_type == "high_exp_seqs":
        label_name = "High Expression Sequence"
    elif seq_type == "low_exp_seqs":
        label_name = "Low Expression Sequence"
    elif seq_type == "challenging_seqs":
        label_name = "Challenging Sequence"
    elif seq_type == "all_random_seqs":
        label_name = "All Random Sequence"
    else:
        label_name = seq_type

    title = f"{label_name}: \nDREAM-RNN Predictions vs. Experimental Measurements"
    output_file = os.path.join(BASE_OUTPUT_DIR, "scatterplots", seq_type, f"scatter_reference_vs_ground_truth_{seq_type}.png")
    plot_scatter_with_regression(gt_subset, ref_subset, title, output_file, pearson_corr, spearman_corr, label_name)
