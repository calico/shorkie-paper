import os
import csv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt

# Directories
ROOT_DIR = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/'
OUTPUT_DIR = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/viz'
EPSILON = 1  # to avoid log(0)

def load_expression(filename):
    """Load expression values (assumes second column) from a file."""
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    expr = np.array([float(line[1]) for line in lines])
    print(f"Loaded {len(expr)} expression values from {os.path.basename(filename)}")
    return expr

def load_pairs(csv_path):
    """Load double-index pairs from a CSV file (expects 'alt_pos' and 'ref_pos' columns)."""
    df = pd.read_csv(csv_path)
    pairs = list(zip(df['alt_pos'], df['ref_pos']))
    print(f"Loaded {len(pairs)} pairs from {os.path.basename(csv_path)}")
    return pairs

def load_sample_indices(tsv_path):
    """Load sample indices from a TSV file and convert to 0-indexed values."""
    df = pd.read_csv(tsv_path, sep="\t")
    sample_indices = df['original_row_id'].astype(int).values - 1
    print(f"Loaded {len(sample_indices)} sample indices from {os.path.basename(tsv_path)}")
    return sample_indices

def get_subset_pairs(pairs, sample_indices):
    """
    Use the sample indices to select a subset of pairs from the full list.
    (Assumes that each sample index corresponds to a row in the pairs list.)
    """
    subset = [pairs[i] for i in sample_indices if i < len(pairs)]
    print(f"Subset contains {len(subset)} pairs")
    return subset

def compute_dual_scores(pairs, gt, pred, epsilon=EPSILON):
    """
    For each pair (alt, ref), compute the log₂ fold-change for both ground truth and model predictions,
    then compute the absolute difference (logSED) between them.
    
    Pairs that yield non-finite values are skipped.
    """
    def strand_clip_save(score):
        score = np.clip(score, np.finfo(np.float16).min, np.finfo(np.float16).max)
        return score.astype("float16")

    print(f"Computing dual scores for {len(pairs)} pairs...")
    print("pairs: ", pairs[:5])
    print("gt: ", gt[:5])
    print("pred: ", pred[:5])

    gt_log_folds = []
    pred_log_folds = []
    logSED_scores = []
    gt_alt_list = []
    gt_ref_list = []
    pred_alt_list = []
    pred_ref_list = []
    
    for alt, ref_idx in pairs:
        gt_alt = gt[alt]
        gt_ref = gt[ref_idx]
        pred_alt = pred[alt]
        pred_ref = pred[ref_idx]
        # print("pred_alt: ", pred_alt)
        # print("pred_ref: ", pred_ref)
        
        # Compute log fold-changes
        # pred_log_fold = np.log2(pred_alt + epsilon) - np.log2(pred_ref + epsilon)
        # gt_log_fold   = np.log2(gt_alt   + epsilon) - np.log2(gt_ref   + epsilon)
        pred_log_fold = pred_alt - pred_ref
        gt_log_fold   = gt_alt - gt_ref

        # Skip pairs with non-finite values
        if not (np.isfinite(pred_log_fold) and np.isfinite(gt_log_fold)):
            print(f"Skipping pair ({alt}, {ref_idx}) due to non-finite values: "
                  f"pred_log_fold={pred_log_fold}, gt_log_fold={gt_log_fold}")
            print(f"  pred_alt={pred_alt}, pred_ref={pred_ref}, "
                  f"gt_alt={gt_alt}, gt_ref={gt_ref}")
            continue

        pred_log_fold = strand_clip_save(pred_log_fold)
        gt_log_fold   = strand_clip_save(gt_log_fold)
        
        gt_log_folds.append(gt_log_fold)
        pred_log_folds.append(pred_log_fold)
        logSED_scores.append(abs(pred_log_fold - gt_log_fold))
        
        gt_alt_list.append(gt_alt)
        gt_ref_list.append(gt_ref)
        pred_alt_list.append(pred_alt)
        pred_ref_list.append(pred_ref)
        
    print(f"  kept {len(gt_log_folds)} of {len(pairs)} pairs")
    print(f"  ground-truth log₂-fold range = {min(gt_log_folds):.2f} … {max(gt_log_folds):.2f}")
    return (np.array(gt_log_folds),
            np.array(pred_log_folds),
            np.array(logSED_scores),
            np.array(gt_alt_list),
            np.array(gt_ref_list),
            np.array(pred_alt_list),
            np.array(pred_ref_list))

def plot_scatter(x, y, title, filename, label_name, type_exp="diff"):
    """Plot a scatter plot with regression line, including Pearson and Spearman correlations."""
    # plt.figure(figsize=(6, 4.5))
    plt.figure(figsize=(5,5))
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    plt.scatter(x, y, color='green', label=label_name, s=15, alpha=0.6)
    try:
        slope, intercept = np.polyfit(x, y, 1)
        x_vals = np.linspace(np.min(x), np.max(x), 100)
        y_vals = slope * x_vals + intercept
        pearson_corr, _ = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)
        plt.plot(x_vals, y_vals, 'r-', label=f"Regression (Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f})")
    except Exception as e:
        print("Error in regression:", e)

    if type_exp == "diff": 
        plt.xlabel("Average expression levels differences (YFP fluorescence, Alt - Ref)", fontsize=10)
        plt.ylabel("DREAM-RNN model prediction differences (Alt - Ref)", fontsize=10)
    elif type_exp == "flatten":
        plt.xlabel("Average expression levels (YFP fluorescence)")
        plt.ylabel("DREAM-RNN Model Prediction")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved scatter plot to {filename}")

def process_seq_type(seq_type, pairs_csv, sample_tsv, gt_full, pred_full, nice_seq_name):
    """
    Extract subset pairs using the sampling approach, compute dual scores, and plot scatter plots.
    Uses the "nice" sequence name for titles and legends.
    """
    full_pairs = load_pairs(pairs_csv)
    sample_indices = load_sample_indices(sample_tsv)
    subset_pairs = get_subset_pairs(full_pairs, sample_indices)
    
    (gt_dual, pred_dual, logSED_scores,
     gt_alt, gt_ref, pred_alt, pred_ref) = compute_dual_scores(subset_pairs, gt_full, pred_full)
    
    print(f"{nice_seq_name} - Number of pairs: {len(subset_pairs)}; Average logSED: {np.mean(logSED_scores):.3f}")
    
    # Scatter plot for dual scores (log fold-changes)
    # title_dual = f"{nice_seq_name}: \nDREAM-RNN Model vs Ground Truth log₂ Fold-Change"
    title_dual = f"{nice_seq_name}: \nDREAM-RNN Model vs Ground Truth ALT / REF Differences"
    filename_dual = os.path.join(OUTPUT_DIR, "scatterplots", seq_type, f"scatter_dual_{seq_type}.png")
    plot_scatter(gt_dual, pred_dual, title_dual, filename_dual, nice_seq_name, type_exp="diff")

    # Flattened raw scores
    flat_indices = [idx for pair in subset_pairs for idx in pair]
    gt_flat   = gt_full[flat_indices]
    pred_flat = pred_full[flat_indices]
    title_flat = f"{nice_seq_name} (Flattened): \nDREAM-RNN vs Ground Truth"
    filename_flat = os.path.join(OUTPUT_DIR, "scatterplots", seq_type, f"scatter_flat_{seq_type}.png")
    plot_scatter(gt_flat, pred_flat, title_flat, filename_flat, nice_seq_name, type_exp="flatten")

def main():
    # Load full ground truth and model predictions.
    gt_full   = load_expression(os.path.join(ROOT_DIR, 'filtered_test_data_with_MAUDE_expression.txt'))
    pred_full = load_expression("/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/random-promoter-dream-challenge-2022/data/DREAM-RNN_output.txt")
    # pred_full = load_expression(os.path.join(ROOT_DIR, 'sample_submission.txt'))
    
    csv_indices_dict = {
        "all_SNVs_seqs": (
            os.path.join(ROOT_DIR, 'test_subset_ids', 'all_SNVs_seqs.csv'),
            os.path.join(ROOT_DIR, 'test_subset_ids', 'fix', 'all_SNVs_seqs_sample_ids.tsv')
        ),
        "motif_perturbation": (
            os.path.join(ROOT_DIR, 'test_subset_ids', 'motif_perturbation.csv'),
            os.path.join(ROOT_DIR, 'test_subset_ids', 'fix', 'motif_perturbation_sample_ids.tsv')
        ),
        "motif_tiling_seqs": (
            os.path.join(ROOT_DIR, 'test_subset_ids', 'motif_tiling_seqs.csv'),
            os.path.join(ROOT_DIR, 'test_subset_ids', 'fix', 'motif_tiling_seqs_sample_ids.tsv')
        )
    }
    
    seq_name_mapping = {
        "all_SNVs_seqs": "SNV Sequences",
        "motif_perturbation": "Motif Perturbation Sequences",
        "motif_tiling_seqs": "Motif Tiling Sequences"
    }
    
    for seq_type, (csv_file, sample_file) in csv_indices_dict.items():
        print(f"\nProcessing sequence type: {seq_type}")
        nice_seq_name = seq_name_mapping.get(seq_type, seq_type)
        process_seq_type(seq_type, csv_file, sample_file, gt_full, pred_full, nice_seq_name)

if __name__ == "__main__":
    main()
