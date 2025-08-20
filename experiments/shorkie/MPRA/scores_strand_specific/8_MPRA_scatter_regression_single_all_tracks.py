import numpy as np
import pandas as pd
import csv
import os
import glob
import re
import json
import pickle
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt

#########################################
# Global Directories and Flags
#########################################
# Use the stranded results directory
BASE_INPUT_DIR = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded'
BASE_OUTPUT_DIR = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/viz'

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Global flag to control per–gene visualization.
VISUALIZE_INDIVIDUAL = False

#########################################
# Helper Function to Map Sequence Type
#########################################
def get_seq_type_label(seq_type):
    mapping = {
        "yeast_seqs": "Yeast Sequence",
        "high_exp_seqs": "High Expression Sequence",
        "low_exp_seqs": "Low Expression Sequence",
        "challenging_seqs": "Challenging Sequence",
        "all_random_seqs": "Random Sequence",
    }
    return mapping.get(seq_type, seq_type)

#########################################
# 1. Load Ground Truth and CSV Data
#########################################
root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/'

# Read test data and corresponding expressions.
filename = os.path.join(root_dir, 'filtered_test_data_with_MAUDE_expression.txt')
with open(filename) as f:
    reader = csv.reader(f, delimiter="\t")
    lines = list(reader)

filtered_tagged_sequences = [line[0] for line in lines]
expressions_str = [line[1] for line in lines]
GROUND_TRUTH_EXP = np.array([float(val) for val in expressions_str])
print("Total ground truth sequences:", len(GROUND_TRUTH_EXP))
print("Max expression:", np.max(GROUND_TRUTH_EXP))
print("Min expression:", np.min(GROUND_TRUTH_EXP))


# Load CSV files for all sequence types.
# --- Challenging sequences ---
# 1. Load the complete challenging sequences CSV.
challenging_csv_path = os.path.join(root_dir, 'test_subset_ids/challenging_seqs.csv')
df_challenging = pd.read_csv(challenging_csv_path)

# 2. Load the sample IDs TSV file.
challenging_sample_path = os.path.join(root_dir, 'test_subset_ids/fix/challenging_seqs_sample_ids.tsv')
df_challenging_sample = pd.read_csv(challenging_sample_path, sep="\t")

# 3. Subtract 1 from the sampled row indices (if they are 1-indexed).
sample_indices_challenging = df_challenging_sample['original_row_id'].values - 1
print("sample_indices_challenging: ", sample_indices_challenging)

# 4. Use these indices to select rows from df_challenging.
sampled_df_challenging = df_challenging.iloc[sample_indices_challenging]

# 5. Extract the 'pos' values.
final_challenging = sampled_df_challenging['pos'].values
print("final_challenging: ", final_challenging)

# 1. Load the complete all_random sequences CSV.
all_random_csv_path = os.path.join(root_dir, 'test_subset_ids/all_random_seqs.csv')
df_all_random = pd.read_csv(all_random_csv_path)

# 2. Load the sample IDs TSV file.
all_random_sample_path = os.path.join(root_dir, 'test_subset_ids/fix/all_random_seqs_sample_ids.tsv')
df_all_random_sample = pd.read_csv(all_random_sample_path, sep="\t")

# 3. Subtract 1 from the sampled row indices.
sample_indices_all_random = df_all_random_sample['original_row_id'].values - 1
print("sample_indices_all_random: ", sample_indices_all_random)

# 4. Use these indices to select rows from df_all_random.
sampled_df_all_random = df_all_random.iloc[sample_indices_all_random]

# 5. Extract the 'pos' values.
final_all_random = sampled_df_all_random['pos'].values
print("final_all_random: ", final_all_random)


# Existing files:
df_high = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/high_exp_seqs.csv'))
final_high = df_high['pos'].values

df_low = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/low_exp_seqs.csv'))
final_low = df_low['pos'].values

df_yeast = pd.read_csv(os.path.join(root_dir, 'test_subset_ids/yeast_seqs.csv'))
final_yeast = df_yeast['pos'].values

print("Challenging sequences:", len(final_challenging))
print("All random sequences:", len(final_all_random))
print("Yeast sequences:", len(final_yeast))
print("High expression sequences:", len(final_high))
print("Low expression sequences:", len(final_low))

# Updated dictionary to include all sequence types.
csv_indices_dict = {
    "challenging_seqs": final_challenging,
    "all_random_seqs": final_all_random,
    "yeast_seqs": final_yeast,
    "high_exp_seqs": final_high,
    "low_exp_seqs": final_low
}

#########################################
# 2. Functions to Process NPZ Files (Predictions)
#########################################
def get_insertion_position(fname):
    """
    Extract the insertion position from the filename.
    Mapping: insertion_position = 100 + (index * 10)
    e.g. '_context_0_' -> 100 bp, '_context_1_' -> 110 bp, etc.
    """
    base = os.path.basename(fname)
    m = re.search(r'_context_(\d+)_', base)
    if m:
        index = int(m.group(1))
        return 100 + index * 10
    return 0

def process_gene_plots(target_gene, gene_name, seq_type, input_dir, output_dir):
    """
    For a given gene, load all NPZ files (each corresponding to a different insertion position)
    from the given input directory (which should be strand–specific), compute the predicted score 
    for each sequence by averaging over tracks, and return a dictionary with per–distance results 
    plus the overall averaged prediction.
    """
    npz_files = glob.glob(os.path.join(input_dir, f"{target_gene}_context_*.npz"))
    if not npz_files:
        print(f"No NPZ files found for gene {target_gene} in {input_dir}.")
        return None

    npz_files = sorted(npz_files, key=get_insertion_position)
    insertion_positions = []
    overall_means = []
    overall_stds = []
    predicted_scores_list = []
    results_by_distance = {}

    for file_path in npz_files:
        distance = get_insertion_position(file_path)
        print(f"Processing file: {file_path} (distance {distance} bp)")
        insertion_positions.append(distance)
        data = np.load(file_path)
        logSED = data["logSED"]  # shape: (num_sequences, num_tracks)
        print(f"Loaded logSED shape: {logSED.shape}")
        # Here we assume the order of sequences matches the CSV order.
        order = np.argsort(np.arange(logSED.shape[0]))
        logSED_sorted = logSED[order, :]
        seq_means = np.mean(logSED_sorted, axis=1)
        predicted_scores_list.append(seq_means)
        
        mean_val = np.mean(seq_means)
        std_val = np.std(seq_means)
        overall_means.append(mean_val)
        overall_stds.append(std_val)
        
        results_by_distance[distance] = {
            "predicted_scores": seq_means,
            "mean": mean_val,
            "std": std_val,
            "file": file_path
        }
    all_seq_means = np.array(predicted_scores_list)  # shape: (n_distances, n_sequences)
    avg_predicted_scores = np.mean(all_seq_means, axis=0)
    return {
        "insertion_positions": np.array(insertion_positions),
        "overall_means": np.array(overall_means),
        "overall_stds": np.array(overall_stds),
        "results_by_distance": results_by_distance,
        "gene_name": gene_name,
        "target_gene": target_gene,
        "seq_type": seq_type,
        "strand": None,  # to be set later
        "average_predicted_scores": avg_predicted_scores
    }

#########################################
# 3. Functions for Correlation Analysis & Scatter Plots
#########################################
def compute_correlation_for_gene(gene_stats, ground_truth, transform=False):
    """
    Compute Pearson and Spearman correlations using the overall (averaged) predicted scores.
    Ground truth values are extracted using CSV indices.
    Optionally log-transform both arrays (using a common shift) before computing correlations.
    """
    epsilon = 1e-8
    seq_type = gene_stats['seq_type']
    if seq_type in csv_indices_dict:
        indices = csv_indices_dict[seq_type]
    else:
        raise ValueError(f"Unknown seq_type {seq_type} for gene {gene_stats['gene_name']}")
    
    pred = gene_stats['average_predicted_scores']
    gt = ground_truth[indices]
    
    if transform:
        combined = np.concatenate((gt, pred))
        shift = 1 - np.min(combined) if np.min(combined) <= 0 else 0
        gt = np.log(gt + shift + epsilon)
        pred = np.log(pred + shift + epsilon)
    
    pearson, _ = pearsonr(gt, pred)
    spearman, _ = spearmanr(gt, pred)
    return pearson, spearman, gt, pred

def plot_scatter_with_regression(gt, pred, title, filename, corr_pearson, corr_spearman, seq_type=None):
    """
    Create a scatter plot comparing ground truth versus predicted scores.
    A regression line is computed and overlaid. If seq_type is provided, the blue dots are labeled
    using the corresponding friendly name (e.g. "High Expression Sequence").
    """
    # plt.figure(figsize=(6, 4.5))
    plt.figure(figsize=(5, 5))
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    # Use the friendly label if seq_type is provided.
    if seq_type is not None:
        scatter_label = get_seq_type_label(seq_type)
    else:
        scatter_label = "Data"
    plt.scatter(gt, pred, label=scatter_label, s=15, alpha=0.6)
    try:
        slope, intercept = np.polyfit(gt, pred, 1)
        x_reg = np.linspace(np.min(gt), np.max(gt), 100)
        y_reg = slope * x_reg + intercept
        reg_label = f"Regression (Pearson: {corr_pearson:.3f}, Spearman: {corr_spearman:.3f})"
        plt.plot(x_reg, y_reg, 'r-', label=reg_label)
    except Exception as e:
        print("Regression computation error:", e)
    plt.xlabel("MeanYFP (Experimental Measurements)")
    plt.ylabel("Shorkie Predicted logSED")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

#########################################
# 4. Functions for Per-Distance Scatter & Trend Plots
#########################################
def visualize_scatter_by_distance_gene(gene_stats, ground_truth, result_base_dir, transform=False):
    """
    For a single gene, generate a scatter plot for each insertion distance comparing the predicted scores 
    (for that distance) with the corresponding ground truth. A regression line is overlaid and correlations are shown.
    Plots are saved under:
       <result_base_dir>/scatterplots/<seq_type>/individual/<gene_name>/<distance>bp/
    """
    seq_type = gene_stats["seq_type"]
    gene_name = gene_stats["gene_name"]
    target_gene = gene_stats["target_gene"]
    strand = gene_stats["strand"]
    # Use the friendly label in the title.
    friendly_label = get_seq_type_label(seq_type)
    indices = csv_indices_dict[seq_type]
    gt_all = ground_truth[indices]
    
    for distance, res in sorted(gene_stats["results_by_distance"].items()):
        pred = res["predicted_scores"]
        if len(pred) != len(gt_all):
            print(f"Warning: length mismatch for {gene_name} at {distance} bp.")
            continue
        if transform:
            epsilon = 1e-8
            combined = np.concatenate((gt_all, pred))
            shift = 1 - np.min(combined) if np.min(combined) <= 0 else 0
            gt_plot = np.log(gt_all + shift + epsilon)
            pred_plot = np.log(pred + shift + epsilon)
        else:
            gt_plot = gt_all
            pred_plot = pred
        if np.std(gt_plot) == 0 or np.std(pred_plot) == 0:
            corr_pearson, corr_spearman = 0.0, 0.0
        else:
            corr_pearson, _ = pearsonr(gt_plot, pred_plot)
            corr_spearman, _ = spearmanr(gt_plot, pred_plot)
        title = f"{gene_name} ({target_gene}), strand {strand}, {friendly_label}\nDistance: {distance} bp"
        out_dir = os.path.join(result_base_dir, "scatterplots", seq_type, "individual", gene_name, f"{distance}bp")
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"scatter_{gene_name}_{distance}bp{'_log' if transform else ''}.png")
        plot_scatter_with_regression(gt_plot, pred_plot, title, filename, corr_pearson, corr_spearman, seq_type)
        print(f"Saved scatter plot for gene {gene_name} at {distance} bp to {filename}")


def plot_trend_correlation_gene(gene_stats, ground_truth, result_base_dir, transform=False):
    """
    For a single gene, compute and plot the trend of Pearson and Spearman correlations across insertion distances.
    The plot is saved under:
       <result_base_dir>/correlation_trends/<seq_type>/individual/<gene_name>/
    """
    seq_type = gene_stats["seq_type"]
    gene_name = gene_stats["gene_name"]
    target_gene = gene_stats["target_gene"]
    strand = gene_stats["strand"]
    indices = csv_indices_dict[seq_type]
    gt_all = ground_truth[indices]
    
    distances = sorted(gene_stats["results_by_distance"].keys())
    pearson_vals = []
    spearman_vals = []
    
    for distance in distances:
        pred = gene_stats["results_by_distance"][distance]["predicted_scores"]
        if len(pred) != len(gt_all):
            pearson_vals.append(np.nan)
            spearman_vals.append(np.nan)
            continue
        if transform:
            epsilon = 1e-8
            combined = np.concatenate((gt_all, pred))
            shift = 1 - np.min(combined) if np.min(combined) <= 0 else 0
            gt_vals = np.log(gt_all + shift + epsilon)
            pred_vals = np.log(pred + shift + epsilon)
        else:
            gt_vals = gt_all
            pred_vals = pred
        if np.std(gt_vals)==0 or np.std(pred_vals)==0:
            pearson_vals.append(0.0)
            spearman_vals.append(0.0)
        else:
            p_val, _ = pearsonr(gt_vals, pred_vals)
            s_val, _ = spearmanr(gt_vals, pred_vals)
            pearson_vals.append(p_val)
            spearman_vals.append(s_val)
    
    if distances:
        plt.figure(figsize=(6, 4.5))
        plt.plot(distances, pearson_vals, marker='o', label='Pearson')
        plt.plot(distances, spearman_vals, marker='s', label='Spearman')
        plt.xlabel("Insertion Distance (bp)")
        plt.ylabel("Correlation Coefficient")
        trans_label = " (log-transformed)" if transform else ""
        title = f"Correlation Trend for {gene_name} ({target_gene}), strand {strand}{trans_label}"
        plt.title(title)
        plt.legend()
        plt.grid(True)
        out_dir = os.path.join(result_base_dir, "correlation_trends", seq_type, "individual", gene_name)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"correlation_trend_{gene_name}_{strand}{'_log' if transform else ''}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved correlation trend plot for gene {gene_name} to {filename}")
    else:
        print(f"No valid distances for correlation trend plot for gene {gene_name}")


#########################################
# 5. Functions for Analysis of Individual & Group Genes
#########################################
def analyze_and_plot_individual_genes(aggregated_data, ground_truth, result_base_dir, transform=False):
    suffix = "_log" if transform else ""
    if not VISUALIZE_INDIVIDUAL:
        print("Skipping individual gene visualization as per flag.")
        return
    for strand in aggregated_data:
        for gene_stats in aggregated_data[strand]:
            gene_name = gene_stats['gene_name']
            target_gene = gene_stats['target_gene']
            seq_type = gene_stats['seq_type']
            friendly_label = get_seq_type_label(seq_type)
            title = f"{gene_name} ({target_gene}), strand {strand}, {friendly_label}"
            pearson, spearman, gt, pred = compute_correlation_for_gene(gene_stats, ground_truth, transform)
            out_dir = os.path.join(result_base_dir, "scatterplots", seq_type, "individual")
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f"scatter_{seq_type}_{gene_name}_{strand}{suffix}.png")
            plot_scatter_with_regression(gt, pred, title, filename, pearson, spearman, seq_type)
            print(f"Saved overall scatter plot for gene {gene_name} ({target_gene}), strand {strand}{suffix} to {filename}")
            visualize_scatter_by_distance_gene(gene_stats, ground_truth, result_base_dir, transform)
            plot_trend_correlation_gene(gene_stats, ground_truth, result_base_dir, transform)

def visualize_scatter_by_distance_group(gene_list, ground_truth, seq_type, group_label, result_base_dir, transform=False):
    """
    For a group of genes, for each insertion distance, aggregate predicted scores
    and generate a scatter plot comparing the aggregated predictions with ground truth.
    Plots are saved under:
         <result_base_dir>/scatterplots/<seq_type>/aggregated/<group_label>/<distance>bp/
    """
    distances = sorted(gene_list[0]["results_by_distance"].keys())
    friendly_label = get_seq_type_label(seq_type)
    for distance in distances:
        preds_list = []
        for gene_stats in gene_list:
            if distance in gene_stats["results_by_distance"]:
                preds_list.append(gene_stats["results_by_distance"][distance]["predicted_scores"])
        if not preds_list:
            continue
        preds_array = np.array(preds_list)
        aggregated_pred = np.mean(preds_array, axis=0)
        indices = csv_indices_dict[seq_type]
        aggregated_gt = ground_truth[indices]
        if transform:
            epsilon = 1e-8
            combined = np.concatenate((aggregated_gt, aggregated_pred))
            shift = 1 - np.min(combined) if np.min(combined) <= 0 else 0
            gt_plot = np.log(aggregated_gt + shift + epsilon)
            pred_plot = np.log(aggregated_pred + shift + epsilon)
        else:
            gt_plot = aggregated_gt
            pred_plot = aggregated_pred
        if np.std(gt_plot)==0 or np.std(pred_plot)==0:
            corr_pearson, corr_spearman = 0.0, 0.0
        else:
            corr_pearson, _ = pearsonr(gt_plot, pred_plot)
            corr_spearman, _ = spearmanr(gt_plot, pred_plot)
        title = f"{friendly_label}:\nAggregated across {group_label} genes, {distance} bp"
        out_dir = os.path.join(result_base_dir, "scatterplots", seq_type, "aggregated", group_label, f"{distance}bp")
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"scatter_aggregated_{group_label}_{seq_type}_{distance}bp{'_log' if transform else ''}.png")
        plot_scatter_with_regression(gt_plot, pred_plot, title, filename, corr_pearson, corr_spearman, seq_type)
        print(f"Saved aggregated scatter plot for {group_label} genes at {distance} bp to {filename}")

def plot_trend_correlation_group(gene_list, ground_truth, seq_type, group_label, result_base_dir, transform=False):
    """
    For a group of genes, aggregate predicted scores at each insertion distance,
    compute correlations with ground truth, and plot the trend of Pearson and Spearman coefficients.
    """
    distances = sorted(gene_list[0]["results_by_distance"].keys())
    agg_pearsons = []
    agg_spearmans = []
    
    for distance in distances:
        preds_list = []
        for gene_stats in gene_list:
            if distance in gene_stats["results_by_distance"]:
                preds_list.append(gene_stats["results_by_distance"][distance]["predicted_scores"])
        if not preds_list:
            agg_pearsons.append(np.nan)
            agg_spearmans.append(np.nan)
            continue
        preds_array = np.array(preds_list)
        aggregated_pred = np.mean(preds_array, axis=0)
        indices = csv_indices_dict[seq_type]
        aggregated_gt = ground_truth[indices]
        if transform:
            epsilon = 1e-8
            combined = np.concatenate((aggregated_gt, aggregated_pred))
            shift = 1 - np.min(combined) if np.min(combined) <= 0 else 0
            aggregated_gt = np.log(aggregated_gt + shift + epsilon)
            aggregated_pred = np.log(aggregated_pred + shift + epsilon)
        if np.std(aggregated_gt)==0 or np.std(aggregated_pred)==0:
            agg_pearsons.append(0.0)
            agg_spearmans.append(0.0)
        else:
            p_val, _ = pearsonr(aggregated_gt, aggregated_pred)
            s_val, _ = spearmanr(aggregated_gt, aggregated_pred)
            agg_pearsons.append(p_val)
            agg_spearmans.append(s_val)
    
    plt.figure(figsize=(6, 4.5))
    plt.plot(distances, agg_pearsons, marker='o', label='Pearson')
    plt.plot(distances, agg_spearmans, marker='s', label='Spearman')
    plt.xlabel("Insertion Distance (bp)")
    plt.ylabel("Correlation Coefficient")
    trans_label = " (log-transformed)" if transform else ""
    friendly_label = get_seq_type_label(seq_type)
    plt.title(f"Aggregated Correlation Trend for {group_label} genes, {friendly_label}{trans_label}")
    plt.legend()
    plt.grid(True)
    out_dir = os.path.join(result_base_dir, "correlation_trends", seq_type, "aggregated", group_label)
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"correlation_trend_aggregated_{group_label}_{seq_type}{'_log' if transform else ''}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved aggregated correlation trend plot for {group_label} genes to {filename}")

def analyze_and_plot_group(aggregated_data, ground_truth, seq_type, group_label, result_base_dir, transform=False):
    suffix = "_log" if transform else ""
    gene_list = []
    if group_label == "positive":
        gene_list = [gs for gs in aggregated_data.get('+', []) if gs['seq_type'] == seq_type]
    elif group_label == "negative":
        gene_list = [gs for gs in aggregated_data.get('-', []) if gs['seq_type'] == seq_type]
    elif group_label == "all":
        gene_list = [gs for strand in aggregated_data for gs in aggregated_data[strand] if gs['seq_type'] == seq_type]
    
    if not gene_list:
        print(f"No genes found for {group_label} in {seq_type}")
        return

    predicted_scores_array = np.array([gs["average_predicted_scores"] for gs in gene_list])
    # Average the predicted scores across all genes in the group.
    aggregated_pred = np.mean(predicted_scores_array, axis=0)
    indices = csv_indices_dict[seq_type]
    aggregated_gt = ground_truth[indices]
    if transform:
        epsilon = 1e-8
        combined = np.concatenate((aggregated_gt, aggregated_pred))
        shift = 1 - np.min(combined) if np.min(combined) <= 0 else 0
        aggregated_gt = np.log(aggregated_gt + shift + epsilon)
        aggregated_pred = np.log(aggregated_pred + shift + epsilon)
    if np.std(aggregated_gt)==0 or np.std(aggregated_pred)==0:
        overall_pearson, overall_spearman = 0.0, 0.0
    else:
        overall_pearson, _ = pearsonr(aggregated_gt, aggregated_pred)
        overall_spearman, _ = spearmanr(aggregated_gt, aggregated_pred)
    friendly_label = get_seq_type_label(seq_type)
    title = f"Aggregated Scatter Plot for {group_label} genes, {friendly_label}"
    out_dir = os.path.join(result_base_dir, "scatterplots", seq_type, "aggregated")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"scatter_aggregated_{group_label}_{seq_type}{suffix}.png")
    plot_scatter_with_regression(aggregated_gt, aggregated_pred, title, filename, overall_pearson, overall_spearman, seq_type)
    print(f"Saved overall aggregated scatter plot for {group_label} genes, {seq_type}{suffix} to {filename}")
    visualize_scatter_by_distance_group(gene_list, ground_truth, seq_type, group_label, result_base_dir, transform)
    plot_trend_correlation_group(gene_list, ground_truth, seq_type, group_label, result_base_dir, transform)

#########################################
# 6. Main Processing and Analysis
#########################################
def main():
    result_base_dir = BASE_OUTPUT_DIR  # using the stranded results directory
    aggregated_data = {'+': [], '-': []}
    # Process the three sequence types.
    seq_types = ["challenging_seqs", "all_random_seqs", "yeast_seqs", "high_exp_seqs", "low_exp_seqs"]

    # Define gene mappings for positive and negative strands.
    pos_gene_mapping = {
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
    neg_gene_mapping = {
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
    
    # Loop over each sequence type and process genes for both strands.
    for seq_type in seq_types:
        # Process positive strand genes.
        strand = '+'
        for gene_name, target_gene in pos_gene_mapping.items():
            print(f"\nProcessing {seq_type} gene {target_gene} (symbol: {gene_name}) on strand {strand}...")
            # Construct input directory using stranded nomenclature.
            input_dir = os.path.join(BASE_INPUT_DIR, 'all_seq_types', seq_type, f"{gene_name}_{target_gene}_pos_outputs")
            output_dir = os.path.join(input_dir, "plots")
            os.makedirs(output_dir, exist_ok=True)
            gene_stats = process_gene_plots(target_gene, gene_name, seq_type, input_dir=input_dir, output_dir=output_dir)
            if gene_stats is not None:
                gene_stats['strand'] = strand
                aggregated_data[strand].append(gene_stats)

        # Process negative strand genes.
        strand = '-'
        for gene_name, target_gene in neg_gene_mapping.items():
            print(f"\nProcessing {seq_type} gene {target_gene} (symbol: {gene_name}) on strand {strand}...")
            input_dir = os.path.join(BASE_INPUT_DIR, 'all_seq_types', seq_type, f"{gene_name}_{target_gene}_neg_outputs")
            output_dir = os.path.join(input_dir, "plots")
            os.makedirs(output_dir, exist_ok=True)
            gene_stats = process_gene_plots(target_gene, gene_name, seq_type, input_dir=input_dir, output_dir=output_dir)
            if gene_stats is not None:
                gene_stats['strand'] = strand
                aggregated_data[strand].append(gene_stats)
    
    for strand in aggregated_data:
        print(f"\nTotal genes processed on strand {strand}: {len(aggregated_data[strand])}")
        for gene_stats in aggregated_data[strand]:
            print(f"  - {gene_stats['gene_name']} ({gene_stats['target_gene']}) in {gene_stats['seq_type']}")
            print(f"    - Insertion positions: {gene_stats['insertion_positions']}")
            print(f"    - Overall means: {gene_stats['overall_means']}")
            print(f"    - Overall stds: {gene_stats['overall_stds']}")
            print(f"    - Average predicted scores: {gene_stats['average_predicted_scores'][:5]}... (total {len(gene_stats['average_predicted_scores'])})")
            # for distance, res in gene_stats["results_by_distance"].items():
            #     print(lf"    - Distance {distance} bp: mean={res['mean']:.3f}, std={res['std']:.3f}, file={res['file']}")

    # Optionally save aggregated data for future use.
    with open(os.path.join(BASE_OUTPUT_DIR, "aggregated_MPRA_data.pkl"), "wb") as f:
        pickle.dump(aggregated_data, f)
    print(f"Aggregated data saved to {os.path.join(BASE_OUTPUT_DIR, 'aggregated_MPRA_data.pkl')}")
    
    # # Analyze individual genes (both original and log-transformed).
    # analyze_and_plot_individual_genes(aggregated_data, GROUND_TRUTH_EXP, result_base_dir, transform=False)
    # analyze_and_plot_individual_genes(aggregated_data, GROUND_TRUTH_EXP, result_base_dir, transform=True)
    
    # Analyze aggregated groups for each sequence type.
    for seq_type in seq_types:
        for group_label in ["positive", "negative", "all"]:
            analyze_and_plot_group(aggregated_data, GROUND_TRUTH_EXP, seq_type, group_label, result_base_dir, transform=False)
            # analyze_and_plot_group(aggregated_data, GROUND_TRUTH_EXP, seq_type, group_label, result_base_dir, transform=True)

if __name__ == "__main__":
    main()
