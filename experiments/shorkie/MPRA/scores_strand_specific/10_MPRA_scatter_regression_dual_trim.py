import numpy as np
import pandas as pd
import csv
import os
import glob
import re
import pickle
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

######################################
# 1. Load Ground Truth and CSV Data  #
######################################
root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/'
TRIM_DOTS = False       # Whether to trim furthest‐away points (unchanged)
ZERO_THRESHOLD = 1e-3  # “Close to y=0” threshold for predicted scores
    
# Map the original sequence keys to nicer display names.
SEQ_NAME_MAPPING = {
    "all_SNVs_seqs": "SNV Sequences",
    "motif_perturbation": "Motif Perturbation Sequences",
    "motif_tiling_seqs": "Motif Tiling Sequences"
}

# Read test data and corresponding expressions
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

# Load sequence index mappings
df = pd.read_csv(f'{root_dir}test_subset_ids/all_SNVs_seqs.csv')
SNVs_alt = list(df['alt_pos'])
SNVs_ref = list(df['ref_pos'])
SNVs = list(zip(SNVs_alt, SNVs_ref))
SNVs_idx_df = pd.read_csv(f'{root_dir}test_subset_ids/fix/all_SNVs_seqs_sample_ids.tsv', sep='\t')

df = pd.read_csv(f'{root_dir}test_subset_ids/motif_perturbation.csv')
motif_perturbation_alt = list(df['alt_pos'])
motif_perturbation_ref = list(df['ref_pos'])
motif_perturbation = list(zip(motif_perturbation_alt, motif_perturbation_ref))
motif_perturbation_idx_df = pd.read_csv(f'{root_dir}test_subset_ids/fix/motif_perturbation_sample_ids.tsv', sep='\t')

df = pd.read_csv(f'{root_dir}test_subset_ids/motif_tiling_seqs.csv')
motif_tiling_alt = list(df['alt_pos'])
motif_tiling_ref = list(df['ref_pos'])
motif_tiling = list(zip(motif_tiling_alt, motif_tiling_ref))
motif_tiling_idx_df = pd.read_csv(f'{root_dir}test_subset_ids/fix/motif_tiling_seqs_sample_ids.tsv', sep='\t')

print("\n\nCounts:")
print("SNVs:", len(SNVs))
print("SNVs IDs:", len(SNVs_idx_df))
print("Motif Perturbation:", len(motif_perturbation))
print("Motif Perturbation IDs:", len(motif_perturbation_idx_df))
print("Motif Tiling:", len(motif_tiling))
print("Motif Tiling IDs:", len(motif_tiling_idx_df))
print("Total:", len(SNVs) + len(motif_perturbation) + len(motif_tiling))
print("Total IDs:", len(SNVs_idx_df) + len(motif_perturbation_idx_df) + len(motif_tiling_idx_df))

# Subsample sequences
sample_indices_SNVs = SNVs_idx_df['original_row_id'].astype(int).values - 1
SNVs_subset = [SNVs[i] for i in sample_indices_SNVs]

sample_indices_mp = motif_perturbation_idx_df['original_row_id'].astype(int).values - 1
motif_perturbation_subset = [motif_perturbation[i] for i in sample_indices_mp]

sample_indices_mt = motif_tiling_idx_df['original_row_id'].astype(int).values - 1
motif_tiling_subset = [motif_tiling[i] for i in sample_indices_mt]

print("\n\nSubsampled counts:")
print("SNVs:", len(SNVs_subset))
print("Motif Perturbation:", len(motif_perturbation_subset))
print("Motif Tiling:", len(motif_tiling_subset))
print("Total:", len(SNVs_subset) + len(motif_perturbation_subset) + len(motif_tiling_subset))

csv_indices_dict = {
    "all_SNVs_seqs": SNVs_subset,
    "motif_perturbation": motif_perturbation_subset,
    "motif_tiling_seqs": motif_tiling_subset
}

##############################################
# 2. Functions to Process NPZ Files (Predictions)
##############################################
def get_insertion_position(fname):
    base = os.path.basename(fname)
    m = re.search(r'_ctx(\d+)_', base)
    if m:
        index = int(m.group(1))
        return 100 + index * 10
    return 0


def process_gene_plots(target_gene, gene_name, seq_type, input_dir, output_dir):
    npz_files = glob.glob(os.path.join(input_dir, f"{target_gene}_ctx*.npz"))
    if not npz_files:
        print(f"No NPZ files found for gene {target_gene} in {input_dir}.")
        return None

    npz_files = sorted(npz_files, key=get_insertion_position)
    results_by_distance = {}
    all_insertion_positions = []
    overall_means = []
    overall_stds = []
    predicted_scores_list = []
    
    for file_path in npz_files:
        distance = get_insertion_position(file_path)
        all_insertion_positions.append(distance)
        print(f"\tLoading data from {file_path} for distance {distance} bp...")
        data = np.load(file_path)

        # 'logSED': sampled_logSED[mask, :],
        # 'logSED_ALT_ORIG': sampled_ALT[mask, :],
        # 'logSED_REF_ORIG': sampled_REF[mask, :],
        # logSED = data["logSED"]
        logSED_ALT_ORIG = data["logSED_ALT_ORIG"]
        logSED_REF_ORIG = data["logSED_REF_ORIG"]
        logSED = logSED_ALT_ORIG - logSED_REF_ORIG
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
    
    predicted_scores_array = np.array(predicted_scores_list)
    average_predicted_scores = np.mean(predicted_scores_array, axis=0)
    
    return {
        "results_by_distance": results_by_distance,
        "all_insertion_positions": np.array(all_insertion_positions),
        "overall_means": np.array(overall_means),
        "overall_stds": np.array(overall_stds),
        "average_predicted_scores": average_predicted_scores,
        "gene_name": gene_name,
        "target_gene": target_gene,
        "seq_type": seq_type,
        "strand": None
    }

##############################################
# 3. Functions for Correlation Analysis & Plotting
##############################################
def plot_scatter(gt, pred, title, filename, corr_pearson=None, corr_spearman=None):
    gt = np.array(gt, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)
    valid = np.isfinite(gt) & np.isfinite(pred)

    # plt.figure(figsize=(6, 4.5))
    plt.figure(figsize=(5,5))
    plt.scatter(gt[valid], pred[valid], s=15, alpha=0.6)
    
    if np.sum(valid) >= 2:
        try:
            m, b = np.polyfit(gt[valid], pred[valid], 1)
            x_reg = np.linspace(np.min(gt[valid]), np.max(gt[valid]), 100)
            y_reg = m * x_reg + b
            reg_label = f"Pearson: {corr_pearson:.3f}, Spearman: {corr_spearman:.3f}" if (corr_pearson is not None and corr_spearman is not None) else "Regression line"
            plt.plot(x_reg, y_reg, color='red', linestyle='-', linewidth=2, label=reg_label)
        except np.linalg.LinAlgError:
            print("LinAlgError in polyfit, skipping regression line")
    else:
        print("Not enough valid points for regression line.")
        if corr_pearson is not None and corr_spearman is not None:
            dummy_line = Line2D([], [], color='red', label=f"Pearson: {corr_pearson:.3f}, Spearman: {corr_spearman:.3f}")
            plt.legend(handles=[dummy_line])
    
    plt.legend()
    # plt.xlabel("Log fold change in average expression levels (YFP fluorescence)", fontsize=9)
    plt.xlabel("Average expression levels differences (YFP fluorescence, Alt - Ref)", fontsize=8.5)
    # plt.ylabel("Shorkie Predicted logSED")
    plt.ylabel("Shorkie predicted logSED differences (Alt - Ref)", fontsize=10)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

##############################################
# 4. New Visualization Functions for Per-Distance Results
##############################################
def visualize_group_by_distance(gene_list, ground_truth, seq_type, group_label):
    if not gene_list:
        return
    distances = sorted(gene_list[0]["results_by_distance"].keys())
    for distance in distances:
        preds_list = []
        for gene_stats in gene_list:
            if distance in gene_stats["results_by_distance"]:
                preds_list.append(gene_stats["results_by_distance"][distance]["predicted_scores"])
        if not preds_list:
            continue
        preds_array = np.array(preds_list)
        aggregated_pred = np.mean(preds_array, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            aggregated_gt = np.array([
                # 0.0 if (diff := np.log2(GROUND_TRUTH_EXP[alt] + 1) - np.log2(GROUND_TRUTH_EXP[ref] + 1)) != diff else diff
                GROUND_TRUTH_EXP[alt] - GROUND_TRUTH_EXP[ref]
                for alt, ref in csv_indices_dict[seq_type]
            ], dtype=np.float64)
        mask = np.isfinite(aggregated_gt) & np.isfinite(aggregated_pred)
        aggregated_gt = aggregated_gt[mask]
        aggregated_pred = aggregated_pred[mask]

        # ----------------------------------------------------------------------------
        # * Remove any points where either gt or pred is exactly zero
        # ----------------------------------------------------------------------------
        non_zero_mask = (aggregated_gt != 0) & (aggregated_pred != 0)
        removed_count = len(aggregated_gt) - np.count_nonzero(non_zero_mask)
        if removed_count > 0:
            print(f"Removed {removed_count} points where either ground-truth or predicted score was exactly zero.")
        aggregated_gt = aggregated_gt[non_zero_mask]
        aggregated_pred = aggregated_pred[non_zero_mask]

        if len(aggregated_gt) < 2:
            print(f"Not enough data for group {group_label} at distance {distance} bp after trimming")
            continue
        if np.std(aggregated_gt) == 0 or np.std(aggregated_pred) == 0:
            corr_pearson, corr_spearman = 0.0, 0.0
        else:
            try:
                corr_pearson, _ = pearsonr(aggregated_gt, aggregated_pred)
                corr_spearman, _ = spearmanr(aggregated_gt, aggregated_pred)
            except Exception:
                corr_pearson, corr_spearman = np.nan, np.nan
        title = f"{SEQ_NAME_MAPPING[seq_type]}:\nAggregated across {group_label} genes, {distance} bp"
        out_dir = os.path.join(
            "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA",
            "results", "single_measurement_stranded", "viz",
            "scatterplots", seq_type, "aggregated", group_label, f"{distance}bp"
        )
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"scatter_aggregated_{group_label}_{seq_type}_{distance}bp.png")
        plot_scatter(aggregated_gt, aggregated_pred, title, filename, corr_pearson, corr_spearman)
        print(f"Saved group-level scatter plot for {group_label} genes, {seq_type}, {distance} bp to {filename}")

##############################################
# 5. New Functions for Trend Plots of Correlations
##############################################
def plot_trend_correlation_group(gene_list, ground_truth, seq_type, group_label):
    if not gene_list:
        return
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
        with np.errstate(divide='ignore', invalid='ignore'):
            aggregated_gt = np.array([
                # 0.0 if (diff := np.log2(GROUND_TRUTH_EXP[alt] + 1) - np.log2(GROUND_TRUTH_EXP[ref] + 1)) != diff else diff
                GROUND_TRUTH_EXP[alt] - GROUND_TRUTH_EXP[ref]
                for alt, ref in csv_indices_dict[seq_type]
            ], dtype=np.float64)
        mask = np.isfinite(aggregated_gt) & np.isfinite(aggregated_pred)
        aggregated_gt = aggregated_gt[mask]
        aggregated_pred = aggregated_pred[mask]

        # ----------------------------------------------------------------------------
        # * Remove any points where either gt or pred is exactly zero
        # ----------------------------------------------------------------------------
        non_zero_mask = (aggregated_gt != 0) & (aggregated_pred != 0)
        removed_count = len(aggregated_gt) - np.count_nonzero(non_zero_mask)
        if removed_count > 0:
            print(f"Removed {removed_count} points where either ground-truth or predicted score was exactly zero.")
        aggregated_gt = aggregated_gt[non_zero_mask]
        aggregated_pred = aggregated_pred[non_zero_mask]

        if len(aggregated_gt) < 2 or np.std(aggregated_gt)==0 or np.std(aggregated_pred)==0:
            agg_pearsons.append(0.0)
            agg_spearmans.append(0.0)
        else:
            try:
                gt_trim = aggregated_gt
                pred_trim = aggregated_pred
                p_val, _ = pearsonr(gt_trim, pred_trim)
                s_val, _ = spearmanr(gt_trim, pred_trim)
            except Exception:
                p_val, s_val = np.nan, np.nan
            agg_pearsons.append(p_val)
            agg_spearmans.append(s_val)
    
    plt.figure(figsize=(6, 4.5))
    plt.plot(distances, agg_pearsons, marker='o', label='Pearson')
    plt.plot(distances, agg_spearmans, marker='s', label='Spearman')
    plt.xlabel("Insertion Distance (bp)")
    plt.ylabel("Correlation Coefficient")
    plt.title(f"Aggregated Correlation Trend for {group_label} genes, {seq_type}")
    plt.grid(True)
    plt.legend()
    out_dir = os.path.join(
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA",
        "results", "single_measurement_stranded", "viz",
        "correlation_trends", seq_type, "aggregated", group_label
    )
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"correlation_trend_aggregated_{group_label}_{seq_type}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved aggregated correlation trend plot for {group_label} genes to {filename}")

##############################################
# 6. Analysis Functions
##############################################

def analyze_and_plot_group(aggregated_data, ground_truth, seq_type, group_label):
    if group_label == "positive":
        gene_list = [gs for gs in aggregated_data.get('+', []) if gs['seq_type'] == seq_type]
    elif group_label == "negative":
        gene_list = [gs for gs in aggregated_data.get('-', []) if gs['seq_type'] == seq_type]
    elif group_label == "all":
        gene_list = [gs for strand in aggregated_data for gs in aggregated_data[strand] if gs['seq_type'] == seq_type]
    else:
        gene_list = []
    
    if not gene_list:
        print(f"No genes found for {group_label} in {seq_type}")
        return

    predicted_scores_array = np.array([gs["average_predicted_scores"] for gs in gene_list])
    aggregated_pred = np.mean(predicted_scores_array, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        aggregated_gt = np.array([
            # 0.0 if (diff := np.log2(GROUND_TRUTH_EXP[alt] + 1) - np.log2(GROUND_TRUTH_EXP[ref] + 1)) != diff else diff
            GROUND_TRUTH_EXP[alt] - GROUND_TRUTH_EXP[ref]
            for alt, ref in csv_indices_dict[seq_type]
        ], dtype=np.float64)
    
    mask = np.isfinite(aggregated_gt) & np.isfinite(aggregated_pred)
    aggregated_gt = aggregated_gt[mask]
    aggregated_pred = aggregated_pred[mask]
    
    if len(aggregated_gt) < 2:
        print("Not enough data points after filtering non-finite values for correlation.")
        return

    # ----------------------------------------------------------------------------
    # 3) Extract “zero‐dots”: those with |predicted_score| <= ZERO_THRESHOLD
    # ----------------------------------------------------------------------------
    zero_indices = np.where(np.abs(aggregated_pred) <= ZERO_THRESHOLD)[0]
    if len(zero_indices) > 0:
        full_indices = np.arange(len(csv_indices_dict[seq_type]))
        valid_indices = full_indices[mask]
        zero_full_indices = valid_indices[zero_indices]  
        out_dir = os.path.join(
            "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA",
            "results", "single_measurement_stranded", "viz", "scatterplots", seq_type, "zero_dot_sequences"
        )
        os.makedirs(out_dir, exist_ok=True)
        csv_path_pred = os.path.join(out_dir, f"zero_dot_sequences_{group_label}_{seq_type}_pred.csv")
        with open(csv_path_pred, "w", newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow([
                "position_in_aggregated_array",
                "alt_index", "ref_index",
                "sequence_alt", "sequence_ref",
                "ground_truth_ratio", "predicted_score"
            ])
            for local_i, full_i in enumerate(zero_full_indices):
                alt_idx, ref_idx = csv_indices_dict[seq_type][full_i]
                seq_alt = filtered_tagged_sequences[alt_idx]
                seq_ref = filtered_tagged_sequences[ref_idx]
                gt_val = aggregated_gt[zero_indices[local_i]]
                pred_val = aggregated_pred[zero_indices[local_i]]
                writer.writerow([full_i, alt_idx, ref_idx, seq_alt, seq_ref, gt_val, pred_val])
        print(f"Wrote {len(zero_full_indices)} 'zero‐dot' predicted sequences to {csv_path_pred}")
    else:
        print("No points with |predicted_score| ≤ ZERO_THRESHOLD in this group.")
    
    # ----------------------------------------------------------------------------
    # 4) Extract “zero‐dots”: those with |ground_truth_ratio| <= ZERO_THRESHOLD
    # ----------------------------------------------------------------------------
    zero_gt_indices = np.where(np.abs(aggregated_gt) <= ZERO_THRESHOLD)[0]
    print(f"Found {len(zero_gt_indices)} points with |ground_truth_ratio| ≤ {ZERO_THRESHOLD} in this group.")
    if len(zero_gt_indices) > 0:
        full_indices = np.arange(len(csv_indices_dict[seq_type]))
        valid_indices = full_indices[mask]
        zero_gt_full_indices = valid_indices[zero_gt_indices]

        out_dir = os.path.join(
            "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA",
            "results", "single_measurement_stranded", "viz",
            "scatterplots", seq_type, "zero_dot_sequences"
        )
        os.makedirs(out_dir, exist_ok=True)

        csv_path_gt = os.path.join(
            out_dir,
            f"zero_dot_sequences_{group_label}_{seq_type}_ground_truth.csv"
        )
        with open(csv_path_gt, "w", newline="") as out_f_gt:
            writer_gt = csv.writer(out_f_gt)
            writer_gt.writerow([
                "position_in_aggregated_array",
                "alt_index", "ref_index",
                "sequence_alt", "sequence_ref",
                "ground_truth_ratio", "predicted_score"
            ])
            for local_i, full_i in enumerate(zero_gt_full_indices):
                alt_idx, ref_idx = csv_indices_dict[seq_type][full_i]
                seq_alt = filtered_tagged_sequences[alt_idx]
                seq_ref = filtered_tagged_sequences[ref_idx]
                gt_val = aggregated_gt[zero_gt_indices[local_i]]
                pred_val = aggregated_pred[zero_gt_indices[local_i]]
                writer_gt.writerow([full_i, alt_idx, ref_idx, seq_alt, seq_ref, gt_val, pred_val])
        print(f"Wrote {len(zero_gt_full_indices)} 'zero‐dot' ground-truth sequences to {csv_path_gt}")
    else:
        print(f"No points with |ground_truth_ratio| ≤ ZERO_THRESHOLD in this group.")

    # ----------------------------------------------------------------------------
    # 4.5) Remove any points where either gt or pred is exactly zero
    # ----------------------------------------------------------------------------
    non_zero_mask = (aggregated_gt != 0) & (aggregated_pred != 0)
    removed_count = len(aggregated_gt) - np.count_nonzero(non_zero_mask)
    if removed_count > 0:
        print(f"Removed {removed_count} points where either ground-truth or predicted score was exactly zero.")
    aggregated_gt = aggregated_gt[non_zero_mask]
    aggregated_pred = aggregated_pred[non_zero_mask]

    # ----------------------------------------------------------------------------
    # 5) Now proceed with the usual trimming‐and‐plotting logic
    # ----------------------------------------------------------------------------
    if len(aggregated_gt) < 2:
        print("Not enough data points after removing zeros for correlation.")
        return

    if np.std(aggregated_gt) != 0 and np.std(aggregated_pred) != 0:
        try:
            pearson_trim, _ = pearsonr(aggregated_gt, aggregated_pred)
            spearman_trim, _ = spearmanr(aggregated_gt, aggregated_pred)
        except Exception:
            pearson_trim, spearman_trim = np.nan, np.nan
    else:
        pearson_trim, spearman_trim = 0.0, 0.0

    title = f"{SEQ_NAME_MAPPING[seq_type]}:\nAggregated across {group_label} genes"
    out_dir = os.path.join(
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA",
        "results", "single_measurement_stranded", "viz",
        "scatterplots", seq_type, "aggregated"
    )
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"scatter_aggregated_{group_label}_{seq_type}.png")
    plot_scatter(aggregated_gt, aggregated_pred, title, filename, pearson_trim, spearman_trim)
    print(f"Saved overall aggregated scatter plot for {group_label} genes to {filename}")

    visualize_group_by_distance(gene_list, ground_truth, seq_type, group_label)
    plot_trend_correlation_group(gene_list, ground_truth, seq_type, group_label)

##############################################
# 7. Main Processing and Analysis
##############################################
def main():
    aggregated_data = {'+': [], '-': []}
    seq_types = ["motif_tiling_seqs", "motif_perturbation", "all_SNVs_seqs"]
    
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
    
    for seq_type in seq_types:
        for strand, mapping in [('+', pos_gene_mapping), ('-', neg_gene_mapping)]:
            for gene_name, target_gene in mapping.items():
                print(f"\nProcessing {seq_type} gene {target_gene} (symbol: {gene_name}) on strand {strand}...")
                input_dir = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/all_seq_types/{seq_type}/{gene_name}_{target_gene}_{'pos' if strand=='+' else 'neg'}_outputs"
                output_dir = os.path.join(input_dir, "plots")
                gene_stats = process_gene_plots(target_gene, gene_name, seq_type, input_dir=input_dir, output_dir=output_dir)
                if gene_stats is not None:
                    gene_stats['strand'] = strand
                    aggregated_data[strand].append(gene_stats)

    with open("aggregated_data.pkl", "wb") as f:
        pickle.dump(aggregated_data, f)
    print("Aggregated data saved to aggregated_data.pkl")
        
    for seq_type in seq_types:
        for group_label in ["all"]:
            analyze_and_plot_group(aggregated_data, GROUND_TRUTH_EXP, seq_type, group_label)

if __name__ == "__main__":
    main()
