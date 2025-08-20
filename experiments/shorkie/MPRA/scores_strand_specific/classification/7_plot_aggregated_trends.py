#!/usr/bin/env python
"""
Classification analysis for distinguishing high_exp_seqs vs low_exp_seqs
using the (signed) logSED score as classifier.

This script performs the analysis separately for:
  - 10 positive–strand genes
  - 12 negative–strand genes
  - All genes (combined)

For each gene and for each insertion context, the script:
  - Loads NPZ files from the proper directory (depending on strand).
  - Computes the per–sequence classifier score (averaging over targets).
  - Labels sequences from high_exp_seqs as 1 and from low_exp_seqs as 0.
  - Computes ROC/PR curves, AUC, average precision and a simple accuracy (using 0 as threshold).
  - Saves per–insertion plots (ROC and PR) and a text file with metrics.

Then, aggregated analysis is performed for:
   • Positive genes (10 genes)
   • Negative genes (12 genes)
   • All genes (combined)
with trend plots for AUROC and AUPRC across insertion positions.
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ============================================================================
# --- Utility Functions ---
# ----------------------------------------------------------------------------
def get_insertion_position(fname):
    """
    Extract the insertion position from the filename.
    Mapping: insertion_position = 60 + (index * 5)
    Expected filename format:
      <target_gene>_context_<index>_<...>.npz  
    """
    base = os.path.basename(fname)
    m = re.search(r'_context_(\d+)_', base)
    if m:
        index = int(m.group(1))
        return 100 + index * 10
    return 0

# ----------------------------------------------------------------------------
def load_npz_files(seq_type, gene_name, target_gene, strand, base_results_dir="results"):
    """
    For a given gene and seq_type, returns a dictionary mapping each
    insertion position (nt upstream) to its NPZ file path.
    
    The NPZ files are expected in a directory that depends on the strand.
    """
    if strand == '+':
        folder = os.path.join(base_results_dir, seq_type, f"{gene_name}_{target_gene}_pos_outputs")
    else:
        folder = os.path.join(base_results_dir, seq_type, f"{gene_name}_{target_gene}_neg_outputs")
    
    npz_files = glob.glob(os.path.join(folder, f"{target_gene}_context_*.npz"))
    # Sort files by insertion position.
    npz_files = sorted(npz_files, key=get_insertion_position)
    data_dict = {}
    for file in npz_files:
        ins_pos = get_insertion_position(file)
        data_dict[ins_pos] = file
    return data_dict

# ----------------------------------------------------------------------------
def load_scores_from_npz(npz_file):
    """
    Load the NPZ file and compute the per-sequence score as the average
    over targets (logSED).
    """
    data = np.load(npz_file)
    logSED = data["logSED"]  # shape: (num_entries, num_targets)
    scores = np.mean(logSED, axis=1)
    return scores

# ----------------------------------------------------------------------------
def compute_classification_metrics(scores, labels):
    """
    Compute ROC and PR metrics from the given scores and binary labels.
    Returns a dictionary with fpr, tpr, roc_auc, precision, recall,
    average precision, and a simple accuracy (using threshold 0).
    """
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    
    predicted = (scores >= 0).astype(int)
    accuracy = np.mean(predicted == labels)
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "avg_precision": avg_precision,
        "accuracy": accuracy
    }

# ----------------------------------------------------------------------------
def plot_roc_curve(metrics, title, out_path):
    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"ROC (AUC = {metrics['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

# ----------------------------------------------------------------------------
def plot_pr_curve(metrics, title, out_path):
    plt.figure()
    plt.plot(metrics["recall"], metrics["precision"],
             label=f"PR (AP = {metrics['avg_precision']:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

# ============================================================================
# --- Per-Gene Analysis ---
# ----------------------------------------------------------------------------
def analyze_gene_classification(gene_name, target_gene, strand, seq_type="high_exp_seqs",
                                base_results_dir="results", output_dir="classification_results"):
    """
    For a single gene, load NPZ files from high_exp_seqs and low_exp_seqs,
    compute classification metrics for each common insertion position,
    and save plots and metrics files.
    """
    # Create output directory for the gene.
    gene_output_dir = os.path.join(output_dir, gene_name)
    os.makedirs(gene_output_dir, exist_ok=True)
    
    # Load NPZ files for both groups.
    high_data = load_npz_files(seq_type, gene_name, target_gene, strand, base_results_dir)
    low_data = load_npz_files(seq_type, gene_name, target_gene, strand, base_results_dir.replace("high_exp_seqs", "low_exp_seqs"))
    
    # Find common insertion positions.
    common_ins = sorted(set(high_data.keys()).intersection(set(low_data.keys())))
    gene_results = {"insertion_positions": common_ins, "metrics": {}}
    
    for ins in common_ins:
        high_file = high_data[ins]
        low_file = low_data[ins]
        high_scores = load_scores_from_npz(high_file)
        low_scores = load_scores_from_npz(low_file)
        
        # Concatenate scores and labels (1 for high, 0 for low).
        scores = np.concatenate([high_scores, low_scores])
        labels = np.concatenate([np.ones_like(high_scores), np.zeros_like(low_scores)])
        
        metrics = compute_classification_metrics(scores, labels)
        gene_results["metrics"][ins] = metrics
        
        # Save ROC and PR plots.
        roc_title = f"{gene_name} ({target_gene}, {strand}) - Insertion {ins} nt: ROC Curve"
        pr_title  = f"{gene_name} ({target_gene}, {strand}) - Insertion {ins} nt: PR Curve"
        roc_out_path = os.path.join(gene_output_dir, f"{target_gene}_ins_{ins}_ROC.png")
        pr_out_path  = os.path.join(gene_output_dir, f"{target_gene}_ins_{ins}_PR.png")
        plot_roc_curve(metrics, roc_title, roc_out_path)
        plot_pr_curve(metrics, pr_title, pr_out_path)
        
        # Save a text file with metrics.
        metrics_out_path = os.path.join(gene_output_dir, f"{target_gene}_ins_{ins}_metrics.txt")
        with open(metrics_out_path, "w") as f:
            f.write(f"Gene: {gene_name} ({target_gene}, {strand}) - Insertion {ins} nt\n")
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"Average Precision: {metrics['avg_precision']:.4f}\n")
            f.write(f"Accuracy (threshold=0): {metrics['accuracy']:.4f}\n")
        print(f"Processed {gene_name} insertion {ins} nt. Metrics saved to {metrics_out_path}.")
    
    return gene_results

# ============================================================================
# --- Aggregated Analysis Functions ---
# ----------------------------------------------------------------------------
def aggregate_classification_data(gene_list, gene_dict, group_name, strand, seq_type="high_exp_seqs",
                                  base_results_dir="results", agg_output_dir="classification_results/aggregated"):
    """
    Aggregate data for a given list of genes (all on the same strand).
    gene_dict maps gene_name -> target_gene.
    Saves ROC/PR plots for each common insertion position.
    """
    os.makedirs(os.path.join(agg_output_dir, group_name), exist_ok=True)
    
    common_ins = None
    # First determine the set of insertion positions common to all genes.
    for gene in gene_list:
        target_gene = gene_dict[gene]
        high_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir)
        low_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir.replace("high_exp_seqs", "low_exp_seqs"))
        gene_ins = set(high_files.keys()).intersection(set(low_files.keys()))
        if common_ins is None:
            common_ins = gene_ins
        else:
            common_ins = common_ins.intersection(gene_ins)
    if common_ins is None or len(common_ins) == 0:
        print(f"No common insertion positions for group {group_name}.")
        return None
    common_ins = sorted(common_ins)
    
    agg_results = {"insertion_positions": common_ins, "metrics": {}}
    
    # For each common insertion, aggregate scores/labels across genes.
    for ins in common_ins:
        all_scores = []
        all_labels = []
        for gene in gene_list:
            target_gene = gene_dict[gene]
            high_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir)
            low_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir.replace("high_exp_seqs", "low_exp_seqs"))
            if ins in high_files and ins in low_files:
                high_scores = load_scores_from_npz(high_files[ins])
                low_scores = load_scores_from_npz(low_files[ins])
                all_scores.extend(high_scores.tolist())
                all_labels.extend(np.ones_like(high_scores).tolist())
                all_scores.extend(low_scores.tolist())
                all_labels.extend(np.zeros_like(low_scores).tolist())
        if len(all_scores) > 0:
            metrics = compute_classification_metrics(np.array(all_scores), np.array(all_labels))
            agg_results["metrics"][ins] = metrics
            group_out_dir = os.path.join(agg_output_dir, group_name)
            roc_title = f"Group {group_name} ({strand}) - Insertion {ins} nt: ROC Curve"
            pr_title  = f"Group {group_name} ({strand}) - Insertion {ins} nt: PR Curve"
            roc_out_path = os.path.join(group_out_dir, f"group_{group_name}_ins_{ins}_ROC.png")
            pr_out_path  = os.path.join(group_out_dir, f"group_{group_name}_ins_{ins}_PR.png")
            plot_roc_curve(metrics, roc_title, roc_out_path)
            plot_pr_curve(metrics, pr_title, pr_out_path)
            metrics_out_path = os.path.join(group_out_dir, f"group_{group_name}_ins_{ins}_metrics.txt")
            with open(metrics_out_path, "w") as f:
                f.write(f"Group {group_name} ({strand}) - Insertion {ins} nt\n")
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"Average Precision: {metrics['avg_precision']:.4f}\n")
                f.write(f"Accuracy (threshold=0): {metrics['accuracy']:.4f}\n")
            print(f"Aggregated group {group_name} insertion {ins} nt processed. Metrics saved to {metrics_out_path}.")
    return agg_results

# ----------------------------------------------------------------------------
def aggregate_all_genes(gene_list_info, seq_type="high_exp_seqs",
                        base_results_dir="results", agg_output_dir="classification_results/aggregated"):
    """
    Aggregate classification data for all genes (mixed strands).
    gene_list_info is a list of tuples: (gene_name, target_gene, strand)
    """
    os.makedirs(os.path.join(agg_output_dir, "all_genes"), exist_ok=True)
    
    common_ins = None
    for gene, target_gene, strand in gene_list_info:
        high_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir)
        low_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir.replace("high_exp_seqs", "low_exp_seqs"))
        gene_ins = set(high_files.keys()).intersection(set(low_files.keys()))
        if common_ins is None:
            common_ins = gene_ins
        else:
            common_ins = common_ins.intersection(gene_ins)
    if common_ins is None or len(common_ins) == 0:
        print("No common insertion positions across all genes.")
        return None
    common_ins = sorted(common_ins)
    
    agg_results = {"insertion_positions": common_ins, "metrics": {}}
    
    for ins in common_ins:
        all_scores = []
        all_labels = []
        for gene, target_gene, strand in gene_list_info:
            high_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir)
            low_files = load_npz_files(seq_type, gene, target_gene, strand, base_results_dir.replace("high_exp_seqs", "low_exp_seqs"))
            if ins in high_files and ins in low_files:
                high_scores = load_scores_from_npz(high_files[ins])
                low_scores = load_scores_from_npz(low_files[ins])
                all_scores.extend(high_scores.tolist())
                all_labels.extend(np.ones_like(high_scores).tolist())
                all_scores.extend(low_scores.tolist())
                all_labels.extend(np.zeros_like(low_scores).tolist())
        if len(all_scores) > 0:
            metrics = compute_classification_metrics(np.array(all_scores), np.array(all_labels))
            agg_results["metrics"][ins] = metrics
            group_out_dir = os.path.join(agg_output_dir, "all_genes")
            roc_title = f"All Genes - Insertion {ins} nt: ROC Curve"
            pr_title  = f"All Genes - Insertion {ins} nt: PR Curve"
            roc_out_path = os.path.join(group_out_dir, f"all_genes_ins_{ins}_ROC.png")
            pr_out_path  = os.path.join(group_out_dir, f"all_genes_ins_{ins}_PR.png")
            plot_roc_curve(metrics, roc_title, roc_out_path)
            plot_pr_curve(metrics, pr_title, pr_out_path)
            metrics_out_path = os.path.join(group_out_dir, f"all_genes_ins_{ins}_metrics.txt")
            with open(metrics_out_path, "w") as f:
                f.write(f"All Genes - Insertion {ins} nt\n")
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"Average Precision: {metrics['avg_precision']:.4f}\n")
                f.write(f"Accuracy (threshold=0): {metrics['accuracy']:.4f}\n")
            print(f"Aggregated all genes insertion {ins} nt processed. Metrics saved to {metrics_out_path}.")
    return agg_results

# ============================================================================
# --- Plotting Aggregated Trends ---
# ----------------------------------------------------------------------------
def plot_aggregated_trends(agg_results_dict, output_dir):
    """
    Plot the trend across insertion positions for AUROC and AUPRC
    for each aggregated experiment in agg_results_dict.
    
    agg_results_dict is a dict mapping group names (e.g. "positive", "negative", "all")
    to an aggregated results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # AUROC trend plot.
    plt.figure(figsize=(10, 6))
    for group_name, agg_results in agg_results_dict.items():
        insertion_positions = agg_results["insertion_positions"]
        aurocs = [agg_results["metrics"][ins]["roc_auc"] for ins in insertion_positions]
        plt.plot(insertion_positions, aurocs, marker='o', label=group_name)
    plt.xlabel("Insertion Position (nt upstream)")
    plt.ylabel("AUROC")
    plt.title("Trend of AUROC across insertion positions")
    plt.legend()
    plt.grid(True)
    auroc_plot_path = os.path.join(output_dir, "aggregated_AUROC_trend.png")
    plt.savefig(auroc_plot_path)
    plt.close()
    print(f"Saved AUROC trend plot to {auroc_plot_path}")
    
    # AUPRC trend plot.
    plt.figure(figsize=(10, 6))
    for group_name, agg_results in agg_results_dict.items():
        insertion_positions = agg_results["insertion_positions"]
        auprcs = [agg_results["metrics"][ins]["avg_precision"] for ins in insertion_positions]
        plt.plot(insertion_positions, auprcs, marker='o', label=group_name)
    plt.xlabel("Insertion Position (nt upstream)")
    plt.ylabel("AUPRC")
    plt.title("Trend of AUPRC across insertion positions")
    plt.legend()
    plt.grid(True)
    auprc_plot_path = os.path.join(output_dir, "aggregated_AUPRC_trend.png")
    plt.savefig(auprc_plot_path)
    plt.close()
    print(f"Saved AUPRC trend plot to {auprc_plot_path}")

# ============================================================================
# --- Main Routine ---
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------
    # Define gene lists and mappings.
    # Positive–strand genes (10 genes)

    seq_types = ["challenging_seqs", "all_random_seqs", "yeast_seqs", "high_exp_seqs", "low_exp_seqs"]
    pos_genes = ["GPM3", "SLI1", "VPS52", "YMR160W", "MRPS28", "YCT1", "RDL2", "PHS1", "RTC3", "MSN4"]
    pos_gene_name_2_id = {
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
    
    # Negative–strand genes (12 genes)
    neg_genes = ["COA4", "ERI1", "RSM25", "ERD1", "MRM2", "SNT2", "CSI2", "RPE1", "PKC1", "AIM11", "MAE1", "MRPL1"]
    neg_gene_name_2_id = {
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
    
    # Base directory for NPZ files.
    base_results_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/all_seq_types"
    # Output directory for classification results.
    output_dir = f"{base_results_dir}/classification_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # # ---------------------------
    # # (A) Per–gene analysis (optional) for each gene.
    # # You can run per–gene analysis if you wish; here we loop over each gene.
    # for gene in pos_genes:
    #     target_gene = pos_gene_name_2_id[gene]
    #     print(f"Processing positive gene: {gene} ({target_gene}) ...")
    #     analyze_gene_classification(gene, target_gene, strand='+', seq_type="high_exp_seqs",
    #                                 base_results_dir=base_results_dir, output_dir=output_dir)
    
    # for gene in neg_genes:
    #     target_gene = neg_gene_name_2_id[gene]
    #     print(f"Processing negative gene: {gene} ({target_gene}) ...")
    #     analyze_gene_classification(gene, target_gene, strand='-', seq_type="high_exp_seqs",
    #                                 base_results_dir=base_results_dir, output_dir=output_dir)
    
    # ---------------------------
    # (B) Aggregated analysis for positive genes.
    agg_pos = aggregate_classification_data(pos_genes, pos_gene_name_2_id, "positive", strand='+',
                                            seq_type="high_exp_seqs", base_results_dir=base_results_dir)
    
    # Aggregated analysis for negative genes.
    agg_neg = aggregate_classification_data(neg_genes, neg_gene_name_2_id, "negative", strand='-',
                                            seq_type="high_exp_seqs", base_results_dir=base_results_dir)
    
    # ---------------------------
    # (C) Aggregated analysis for ALL genes.
    # Build a list of tuples: (gene, target_gene, strand) for both groups.
    all_genes_info = []
    for gene in pos_genes:
        all_genes_info.append((gene, pos_gene_name_2_id[gene], '+'))
    for gene in neg_genes:
        all_genes_info.append((gene, neg_gene_name_2_id[gene], '-'))
    
    agg_all = aggregate_all_genes(all_genes_info, seq_type="high_exp_seqs", base_results_dir=base_results_dir)
    
    # ---------------------------
    # (D) Plot aggregated AUROC and AUPRC trends.
    aggregated_results = {
        "positive": agg_pos,
        "negative": agg_neg,
        "all": agg_all
    }
    trend_output_dir = os.path.join(output_dir, "aggregated")
    plot_aggregated_trends(aggregated_results, trend_output_dir)
    
    print("Classification analysis complete. All figures and metrics files have been saved.")
