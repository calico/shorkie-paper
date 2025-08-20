#!/usr/bin/env python
import os
import math
import glob
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib.lines import Line2D

#############################
# Helper Functions
#############################

def get_insertion_position(fname):
    """
    Extract insertion position from filename.
    Mapping: insertion_position = 100 + (index * 10)
    """
    base = os.path.basename(fname)
    m = re.search(r'_context_(\d+)_', base)
    if m:
        index = int(m.group(1))
        return 100 + index * 10
    return 0

def compute_classifier_metrics_for_gene(target_gene, gene_name, strand, high_dir, low_dir):
    """
    For a given gene, load NPZ files from high and low directories,
    compute per-sequence scores (average logSED) and then compute ROC/PR curves
    (with AUROC and AUPRC) for each insertion position.
    
    Returns a dictionary keyed by insertion position.
    """
    high_files = sorted(glob.glob(os.path.join(high_dir, f"{target_gene}_context_*.npz")),
                        key=get_insertion_position)
    low_files  = sorted(glob.glob(os.path.join(low_dir, f"{target_gene}_context_*.npz")),
                        key=get_insertion_position)
    if not high_files or not low_files:
        print(f"WARNING: Missing files for gene {gene_name} in one or both directories.")
        return None
    metrics = {}
    for hf, lf in zip(high_files, low_files):
        pos_val = get_insertion_position(hf)
        # Load high expression data
        data_high = np.load(hf)
        scores_high = np.mean(data_high["logSED"], axis=1)
        # Load low expression data
        data_low = np.load(lf)
        scores_low = np.mean(data_low["logSED"], axis=1)
        # Combine scores and labels (1 for high, 0 for low)
        scores = np.concatenate([scores_high, scores_low])
        labels = np.concatenate([np.ones(len(scores_high)), np.zeros(len(scores_low))])
        # Compute ROC and AUROC
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        # Compute Precision-Recall and AUPRC
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = average_precision_score(labels, scores)
        metrics[pos_val] = {
            'auroc': roc_auc,
            'auprc': auprc,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }
    return metrics

def plot_gene_roc_pr(gene_metrics, gene_symbol, target_gene, strand, output_dir):
    """
    For one gene (with multiple insertion positions), plot ROC and PR curves.
    The left subplot shows ROC curves and the right subplot shows PR curves,
    with one curve per insertion position.
    """
    positions = sorted(gene_metrics.keys())
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for pos in positions:
        fpr, tpr = gene_metrics[pos]['roc_curve']
        precision, recall = gene_metrics[pos]['pr_curve']
        axs[0].plot(fpr, tpr, label=f"{pos} nt", alpha=0.3)
        axs[1].plot(recall, precision, label=f"{pos} nt", alpha=0.3)
    axs[0].set_title(f"ROC Curves for {gene_symbol} ({target_gene}) - {strand}")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_title(f"PR Curves for {gene_symbol} ({target_gene}) - {strand}")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    out_file = os.path.join(output_dir, f"{gene_symbol}_{target_gene}_{strand}_roc_pr.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved ROC/PR curves for {gene_symbol} to {out_file}")

def plot_trend_with_individuals(metrics_list, metric, title, output_file):
    """
    Given a list of gene metric dictionaries (each of the form:
    { "metrics": {insertion_pos: {metric: value, ...}, ... },
      "gene_symbol": <gene_symbol>,
      "strand": <'pos' or 'neg'> }
    plot a trend (AUROC or AUPRC) vs. insertion position.
    Each gene’s individual trend is plotted with a light dashed line and the legend
    shows the gene name and strand. The aggregated trend (mean ± std) is overlaid in bold.
    The legend is placed on the right side of the figure.
    """
    positions = sorted(metrics_list[0]["metrics"].keys())
    fig, ax = plt.subplots(figsize=(10, 3))
    for gene in metrics_list:
        gene_values = [gene["metrics"][pos][metric] for pos in positions]
        label = f"{gene['gene_symbol']} ({gene['strand']})"
        ax.plot(positions, gene_values, marker='o', linestyle='--', alpha=0.3, label=label)
    # Aggregated trend:
    agg_means = []
    agg_stds = []
    for pos in positions:
        values = [gene["metrics"][pos][metric] for gene in metrics_list if pos in gene["metrics"]]
        agg_means.append(np.mean(values))
        agg_stds.append(np.std(values))
    ax.errorbar(positions, agg_means, yerr=agg_stds, fmt='o-', color='black', markersize=8,
                capsize=5, label='Aggregate')
    ax.set_xlabel("Insertion Position (nt upstream)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True)
    fig.subplots_adjust(right=0.75)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved trend plot ({metric.upper()}) to {output_file}")

def plot_merged_trend_subplots(combined_quantiles, metric, title, output_file):
    """
    (Deprecated) Original function that created separate subplots for each quantile.
    """
    quant_order = ["5-25", "25-75", "75-95"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharey=True)
    
    for i, quant in enumerate(quant_order):
        ax = axs[i]
        group_list = combined_quantiles.get(quant, [])
        if not group_list:
            ax.text(0.5, 0.5, f"No data for {quant}", ha="center", va="center")
            ax.set_title(f"{quant} Quantile")
            continue
        positions = sorted(group_list[0]["metrics"].keys())
        for gene in group_list:
            gene_values = [gene["metrics"][pos][metric] for pos in positions]
            gene_label = f"{gene['gene_symbol']} ({gene['strand']})"
            ax.plot(positions, gene_values, marker='o', linestyle='--', alpha=0.3, label=gene_label)
        agg_means = []
        agg_stds = []
        for pos in positions:
            values = [gene["metrics"][pos][metric] for gene in group_list if pos in gene["metrics"]]
            agg_means.append(np.mean(values))
            agg_stds.append(np.std(values))
        ax.errorbar(positions, agg_means, yerr=agg_stds, fmt='o-', color='black', markersize=8,
                    capsize=5, label='Aggregate')
        ax.set_xlabel("Insertion Position (nt upstream)")
        if i == 0:
            ax.set_ylabel(metric.upper())
        ax.set_title(f"{quant} Quantile")
        ax.grid(True)
        ax.legend(loc='lower right', fontsize=7.5)
    
    fig.suptitle(title, y=0.89, fontsize=18)
    fig.tight_layout(rect=[0, 0, 0.95, 0.90])
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved merged trend subplots ({metric.upper()}) to {output_file}")

def plot_combined_trend_quantiles(combined_quantiles, metric, title, output_file):
    """
    For combined genes, create one plot with curves for all three quantile groups overlaid
    on the same axis. Every individual gene is assigned a unique color (across all quantiles)
    from a global colormap, and the aggregated curve for each quantile is drawn using a
    predefined dark color.
    
    Aggregated colors:
      - "5-25": dark blue (#00008B)
      - "25-75": dark green (#006400)
      - "75-95": dark red (#8B0000)
    """
    fig, ax = plt.subplots(figsize=(8, 3.7))
    
    # Build a global mapping for unique gene colors.
    gene_labels_set = []
    for quant, group_list in combined_quantiles.items():
        for gene in group_list:
            label = f"{gene['gene_symbol']} ({gene['strand']})"
            if label not in gene_labels_set:
                gene_labels_set.append(label)
    total_genes = len(gene_labels_set)
    cmap_genes = plt.cm.get_cmap("tab20", total_genes)
    gene_color_mapping = {label: cmap_genes(i) for i, label in enumerate(gene_labels_set)}
    
    # Predefined aggregated colors for each quantile group.
    agg_colors = {
        "5-25": "#006400",   # Dark Green
        "25-75": "#8B0000",  # Dark Red #8B0000
        "75-95": "#000000"   # Dark Black
    }
    
    # For legend entries.
    legend_lines = []
    legend_labels = []
    
    # Loop through each quantile group.
    for quant, group_list in combined_quantiles.items():
        if not group_list:
            continue
        positions = sorted(group_list[0]["metrics"].keys())
        # Plot each gene using its unique color.
        for gene in group_list:
            label = f"{gene['gene_symbol']} ({gene['strand']})"
            gene_values = [gene["metrics"][pos][metric] for pos in positions]
            line, = ax.plot(positions, gene_values, linestyle='--', marker='o', alpha=0.5,
                            color=gene_color_mapping[label], label=label)
            if label not in legend_labels:
                legend_lines.append(line)
                legend_labels.append(label)
        # Compute aggregated trend for the current quantile.
        agg_means = []
        agg_stds = []
        for pos in positions:
            values = [gene["metrics"][pos][metric] for gene in group_list if pos in gene["metrics"]]
            agg_means.append(np.mean(values))
            agg_stds.append(np.std(values))
        # Plot the aggregated curve with error bars in the designated dark color.
        agg_container = ax.errorbar(positions, agg_means, yerr=agg_stds, fmt='o-', color=agg_colors.get(quant, 'black'),
                                    markersize=8, capsize=5, label=f"{quant} Aggregate")
        # Extract the main line from the errorbar container.
        agg_line = agg_container.lines[0]
        legend_lines.append(agg_line)
        legend_labels.append(f"{quant} Aggregate")
    
    ax.set_xlabel("Insertion Position (nt upstream)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True)
    ax.legend(legend_lines, legend_labels, loc='best', fontsize=8, ncol=3)    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved combined quantile trend plot ({metric.upper()}) to {output_file}")

def average_roc_curve(metrics_list, insertion_position, grid_fpr):
    """
    Average ROC curves for a given insertion position across genes.
    Interpolates each gene's TPR at the common FPR grid.
    """
    tprs = []
    for gene in metrics_list:
        if insertion_position in gene["metrics"]:
            fpr, tpr = gene["metrics"][insertion_position]["roc_curve"]
            interp_tpr = np.interp(grid_fpr, fpr, tpr)
            tprs.append(interp_tpr)
    if not tprs:
        return None, None
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    return mean_tpr, std_tpr

def average_pr_curve(metrics_list, insertion_position, grid_recall):
    """
    Average PR curves for a given insertion position across genes.
    Interpolates each gene's Precision at the common Recall grid.
    """
    precisions = []
    for gene in metrics_list:
        if insertion_position in gene["metrics"]:
            precision, recall = gene["metrics"][insertion_position]["pr_curve"]
            recall = recall[::-1]
            precision = precision[::-1]
            interp_precision = np.interp(grid_recall, recall, precision)
            precisions.append(interp_precision)
    if not precisions:
        return None, None
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    return mean_precision, std_precision

def plot_group_roc_pr(metrics_list, group_label, output_dir):
    """
    Given a list of gene metric dictionaries for a group (e.g. all positive genes),
    this function creates multi-panel figures for ROC and PR curves for each insertion position.
    A single shared legend (with gene names and "Aggregate") is placed outside the subplots.
    """
    insertion_positions = sorted(metrics_list[0]["metrics"].keys())
    
    grid_fpr = np.linspace(0, 1, 100)
    grid_recall = np.linspace(0, 1, 100)
    
    n_panels = len(insertion_positions)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    
    cmap = plt.cm.get_cmap("tab10", len(metrics_list))
    
    ############################################################################
    # ROC PANELS
    ############################################################################
    fig_roc, axs_roc = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows * ncols > 1:
        axs_roc = axs_roc.flatten()
    else:
        axs_roc = [axs_roc]
    
    gene_handles = []
    gene_labels = []
    
    for i, pos in enumerate(insertion_positions):
        ax = axs_roc[i]
        for j, gene_data in enumerate(metrics_list):
            if pos in gene_data["metrics"]:
                fpr, tpr = gene_data["metrics"][pos]["roc_curve"]
                color = cmap(j)
                line, = ax.plot(fpr, tpr, color=color, alpha=0.8)
                if i == 0:
                    gene_labels.append(f"{gene_data['gene_symbol']} ({gene_data['strand']})")
                    gene_handles.append(line)
        mean_tpr, std_tpr = average_roc_curve(metrics_list, pos, grid_fpr)
        if mean_tpr is not None:
            ax.plot(grid_fpr, mean_tpr, color='black', lw=2)
            ax.fill_between(grid_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='black', alpha=0.2)
            avg_roc_auc = auc(grid_fpr, mean_tpr)
            ax.text(0.05, 0.1, f"AUROC = {avg_roc_auc:.2f}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{pos} nt")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True)
    
    for j in range(i+1, len(axs_roc)):
        fig_roc.delaxes(axs_roc[j])
    
    aggregate_handle = Line2D([0], [0], color='black', lw=2)
    handles = gene_handles + [aggregate_handle]
    labels = gene_labels + ['Aggregate']
    fig_roc.legend(handles, labels, loc='center left', bbox_to_anchor=(0.93, 0.5))
    
    fig_roc.suptitle(f"{group_label} Genes - ROC Curves", fontsize=16)
    fig_roc.tight_layout(pad=2, h_pad=3, w_pad=3, rect=[0, 0, 0.88, 0.95])
    roc_output_file = os.path.join(output_dir, f"{group_label}_roc_group.png")
    fig_roc.savefig(roc_output_file, bbox_inches='tight', dpi=300)
    plt.close(fig_roc)
    print(f"Saved group ROC plot to {roc_output_file}")
    
    ############################################################################
    # PR PANELS
    ############################################################################
    fig_pr, axs_pr = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows * ncols > 1:
        axs_pr = axs_pr.flatten()
    else:
        axs_pr = [axs_pr]
    
    gene_handles = []
    gene_labels = []
    
    for i, pos in enumerate(insertion_positions):
        ax = axs_pr[i]
        for j, gene_data in enumerate(metrics_list):
            if pos in gene_data["metrics"]:
                precision, recall = gene_data["metrics"][pos]["pr_curve"]
                recall = recall[::-1]
                precision = precision[::-1]
                color = cmap(j)
                line, = ax.plot(recall, precision, color=color, alpha=0.8)
                if i == 0:
                    gene_labels.append(f"{gene_data['gene_symbol']} ({gene_data['strand']})")
                    gene_handles.append(line)
        mean_precision, std_precision = average_pr_curve(metrics_list, pos, grid_recall)
        if mean_precision is not None:
            ax.plot(grid_recall, mean_precision, color='green', lw=2)
            ax.fill_between(grid_recall, mean_precision - std_precision, mean_precision + std_precision,
                            color='green', alpha=0.2)
            avg_auprc = auc(grid_recall, mean_precision)
            ax.text(0.05, 0.1, f"AUPRC = {avg_auprc:.2f}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{pos} nt")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)
    
    for j in range(len(insertion_positions), len(axs_pr)):
        fig_pr.delaxes(axs_pr[j])
    
    aggregate_handle = Line2D([0], [0], color='green', lw=2)
    handles = gene_handles + [aggregate_handle]
    labels = gene_labels + ['Aggregate']
    fig_pr.legend(handles, labels, loc='center left', bbox_to_anchor=(0.93, 0.5))
    
    fig_pr.suptitle(f"{group_label} Genes - PR Curves", fontsize=16)
    fig_pr.tight_layout(pad=2, h_pad=4, w_pad=4, rect=[0, 0, 0.88, 0.95])
    pr_output_file = os.path.join(output_dir, f"{group_label}_pr_group.png")
    fig_pr.savefig(pr_output_file, bbox_inches='tight', dpi=300)
    plt.close(fig_pr)
    print(f"Saved group PR plot to {pr_output_file}")

def plot_group_roc_pr_by_tss(metrics_list, group_label, output_dir):
    """
    For each TSS distance (insertion position), create separate ROC and PR plots.
    Each plot is saved individually in the output directory.
    """
    insertion_positions = sorted(metrics_list[0]["metrics"].keys())
    grid_fpr = np.linspace(0, 1, 100)
    grid_recall = np.linspace(0, 1, 100)
    
    cmap = plt.cm.get_cmap("tab10", len(metrics_list))
    
    for pos in insertion_positions:
        # ROC plot for this TSS distance
        plt.figure(figsize=(6, 6))
        gene_handles = []
        gene_labels = []
        for j, gene_data in enumerate(metrics_list):
            if pos in gene_data["metrics"]:
                fpr, tpr = gene_data["metrics"][pos]["roc_curve"]
                color = cmap(j)
                plt.plot(fpr, tpr, color=color, alpha=0.3)
                gene_handles.append(Line2D([0], [0], color=color, lw=1, alpha=0.3))
                gene_labels.append(f"{gene_data['gene_symbol']} ({gene_data['strand']})")
        tprs = []
        for gene in metrics_list:
            if pos in gene["metrics"]:
                fpr, tpr = gene["metrics"][pos]["roc_curve"]
                interp_tpr = np.interp(grid_fpr, fpr, tpr)
                tprs.append(interp_tpr)
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            plt.plot(grid_fpr, mean_tpr, color='black', lw=2, label='Aggregate')
            plt.fill_between(grid_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='black', alpha=0.2)
            agg_auc = auc(grid_fpr, mean_tpr)
            plt.text(0.05, 0.1, f"AUROC = {agg_auc:.2f}", transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlim(-0.05, 1.1)
        plt.ylim(-0.05, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"{group_label} Genes - ROC at {pos} nt")
        plt.grid(True)
        aggregate_handle = Line2D([0], [0], color='black', lw=2)
        handles = gene_handles + [aggregate_handle]
        labels = gene_labels + ['Aggregate']
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        roc_out_file = os.path.join(output_dir, f"{group_label}_TSS_{pos}_roc.png")
        plt.savefig(roc_out_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved ROC plot for TSS distance {pos} nt to {roc_out_file}")
        
        # PR plot for this TSS distance
        plt.figure(figsize=(6, 6))
        gene_handles = []
        gene_labels = []
        for j, gene_data in enumerate(metrics_list):
            if pos in gene_data["metrics"]:
                precision, recall = gene_data["metrics"][pos]["pr_curve"]
                recall = recall[::-1]
                precision = precision[::-1]
                color = cmap(j)
                plt.plot(recall, precision, color=color, alpha=0.3)
                gene_handles.append(Line2D([0], [0], color=color, lw=1, alpha=0.3))
                gene_labels.append(f"{gene_data['gene_symbol']} ({gene_data['strand']})")
        precisions_list = []
        for gene in metrics_list:
            if pos in gene["metrics"]:
                precision, recall = gene["metrics"][pos]["pr_curve"]
                recall = recall[::-1]
                precision = precision[::-1]
                interp_precision = np.interp(grid_recall, recall, precision)
                precisions_list.append(interp_precision)
        if precisions_list:
            mean_precision = np.mean(precisions_list, axis=0)
            std_precision = np.std(precisions_list, axis=0)
            plt.plot(grid_recall, mean_precision, color='black', lw=2, label='Aggregate')
            plt.fill_between(grid_recall, mean_precision - std_precision, mean_precision + std_precision,
                             color='black', alpha=0.2)
            agg_auprc = auc(grid_recall, mean_precision)
            plt.text(0.05, 0.1, f"AUPRC = {agg_auprc:.2f}", transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlim(-0.05, 1.1)
        plt.ylim(-0.05, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{group_label} Genes - PR at {pos} nt")
        plt.grid(True)
        aggregate_handle = Line2D([0], [0], color='black', lw=2)
        handles = gene_handles + [aggregate_handle]
        labels = gene_labels + ['Aggregate']
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        pr_out_file = os.path.join(output_dir, f"{group_label}_TSS_{pos}_pr.png")
        plt.savefig(pr_out_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved PR plot for TSS distance {pos} nt to {pr_out_file}")

#############################
# Main Experiment
#############################

def main():
    # Use the stranded results directory
    output_res_base = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/all_seq_types"
    
    # --- Options for Loading/Storing Metrics ---
    load_metrics = True  # Set True to load from file instead of reprocessing
    pos_metrics_file = os.path.join(output_res_base, "pos_metrics.pkl")
    neg_metrics_file = os.path.join(output_res_base, "neg_metrics.pkl")
    
    # --- Gene Mappings and Quantile Definitions ---
    pos_gene_mapping = {
        "GPM3": "YOL056W", 
        "SLI1": "YGR212W", 
        "VPS52": "YDR484W", 
        "YMR160W": "YMR160W", 
        "MRPS28": "YDR337W", 
        "YCT1": "YLL055W", 
        "RDL2": "YOR286W", 
        "PHS1": "YJL097W", 
        "RTC3": "YHR087W"
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
        "PKC1": "YBL105C"
    }
    pos_quantiles = {
        "5-25": ["GPM3", "SLI1", "VPS52"],
        "25-75": ["YMR160W", "MRPS28", "YCT1"],
        "75-95": ["RDL2", "PHS1", "RTC3"]
    }
    neg_quantiles = {
        "5-25": ["COA4", "ERI1", "RSM25"],
        "25-75": ["ERD1", "MRM2", "SNT2"],
        "75-95": ["CSI2", "RPE1", "PKC1"]
    }
    
    # Define sequence types to compare:
    high_seq_type = "high_exp_seqs"
    low_seq_type  = "low_exp_seqs"
    
    # --- Process or Load Per-Gene Metrics ---
    pos_metrics = {}
    neg_metrics = {}
    if load_metrics and os.path.exists(pos_metrics_file) and os.path.exists(neg_metrics_file):
        print("Loading metrics from files...")
        with open(pos_metrics_file, "rb") as f:
            pos_metrics = pickle.load(f)
        with open(neg_metrics_file, "rb") as f:
            neg_metrics = pickle.load(f)
        print("Loaded pos_metrics and neg_metrics.")
        # Backward compatibility: wrap entries if needed.
        for gene_symbol, data in pos_metrics.items():
            if "metrics" not in data:
                pos_metrics[gene_symbol] = {"metrics": data, "gene_symbol": gene_symbol, "strand": "pos"}
        for gene_symbol, data in neg_metrics.items():
            if "metrics" not in data:
                neg_metrics[gene_symbol] = {"metrics": data, "gene_symbol": gene_symbol, "strand": "neg"}
    else:
        # Process positive strand genes
        for gene_symbol, target_gene in pos_gene_mapping.items():
            high_dir = os.path.join(output_res_base, high_seq_type, f"{gene_symbol}_{target_gene}_pos_outputs")
            low_dir  = os.path.join(output_res_base, low_seq_type, f"{gene_symbol}_{target_gene}_pos_outputs")
            if not (os.path.exists(high_dir) and os.path.exists(low_dir)):
                print(f"Skipping {gene_symbol} (pos): directory missing.")
                continue
            print(f"Processing {gene_symbol} (pos)...")
            mets = compute_classifier_metrics_for_gene(target_gene, gene_symbol, "pos", high_dir, low_dir)
            if mets is not None:
                pos_metrics[gene_symbol] = {"metrics": mets, "gene_symbol": gene_symbol, "strand": "pos"}
        # Process negative strand genes
        for gene_symbol, target_gene in neg_gene_mapping.items():
            high_dir = os.path.join(output_res_base, high_seq_type, f"{gene_symbol}_{target_gene}_neg_outputs")
            low_dir  = os.path.join(output_res_base, low_seq_type, f"{gene_symbol}_{target_gene}_neg_outputs")
            if not (os.path.exists(high_dir) and os.path.exists(low_dir)):
                print(f"Skipping {gene_symbol} (neg): directory missing.")
                continue
            print(f"Processing {gene_symbol} (neg)...")
            mets = compute_classifier_metrics_for_gene(target_gene, gene_symbol, "neg", high_dir, low_dir)
            if mets is not None:
                neg_metrics[gene_symbol] = {"metrics": mets, "gene_symbol": gene_symbol, "strand": "neg"}
        # Save metrics for future use
        with open(pos_metrics_file, "wb") as f:
            pickle.dump(pos_metrics, f)
        with open(neg_metrics_file, "wb") as f:
            pickle.dump(neg_metrics, f)
        print("Saved metrics to files.")
    
    # --- Create Output Directories for Plots ---
    roc_pr_dir = os.path.join(output_res_base, "roc_pr_plots")
    trend_dir = os.path.join(output_res_base, "trend_plots")
    group_dir = os.path.join(output_res_base, "group_roc_pr_plots")
    os.makedirs(roc_pr_dir, exist_ok=True)
    os.makedirs(trend_dir, exist_ok=True)
    os.makedirs(group_dir, exist_ok=True)
    
    # --- Aggregated Trend Plots (Individual + Overall) ---
    if pos_metrics:
        pos_list = list(pos_metrics.values())
        plot_trend_with_individuals(pos_list, "auroc", "Positive Strand - AUROC Trend", os.path.join(trend_dir, "pos_auroc_trend.png"))
        plot_trend_with_individuals(pos_list, "auprc", "Positive Strand - AUPRC Trend", os.path.join(trend_dir, "pos_auprc_trend.png"))
    if neg_metrics:
        neg_list = list(neg_metrics.values())
        plot_trend_with_individuals(neg_list, "auroc", "Negative Strand - AUROC Trend", os.path.join(trend_dir, "neg_auroc_trend.png"))
        plot_trend_with_individuals(neg_list, "auprc", "Negative Strand - AUPRC Trend", os.path.join(trend_dir, "neg_auprc_trend.png"))
    if pos_metrics and neg_metrics:
        combined_list = list(pos_metrics.values()) + list(neg_metrics.values())
        plot_trend_with_individuals(combined_list, "auroc", "Combined Strands - AUROC Trend", os.path.join(trend_dir, "combined_auroc_trend.png"))
        plot_trend_with_individuals(combined_list, "auprc", "Combined Strands - AUPRC Trend", os.path.join(trend_dir, "combined_auprc_trend.png"))
    
    # --- Trend Plots Split by Quantiles for Positive and Negative Genes ---
    for quant, genes in pos_quantiles.items():
        group_list = [pos_metrics[g] for g in genes if g in pos_metrics]
        if group_list:
            plot_trend_with_individuals(group_list, "auroc", f"Positive Strand {quant} Quantile - AUROC Trend", os.path.join(trend_dir, f"pos_{quant}_auroc_trend.png"))
            plot_trend_with_individuals(group_list, "auprc", f"Positive Strand {quant} Quantile - AUPRC Trend", os.path.join(trend_dir, f"pos_{quant}_auprc_trend.png"))
    for quant, genes in neg_quantiles.items():
        group_list = [neg_metrics[g] for g in genes if g in neg_metrics]
        if group_list:
            plot_trend_with_individuals(group_list, "auroc", f"Negative Strand {quant} Quantile - AUROC Trend", os.path.join(trend_dir, f"neg_{quant}_auroc_trend.png"))
            plot_trend_with_individuals(group_list, "auprc", f"Negative Strand {quant} Quantile - AUPRC Trend", os.path.join(trend_dir, f"neg_{quant}_auprc_trend.png"))
    
    # --- Merged Trend Plots for Combined Genes (All Quantiles in One Plot) ---
    combined_quantiles = {}
    for quant in ["5-25", "25-75", "75-95"]:
        combined_quantiles[quant] = []
        for gene in pos_quantiles.get(quant, []):
            if gene in pos_metrics:
                combined_quantiles[quant].append(pos_metrics[gene])
        for gene in neg_quantiles.get(quant, []):
            if gene in neg_metrics:
                combined_quantiles[quant].append(neg_metrics[gene])
    if combined_quantiles:
        plot_combined_trend_quantiles(combined_quantiles, "auroc", 
            "AUROC trend for three gene expression quantiles", 
            os.path.join(trend_dir, "combined_auroc_trend.png"))
        plot_combined_trend_quantiles(combined_quantiles, "auprc", 
            "AUPRC trend for three gene expression quantiles", 
            os.path.join(trend_dir, "combined_auprc_trend.png"))

if __name__ == "__main__":
    main()
