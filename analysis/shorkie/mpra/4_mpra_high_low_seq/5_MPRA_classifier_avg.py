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
    high_files = sorted(
        glob.glob(os.path.join(high_dir, f"{target_gene}_context_*.npz")),
        key=get_insertion_position
    )
    low_files = sorted(
        glob.glob(os.path.join(low_dir, f"{target_gene}_context_*.npz")),
        key=get_insertion_position
    )
    if not high_files or not low_files:
        print(f"WARNING: Missing files for gene {gene_name} in one or both directories.")
        return None

    metrics = {}
    for hf, lf in zip(high_files, low_files):
        pos_val = get_insertion_position(hf)
        data_high = np.load(hf)
        scores_high = np.mean(data_high["logSED"], axis=1)
        data_low = np.load(lf)
        scores_low = np.mean(data_low["logSED"], axis=1)

        scores = np.concatenate([scores_high, scores_low])
        labels = np.concatenate([np.ones(len(scores_high)), np.zeros(len(scores_low))])

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = average_precision_score(labels, scores)

        metrics[pos_val] = {
            'auroc': roc_auc,
            'auprc': auprc,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }

    return metrics


def plot_group_roc_pr(metrics_list, group_label, output_dir):
    """
    Given a list of gene metric dictionaries for a group (e.g. all positive genes),
    this function creates multi-panel figures for ROC and PR curves for each insertion position.
    A single shared legend (with gene names and "Aggregate") is placed outside the subplots.
    """
    insertion_positions = sorted(metrics_list[0]["metrics"].keys())

    # Common grids for interpolation
    grid_fpr = np.linspace(0, 1, 100)
    grid_recall = np.linspace(0, 1, 100)

    # Determine layout
    n_panels = len(insertion_positions)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))

    cmap = plt.cm.get_cmap("tab10", len(metrics_list))

    # --- ROC figure ---
    fig_roc, axs_roc = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axs_roc = axs_roc.flatten() if nrows * ncols > 1 else [axs_roc]

    gene_handles = []
    gene_labels = []

    for i, pos in enumerate(insertion_positions):
        ax = axs_roc[i]
        for j, gene_data in enumerate(metrics_list):
            if pos in gene_data["metrics"]:
                fpr, tpr = gene_data["metrics"][pos]["roc_curve"]
                color = cmap(j)
                line, = ax.plot(fpr, tpr, color=color, alpha=0.5)  # lighter gene curves
                if i == 0:
                    gene_handles.append(line)
                    gene_labels.append(f"{gene_data['gene_symbol']} ({gene_data['strand']})")

        # average ROC
        tprs = []
        for gene_data in metrics_list:
            if pos in gene_data["metrics"]:
                fpr, tpr = gene_data["metrics"][pos]["roc_curve"]
                tprs.append(np.interp(grid_fpr, fpr, tpr))
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            ax.plot(grid_fpr, mean_tpr, color='black', lw=2)  # aggregated in black
            ax.fill_between(grid_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='black', alpha=0.2)
            avg_auc = auc(grid_fpr, mean_tpr)
            ax.text(0.05, 0.1, f"AUROC = {avg_auc:.2f}", transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{pos} nt")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True)

    for k in range(i+1, len(axs_roc)):
        fig_roc.delaxes(axs_roc[k])

    agg_handle = Line2D([0], [0], color='black', lw=2)
    handles = gene_handles + [agg_handle]
    labels = gene_labels + ['Aggregate']
    fig_roc.legend(handles, labels, loc='center left', ncol=2, bbox_to_anchor=(0.59, 0.13))
    fig_roc.suptitle(f"{group_label} Genes - ROC Curves", fontsize=22, y=0.925)
    fig_roc.tight_layout(pad=2, h_pad=3, w_pad=3, rect=[0,0,0.88,0.95])
    fig_roc.savefig(os.path.join(output_dir, f"{group_label}_roc_group.png"), bbox_inches='tight', dpi=300)
    plt.close(fig_roc)
    print(f"Saved group ROC plot to {output_dir}")

    # --- PR figure ---
    fig_pr, axs_pr = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axs_pr = axs_pr.flatten() if nrows * ncols > 1 else [axs_pr]

    gene_handles = []
    gene_labels = []

    for i, pos in enumerate(insertion_positions):
        ax = axs_pr[i]
        for j, gene_data in enumerate(metrics_list):
            if pos in gene_data["metrics"]:
                precision, recall = gene_data["metrics"][pos]["pr_curve"]
                recall, precision = recall[::-1], precision[::-1]
                color = cmap(j)
                line, = ax.plot(recall, precision, color=color, alpha=0.5)  # lighter gene curves
                if i == 0:
                    gene_handles.append(line)
                    gene_labels.append(f"{gene_data['gene_symbol']} ({gene_data['strand']})")

        # average PR
        prcs = []
        for gene_data in metrics_list:
            if pos in gene_data["metrics"]:
                precision, recall = gene_data["metrics"][pos]["pr_curve"]
                recall, precision = recall[::-1], precision[::-1]
                prcs.append(np.interp(grid_recall, recall, precision))
        if prcs:
            mean_pr = np.mean(prcs, axis=0)
            std_pr = np.std(prcs, axis=0)
            ax.plot(grid_recall, mean_pr, color='black', lw=2)  # aggregated in black
            ax.fill_between(grid_recall, mean_pr - std_pr, mean_pr + std_pr, color='black', alpha=0.2)
            avg_auprc = auc(grid_recall, mean_pr)
            ax.text(0.05, 0.1, f"AUPRC = {avg_auprc:.2f}", transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{pos} nt")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)

    for k in range(i+1, len(axs_pr)):
        fig_pr.delaxes(axs_pr[k])

    agg_handle = Line2D([0], [0], color='black', lw=2)
    handles = gene_handles + [agg_handle]
    labels = gene_labels + ['Aggregate']
    fig_pr.legend(handles, labels, loc='center left', ncol=2, bbox_to_anchor=(0.59, 0.13))
    fig_pr.suptitle(f"{group_label} Genes - PR Curves", fontsize=22, y=0.925)
    fig_pr.tight_layout(pad=2, h_pad=4, w_pad=4, rect=[0,0,0.88,0.95])
    fig_pr.savefig(os.path.join(output_dir, f"{group_label}_pr_group.png"), bbox_inches='tight', dpi=300)
    plt.close(fig_pr)
    print(f"Saved group PR plot to {output_dir}")

#############################
# Main Experiment
#############################

def main():
    output_res_base = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/all_seq_types"

    load_metrics = True
    pos_metrics_file = os.path.join(output_res_base, "pos_metrics.pkl")
    neg_metrics_file = os.path.join(output_res_base, "neg_metrics.pkl")

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

    # Load or compute metrics
    pos_metrics = {}
    neg_metrics = {}
    if load_metrics and os.path.exists(pos_metrics_file) and os.path.exists(neg_metrics_file):
        with open(pos_metrics_file, "rb") as f:
            pos_metrics = pickle.load(f)
        with open(neg_metrics_file, "rb") as f:
            neg_metrics = pickle.load(f)
        # backward compatibility
        for g, d in list(pos_metrics.items()):
            if "metrics" not in d:
                pos_metrics[g] = {"metrics": d, "gene_symbol": g, "strand": "pos"}
        for g, d in list(neg_metrics.items()):
            if "metrics" not in d:
                neg_metrics[g] = {"metrics": d, "gene_symbol": g, "strand": "neg"}
    else:
        # compute pos
        for sym, tgt in pos_gene_mapping.items():
            high_dir = os.path.join(output_res_base, "high_exp_seqs", f"{sym}_{tgt}_pos_outputs")
            low_dir  = os.path.join(output_res_base, "low_exp_seqs",  f"{sym}_{tgt}_pos_outputs")
            if os.path.isdir(high_dir) and os.path.isdir(low_dir):
                mets = compute_classifier_metrics_for_gene(tgt, sym, "pos", high_dir, low_dir)
                if mets is not None:
                    pos_metrics[sym] = {"metrics": mets, "gene_symbol": sym, "strand": "pos"}
        # compute neg
        for sym, tgt in neg_gene_mapping.items():
            high_dir = os.path.join(output_res_base, "high_exp_seqs", f"{sym}_{tgt}_neg_outputs")
            low_dir  = os.path.join(output_res_base, "low_exp_seqs",  f"{sym}_{tgt}_neg_outputs")
            if os.path.isdir(high_dir) and os.path.isdir(low_dir):
                mets = compute_classifier_metrics_for_gene(tgt, sym, "neg", high_dir, low_dir)
                if mets is not None:
                    neg_metrics[sym] = {"metrics": mets, "gene_symbol": sym, "strand": "neg"}
        # save
        with open(pos_metrics_file, "wb") as f:
            pickle.dump(pos_metrics, f)
        with open(neg_metrics_file, "wb") as f:
            pickle.dump(neg_metrics, f)

    # make dirs
    roc_pr_dir = os.path.join(output_res_base, "roc_pr_plots")
    trend_dir  = os.path.join(output_res_base, "trend_plots")
    group_dir  = os.path.join(output_res_base, "group_roc_pr_plots")
    os.makedirs(roc_pr_dir, exist_ok=True)
    os.makedirs(trend_dir,  exist_ok=True)
    os.makedirs(group_dir,  exist_ok=True)

    # --- Group-level ROC/PR plots per TSS distance ---
    if pos_metrics:
        plot_group_roc_pr(list(pos_metrics.values()), "Positive", group_dir)
    if neg_metrics:
        plot_group_roc_pr(list(neg_metrics.values()), "Negative", group_dir)
    if pos_metrics and neg_metrics:
        plot_group_roc_pr(list(pos_metrics.values()) + list(neg_metrics.values()), "Combined", group_dir)

if __name__ == "__main__":
    main()
