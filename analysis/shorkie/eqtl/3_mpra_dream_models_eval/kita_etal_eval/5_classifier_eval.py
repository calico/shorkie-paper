#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Define TSS distance bins and labels
bins = [0, 600, 1000, 2000, 3000, 5000, 8000, np.inf]
labels = ['0-600b', '600b-1kb', '1kb-2kb', '2kb-3kb', '3kb-5kb', '5kb-8kb', '>8kb']

# Define the score variable (from your prediction file, here "logSED")
score = "logSED"

# List of negative sets to analyze
negsets = [1, 2, 3, 4]

# Lists to store ensemble ROC/PR data across negative sets
ensemble_roc_curves = []  # Each item: dict with keys: 'negset', 'fpr', 'tpr', 'auc'
ensemble_pr_curves = []   # Each item: dict with keys: 'negset', 'recall', 'precision', 'ap'

# Dictionaries to store per–negative set trend metrics.
# For each negative set, we store a dict mapping TSS group -> AUROC/AUPRC.
trend_auroc_all = {}
trend_auprc_all = {}
models = ['DREAM_Atten', 'DREAM_CNN', 'DREAM_RNN']

for model in models:
    # Create base output directory
    output_dir = f"results/{model}/viz"
    os.makedirs(output_dir, exist_ok=True)
    for negset in negsets:
        # File paths for your predictions
        pos_file = f"results/{model}/final_pos_predictions.tsv"
        neg_file = f"results/{model}/final_neg_predictions_{negset}.tsv"

        # Read the TSV files
        pos_df = pd.read_csv(pos_file, sep="\t")
        neg_df = pd.read_csv(neg_file, sep="\t")

        # Assign labels: 1 for positive, 0 for negative
        pos_df['label'] = 1
        neg_df['label'] = 0

        # Bin TSS distances (assuming column name 'tss_dist')
        pos_df['distance_bin'] = pd.cut(pos_df['tss_dist'], bins=bins, labels=labels)
        neg_df['distance_bin'] = pd.cut(neg_df['tss_dist'], bins=bins, labels=labels)

        # Create a label type column for plotting convenience
        pos_df['label_type'] = 'Positive'
        neg_df['label_type'] = 'Negative'

        # If fold information is not present, assign a default fold (e.g. 1)
        if 'fold' not in pos_df.columns:
            pos_df['fold'] = 1
        if 'fold' not in neg_df.columns:
            neg_df['fold'] = 1

        # Combine both dataframes
        combined_df = pd.concat([pos_df, neg_df], ignore_index=True)

        # -------------------------
        # Basic ROC/PR Analysis (Individual per negative set)
        # -------------------------
        y_true = combined_df['label'].values
        scores_array = combined_df[score].values
        # take absolute value of scores if needed
        scores_array = np.abs(scores_array)

        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, scores_array)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkred', lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Negative Set {negset})")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f"roc_curve_negative_set{negset}.png"))
        plt.close()

        # Precision-Recall curve and AP
        precision, recall, _ = precision_recall_curve(y_true, scores_array)
        avg_precision = average_precision_score(y_true, scores_array)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkblue', lw=2, label=f"PR (AP = {avg_precision:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (Negative Set {negset})")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, f"pr_curve_negative_set{negset}.png"))
        plt.close()

        ensemble_roc_curves.append({'negset': negset, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})
        ensemble_pr_curves.append({'negset': negset, 'recall': recall, 'precision': precision, 'ap': avg_precision})

        # -------------------------
        # Extended TSS Distance Analysis (Histograms, Violin, Fold-level curves)
        # -------------------------
        hist_dir = os.path.join(output_dir, f"tss_histograms_negative_set{negset}")
        os.makedirs(hist_dir, exist_ok=True)
        folds = sorted(combined_df['fold'].unique())
        for fold in folds:
            plt.figure(figsize=(5, 2.5))
            subset = combined_df[(combined_df['fold'] == fold) & (combined_df['label_type'] == 'Positive')]
            sns.histplot(subset['tss_dist'], bins=50, kde=True, color='blue')
            plt.title(f'Positive TSS Distance')# (Negative Set {negset}) - Fold {fold}')
            plt.xlabel("TSS Distance")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f'pos_tss_distance_fold{fold}_negative_set{negset}.png'), dpi=300)
            plt.close()

            plt.figure(figsize=(5, 2.5))
            subset = combined_df[(combined_df['fold'] == fold) & (combined_df['label_type'] == 'Negative')]
            sns.histplot(subset['tss_dist'], bins=50, kde=True, color='red')
            plt.title(f'Negative TSS Distance')# (Negative Set {negset}) - Fold {fold}')
            plt.xlabel("TSS Distance")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f'neg_tss_distance_fold{fold}_negative_set{negset}.png'), dpi=300)
            plt.close()

        violin_dir = os.path.join(output_dir, f"tss_violin_negative_set{negset}")
        os.makedirs(violin_dir, exist_ok=True)
        g = sns.catplot(
            data=combined_df,
            x='distance_bin',
            y=score,
            hue='label_type',
            col='fold',
            kind='violin',
            split=True,
            col_wrap=1,
            height=4,
            aspect=1.5,
            palette={'Positive': 'blue', 'Negative': 'red'}
        )
        g.set_xticklabels(rotation=45)
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f'Score by TSS Distance Bin and Fold (Negative Set {negset})')
        plt.savefig(os.path.join(violin_dir, f'{score}_violin_by_fold_negative_set{negset}.png'), dpi=300)
        plt.close()

        roc_pr_dir = os.path.join(output_dir, f"roc_pr_by_fold_negative_set{negset}")
        os.makedirs(roc_pr_dir, exist_ok=True)
        fold_colors = plt.cm.Dark2(np.linspace(0, 1, len(folds)))
        plt.figure(figsize=(8, 8))
        for i, fold in enumerate(folds):
            fold_df = combined_df[combined_df['fold'] == fold].dropna(subset=[score, 'label'])
            if fold_df.empty:
                continue
            fold_fpr, fold_tpr, _ = roc_curve(fold_df['label'], np.abs(fold_df[score]))
            fold_roc_auc = auc(fold_fpr, fold_tpr)
            plt.plot(fold_fpr, fold_tpr, lw=1, alpha=0.7, color=fold_colors[i],
                    label=f'Fold {fold} (AUC = {fold_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'r--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC by Fold (Negative Set {negset})')
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(roc_pr_dir, f'{score}_roc_curves_negative_set{negset}.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 8))
        for i, fold in enumerate(folds):
            fold_df = combined_df[combined_df['fold'] == fold].dropna(subset=[score, 'label'])
            if fold_df.empty:
                continue
            fold_precision, fold_recall, _ = precision_recall_curve(fold_df['label'], np.abs(fold_df[score]))
            fold_pr_auc = average_precision_score(fold_df['label'], np.abs(fold_df[score]))
            plt.plot(fold_recall, fold_precision, lw=1, alpha=0.7, color=fold_colors[i],
                    label=f'Fold {fold} (AP = {fold_pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall by Fold (Negative Set {negset})')
        plt.ylim(-0.05, 1.1)
        plt.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(roc_pr_dir, f'{score}_pr_curves_negative_set{negset}.png'), dpi=300)
        plt.close()

        # -------------------------
        # ROC/PR Curves for Each TSS Group and Per–Negative Set Trend Metrics
        # -------------------------
        tss_group_dir = os.path.join(output_dir, f"roc_pr_by_tss_negative_set{negset}")
        os.makedirs(tss_group_dir, exist_ok=True)
        tss_auroc = {}
        tss_auprc = {}
        
        for tss_group in labels:
            subset = combined_df[combined_df['distance_bin'] == tss_group]
            if subset.empty or subset['label'].nunique() < 2:
                continue
            y_true_tss = subset['label'].values
            scores_tss = subset[score].values
            
            fpr_tss, tpr_tss, _ = roc_curve(y_true_tss, scores_tss)
            auroc_tss = auc(fpr_tss, tpr_tss)
            precision_tss, recall_tss, _ = precision_recall_curve(y_true_tss, scores_tss)
            auprc_tss = average_precision_score(y_true_tss, scores_tss)
            
            tss_auroc[tss_group] = auroc_tss
            tss_auprc[tss_group] = auprc_tss
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr_tss, tpr_tss, color='darkred', lw=2, label=f"ROC (AUC = {auroc_tss:.2f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve for TSS Group {tss_group} (Negative Set {negset})")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(tss_group_dir, f"roc_curve_tss_{tss_group}_negative_set{negset}.png"), dpi=300)
            plt.close()
            
            plt.figure(figsize=(6, 5))
            plt.plot(recall_tss, precision_tss, color='darkblue', lw=2, label=f"PR (AP = {auprc_tss:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve for TSS Group {tss_group} (Negative Set {negset})")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(tss_group_dir, f"pr_curve_tss_{tss_group}_negative_set{negset}.png"), dpi=300)
            plt.close()
        
        if tss_auroc and tss_auprc:
            ordered_groups = [g for g in labels if g in tss_auroc]
            auroc_values = [tss_auroc[g] for g in ordered_groups]
            auprc_values = [tss_auprc[g] for g in ordered_groups]
            
            plt.figure(figsize=(8, 6))
            plt.plot(ordered_groups, auroc_values, marker='o', linestyle='-', color='darkred', label="AUROC")
            plt.plot(ordered_groups, auprc_values, marker='o', linestyle='-', color='darkblue', label="AUPRC")
            plt.xlabel("TSS Group")
            plt.ylabel("Metric Value")
            plt.title(f"Trend of AUROC and AUPRC vs. TSS Group (Negative Set {negset})")
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(tss_group_dir, f"trend_metrics_negative_set{negset}.png"), dpi=300)
            plt.close()
        
        trend_auroc_all[negset] = tss_auroc
        trend_auprc_all[negset] = tss_auprc

    # -------------------------
    # Combined Trend Plot Across All Negative Sets
    # (Scatter individual negative set dots (unconnected) and a linear regression trend line with SE)
    # -------------------------
    # Determine ordered TSS groups that have data across any negative set.
    ordered_groups = [g for g in labels if any(g in trend_auroc_all[neg] for neg in trend_auroc_all)]
    x = np.arange(len(ordered_groups))

    # Compute ensemble averages and SE for each TSS group across negative sets.
    mean_auroc = []
    se_auroc = []
    mean_auprc = []
    se_auprc = []
    for g in ordered_groups:
        auroc_vals = []
        auprc_vals = []
        for neg in negsets:
            if g in trend_auroc_all.get(neg, {}):
                auroc_vals.append(trend_auroc_all[neg][g])
            if g in trend_auprc_all.get(neg, {}):
                auprc_vals.append(trend_auprc_all[neg][g])
        if auroc_vals:
            mean_auroc.append(np.mean(auroc_vals))
            se_auroc.append(np.std(auroc_vals, ddof=1)/np.sqrt(len(auroc_vals)) if len(auroc_vals)>1 else 0)
        else:
            mean_auroc.append(np.nan)
            se_auroc.append(0)
        if auprc_vals:
            mean_auprc.append(np.mean(auprc_vals))
            se_auprc.append(np.std(auprc_vals, ddof=1)/np.sqrt(len(auprc_vals)) if len(auprc_vals)>1 else 0)
        else:
            mean_auprc.append(np.nan)
            se_auprc.append(0)

    # Create a figure with two subplots for AUROC and AUPRC trends.
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Define a colormap for negative set dots.
    neg_colors = plt.cm.Set1(np.linspace(0, 1, len(negsets)))

    # For AUROC: scatter each negative set's dot (no connecting lines).
    for i, neg in enumerate(negsets):
        # For each negative set, get the AUROC values for each TSS group (if available).
        xs = []
        ys = []
        for j, g in enumerate(ordered_groups):
            if g in trend_auroc_all.get(neg, {}):
                xs.append(j)
                ys.append(trend_auroc_all[neg][g])
        axes[0].scatter(xs, ys, color=neg_colors[i], label=f"Negative Set {neg}")

    # Fit a linear regression to the ensemble average AUROC (using np.polyfit).
    coeffs_auroc = np.polyfit(x, mean_auroc, 1)
    trend_line_auroc = np.polyval(coeffs_auroc, x)
    axes[0].plot(x, trend_line_auroc, color='k', linewidth=2, label="Ensemble Trend")
    axes[0].fill_between(x, np.array(mean_auroc) - np.array(se_auroc),
                        np.array(mean_auroc) + np.array(se_auroc), color='k', alpha=0.3)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Trend of AUROC vs. TSS Group (Ensemble Across Negative Sets)")
    axes[0].legend(loc="lower right")
    axes[0].set_ylim(0, 1)

    # For AUPRC: scatter each negative set's dot.
    for i, neg in enumerate(negsets):
        xs = []
        ys = []
        for j, g in enumerate(ordered_groups):
            if g in trend_auprc_all.get(neg, {}):
                xs.append(j)
                ys.append(trend_auprc_all[neg][g])
        axes[1].scatter(xs, ys, color=neg_colors[i], label=f"Negative Set {neg}")

    # Fit a linear regression to the ensemble average AUPRC.
    coeffs_auprc = np.polyfit(x, mean_auprc, 1)
    trend_line_auprc = np.polyval(coeffs_auprc, x)
    axes[1].plot(x, trend_line_auprc, color='k', linewidth=2, label="Ensemble Trend")
    axes[1].fill_between(x, np.array(mean_auprc) - np.array(se_auprc),
                        np.array(mean_auprc) + np.array(se_auprc), color='k', alpha=0.3)
    axes[1].set_xlabel("TSS Group")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_title("Trend of AUPRC vs. TSS Group (Ensemble Across Negative Sets)")
    axes[1].legend(loc="lower right")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ordered_groups)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_trend_with_se_all_negative_sets.png"), dpi=300)
    plt.close()

    # -------------------------
    # (Optional) Ensemble Overall ROC/PR Plots Across Negative Sets
    # -------------------------
    fpr_grid = np.linspace(0, 1, 100)
    all_tpr_interp = []
    plt.figure(figsize=(8, 8))
    ens_colors = plt.cm.Pastel1(np.linspace(0, 1, len(ensemble_roc_curves)))
    for i, roc_data in enumerate(ensemble_roc_curves):
        interp_tpr = np.interp(fpr_grid, roc_data['fpr'], roc_data['tpr'])
        all_tpr_interp.append(interp_tpr)
        plt.plot(roc_data['fpr'], roc_data['tpr'], color=ens_colors[i], lw=1, alpha=0.7,
                label=f"Negative Set {roc_data['negset']} (AUC = {roc_data['auc']:.2f})")
    ensemble_tpr = np.mean(all_tpr_interp, axis=0)
    ensemble_auc = auc(fpr_grid, ensemble_tpr)
    plt.plot(fpr_grid, ensemble_tpr, color='k', lw=2.5, label=f"Ensemble Average (AUC = {ensemble_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'r--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Ensemble ROC Curve Across Negative Sets")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_roc_curve.png"), dpi=300)
    plt.close()

    recall_grid = np.linspace(0, 1, 100)
    all_precision_interp = []
    plt.figure(figsize=(8, 8))
    ens_colors = plt.cm.Pastel1(np.linspace(0, 1, len(ensemble_pr_curves)))
    for i, pr_data in enumerate(ensemble_pr_curves):
        interp_precision = np.interp(recall_grid, pr_data['recall'][::-1], pr_data['precision'][::-1])
        all_precision_interp.append(interp_precision)
        plt.plot(pr_data['recall'], pr_data['precision'], color=ens_colors[i], lw=1, alpha=0.7,
                label=f"Negative Set {pr_data['negset']} (AP = {pr_data['ap']:.2f})")
    ensemble_precision = np.mean(all_precision_interp, axis=0)
    ensemble_ap = auc(recall_grid, ensemble_precision)
    plt.plot(recall_grid, ensemble_precision, color='k', lw=2.5, label=f"Ensemble Average (AP = {ensemble_ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Ensemble Precision-Recall Curve Across Negative Sets")
    plt.ylim(-0.05, 1.1)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_pr_curve.png"), dpi=300)
    plt.close()
