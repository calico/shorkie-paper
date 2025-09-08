import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize lists to store metrics
metrics = {"idx": [], "model": [], "pearsonr": [], "pearsonr_norm": [], "r2": [], "r2_norm": []}

for idx in range(8):
    # File paths
    self_supervised_acc_fn = f"/home/khc/projects/yeast_seqNN/self_supervised/exp_105/histone__chip_exo__rna_seq/16bp/self_supervised_unet_big/gene_level_eval/f{idx}c0/acc.txt"
    supervised_acc_fn = f"/home/khc/projects/yeast_seqNN/self_supervised/exp_105/histone__chip_exo__rna_seq/16bp/supervised_unet_big/gene_level_eval/f{idx}c0/acc.txt"
    
    # Load data
    self_supervised_acc_df = pd.read_csv(self_supervised_acc_fn, sep="\t")
    supervised_acc_df = pd.read_csv(supervised_acc_fn, sep="\t")
    
    # Append self-supervised metrics
    metrics["idx"].append(idx)
    metrics["model"].append("self-supervised")
    metrics["pearsonr"].append(self_supervised_acc_df["pearsonr"].mean())
    metrics["pearsonr_norm"].append(self_supervised_acc_df["pearsonr_norm"].mean())
    metrics["r2"].append(self_supervised_acc_df["r2"].mean())
    metrics["r2_norm"].append(self_supervised_acc_df["r2_norm"].mean())
    
    # Append supervised metrics
    metrics["idx"].append(idx)
    metrics["model"].append("supervised")
    metrics["pearsonr"].append(supervised_acc_df["pearsonr"].mean())
    metrics["pearsonr_norm"].append(supervised_acc_df["pearsonr_norm"].mean())
    metrics["r2"].append(supervised_acc_df["r2"].mean())
    metrics["r2_norm"].append(supervised_acc_df["r2_norm"].mean())

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics)


# Set the style
sns.set(style="whitegrid")

# Plot for Pearsonr
plt.figure(figsize=(12, 8))
sns.lineplot(data=metrics_df, x="idx", y="pearsonr", hue="model", marker="o")
plt.title("Pearson Correlation (Pearsonr) Comparison Across Folds")
plt.xlabel("Fold Index")
plt.ylabel("Pearsonr")
plt.legend(title="Model")
plt.savefig("pearsonr_comparison.png", dpi=300)
plt.clf()

# Plot for R2
plt.figure(figsize=(12, 8))
sns.lineplot(data=metrics_df, x="idx", y="r2", hue="model", marker="o")
plt.title("R-squared (R2) Comparison Across Folds")
plt.xlabel("Fold Index")
plt.ylabel("R2")
plt.legend(title="Model")
plt.savefig("r2_comparison.png", dpi=300)
plt.clf()


# Mean values for each model
mean_metrics = metrics_df.groupby("model").mean().reset_index()

# Bar plot for Pearsonr
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_metrics, x="model", y="pearsonr", palette="Blues_d")
plt.title("Mean Pearson Correlation (Pearsonr) Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Pearsonr")
plt.savefig("pearsonr_mean_comparison.png", dpi=300)
plt.clf()

# Bar plot for R2
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_metrics, x="model", y="r2", palette="Greens_d")
plt.title("Mean R-squared (R2) Comparison")
plt.xlabel("Model")
plt.ylabel("Mean R2")
plt.savefig("r2_mean_comparison.png", dpi=300)
plt.clf()





# Filter data for supervised and self-supervised
supervised_data = metrics_df[metrics_df["model"] == "supervised"].sort_values("idx").reset_index(drop=True)
self_supervised_data = metrics_df[metrics_df["model"] == "self-supervised"].sort_values("idx").reset_index(drop=True)

# Plot settings
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5.714))

# Colors for different folds
colors = sns.color_palette("hsv", 8)

# Scatter plot for Pearsonr
plt.subplot(2, 2, 1)
for idx in range(8):
    plt.scatter(x=supervised_data["pearsonr"][idx], y=self_supervised_data["pearsonr"][idx], color=colors[idx], label=f'Fold {idx}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Pearson Correlation (Pearsonr)")
plt.xlabel("Supervised")
plt.ylabel("Self-Supervised")

# Scatter plot for Pearsonr_norm
plt.subplot(2, 2, 2)
for idx in range(8):
    plt.scatter(x=supervised_data["pearsonr_norm"][idx], y=self_supervised_data["pearsonr_norm"][idx], color=colors[idx], label=f'Fold {idx}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Normalized Pearson Correlation (Pearsonr_norm)")
plt.xlabel("Supervised")
plt.ylabel("Self-Supervised")

# Scatter plot for R2
plt.subplot(2, 2, 3)
for idx in range(8):
    plt.scatter(x=supervised_data["r2"][idx], y=self_supervised_data["r2"][idx], color=colors[idx], label=f'Fold {idx}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("R-squared (R2)")
plt.xlabel("Supervised")
plt.ylabel("Self-Supervised")

# Scatter plot for R2_norm
plt.subplot(2, 2, 4)
for idx in range(8):
    plt.scatter(x=supervised_data["r2_norm"][idx], y=self_supervised_data["r2_norm"][idx], color=colors[idx], label=f'Fold {idx}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Normalized R-squared (R2_norm)")
plt.xlabel("Supervised")
plt.ylabel("Self-Supervised")

# Adjust layout to prevent overlap
plt.tight_layout()

# # Add the legend below all plots
plt.legend(loc='lower right', ncol=8, bbox_to_anchor=(1, -0.5), prop={'size': 9})

# Ensure the legend is fully included
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig("comparison_scatter.png", dpi=300)
plt.clf()
