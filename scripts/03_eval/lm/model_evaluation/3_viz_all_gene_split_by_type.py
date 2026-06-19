import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize a list to collect dataframes for all folds
all_folds = []

# Iterate over each fold (0 through 7)
for idx in range(8):
# for idx in range(1):
    # File paths
    self_supervised_acc_fn = f"/home/khc/projects/yeast_seqNN/self_supervised/exp_105/histone__chip_exo__rna_seq/16bp/self_supervised_unet_big/gene_level_eval/f{idx}c0/acc.txt"
    supervised_acc_fn = f"/home/khc/projects/yeast_seqNN/self_supervised/exp_105/histone__chip_exo__rna_seq/16bp/supervised_unet_big/gene_level_eval/f{idx}c0/acc.txt"
    
    # Load data
    self_supervised_acc_df = pd.read_csv(self_supervised_acc_fn, sep="\t")
    supervised_acc_df = pd.read_csv(supervised_acc_fn, sep="\t")
    print(self_supervised_acc_df.head())    
    print(supervised_acc_df.head())
    
    # Merge the dataframes on the 'identifier' and 'description' columns
    merged_df = pd.merge(self_supervised_acc_df, supervised_acc_df, on=['identifier', 'description', 'group'], suffixes=('_self', '_sup'))
    
    # Append to the list of folds
    all_folds.append(merged_df)

# Concatenate all the folds dataframes
all_folds_df = pd.concat(all_folds)

# Compute the average for each identifier and description across the folds
mean_df = all_folds_df.groupby(['identifier', 'description', 'group']).mean().reset_index()

# print(mean_df.head())   
# Split into three groups based on the description
chip_mnase_df = mean_df[mean_df['group'] == 'CHiP-exo']
rnaseq_df = mean_df[mean_df['group'] == 'RNA-seq']
h_k_df = mean_df[mean_df['group'] == 'Histone Marks']


# h_k_df = mean_df[mean_df['description'].str.contains('[HK]', regex=True)]

# List of dataframes and corresponding labels for plotting
groups = [(chip_mnase_df, 'CHiP-exo'), (rnaseq_df, 'RNAseq'), (h_k_df, 'H and K characters')]

# Metrics to plot
metrics = ['pearsonr', 'r2', 'pearsonr_norm', 'r2_norm']
metrics_print = ['Pearson R', 'R2', 'Pearson R Norm', 'R2 Norm']

# Plot each group
for group_df, group_label in groups:
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axs[i]
        sns.scatterplot(x=f"{metric}_sup", y=f"{metric}_self", data=group_df, ax=ax)
        
        # Calculate the average for the x and y axes
        x_mean = group_df[f"{metric}_sup"][group_df[f"{metric}_sup"] > 0].mean()
        y_mean = group_df[f"{metric}_self"][group_df[f"{metric}_self"] > 0].mean()
        # print(f"Average {metrics_print[i]} for supervised learning: {x_mean}") 
        # print(f"Average {metrics_print[i]} for self-supervised learning: {y_mean}")   
        
        # Add the average as a special dot
        ax.scatter(x_mean, y_mean, color='red', s=100, edgecolor='black', label='Mean')

        ax.plot([-0.05, 1.05], [-0.05, 1.05], 'k--', alpha=0.6)  # Diagonal line
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal', adjustable='box')  # Ensure the subplot is square
        ax.set_title(f"{group_label} - {metric} comparison")
        ax.set_xlabel("Supervised")
        ax.set_ylabel("Self-supervised")

        # # Add the number of dots to the plot
        # num_dots = len(group_df)
        # ax.text(0.05, 0.95, f'n = {num_dots}', transform=ax.transAxes,
        #         fontsize=12, verticalalignment='top', color='black')
    plt.tight_layout()
    plt.savefig(f'3_comparison_scatter_{group_label}.png', dpi=300)
    plt.clf()


colors = ["#6699FF", "#66FF66", "#EE4B2B"]
mean_colors = ["#0000A0", "#005000", "#800020"]
# Plot all groups in the same scatter plot
# for metric in metrics:
for index, metric in enumerate(metrics):
    plt.figure(figsize=(6, 6))
    
    for gp_index, (group_df, group_label) in enumerate(groups):
        scatter = sns.scatterplot(x=f"{metric}_sup", y=f"{metric}_self", data=group_df, label=group_label, color=colors[gp_index])


    for gp_index, (group_df, group_label) in enumerate(groups):
        # Calculate and plot the mean with the same color
        x_mean = group_df[f"{metric}_sup"][group_df[f"{metric}_sup"] > 0].mean()
        y_mean = group_df[f"{metric}_self"][group_df[f"{metric}_self"] > 0].mean()
        plt.scatter(x_mean, y_mean, s=100, edgecolor='black', color=mean_colors[gp_index], label=f'{group_label} Mean')
    
    # Plot the diagonal line
    plt.plot([-0.05, 1.05], [-0.05, 1.05], 'k--', alpha=0.6)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{metrics_print[index]} Comparison Across Groups")
    plt.xlabel("Supervised")
    plt.ylabel("Self-supervised")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'3_comparison_scatter_all_{metric}.png', dpi=300)



# # Set the style
# sns.set(style="whitegrid")

# # Plot for Pearsonr
# plt.figure(figsize=(12, 8))
# sns.lineplot(data=metrics_df, x="idx", y="pearsonr", hue="model", marker="o")
# plt.title("Pearson Correlation (Pearsonr) Comparison Across Folds")
# plt.xlabel("Fold Index")
# plt.ylabel("Pearsonr")
# plt.legend(title="Model")
# plt.savefig("pearsonr_comparison.png", dpi=300)
# plt.clf()

# # Plot for R2
# plt.figure(figsize=(12, 8))
# sns.lineplot(data=metrics_df, x="idx", y="r2", hue="model", marker="o")
# plt.title("R-squared (R2) Comparison Across Folds")
# plt.xlabel("Fold Index")
# plt.ylabel("R2")
# plt.legend(title="Model")
# plt.savefig("r2_comparison.png", dpi=300)
# plt.clf()


# # Mean values for each model
# mean_metrics = metrics_df.groupby("model").mean().reset_index()

# # Bar plot for Pearsonr
# plt.figure(figsize=(10, 6))
# sns.barplot(data=mean_metrics, x="model", y="pearsonr", palette="Blues_d")
# plt.title("Mean Pearson Correlation (Pearsonr) Comparison")
# plt.xlabel("Model")
# plt.ylabel("Mean Pearsonr")
# plt.savefig("pearsonr_mean_comparison.png", dpi=300)
# plt.clf()

# # Bar plot for R2
# plt.figure(figsize=(10, 6))
# sns.barplot(data=mean_metrics, x="model", y="r2", palette="Greens_d")
# plt.title("Mean R-squared (R2) Comparison")
# plt.xlabel("Model")
# plt.ylabel("Mean R2")
# plt.savefig("r2_mean_comparison.png", dpi=300)
# plt.clf()





# # Filter data for supervised and self-supervised
# supervised_data = metrics_df[metrics_df["model"] == "supervised"].sort_values("idx").reset_index(drop=True)
# self_supervised_data = metrics_df[metrics_df["model"] == "self-supervised"].sort_values("idx").reset_index(drop=True)

# # Plot settings
# sns.set(style="whitegrid")
# plt.figure(figsize=(8, 5.714))

# # Colors for different folds
# colors = sns.color_palette("hsv", 8)

# # Scatter plot for Pearsonr
# plt.subplot(2, 2, 1)
# for idx in range(8):
#     plt.scatter(x=supervised_data["pearsonr"][idx], y=self_supervised_data["pearsonr"][idx], color=colors[idx], label=f'Fold {idx}')
# plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title("Pearson Correlation (Pearsonr)")
# plt.xlabel("Supervised")
# plt.ylabel("Self-Supervised")

# # Scatter plot for Pearsonr_norm
# plt.subplot(2, 2, 2)
# for idx in range(8):
#     plt.scatter(x=supervised_data["pearsonr_norm"][idx], y=self_supervised_data["pearsonr_norm"][idx], color=colors[idx], label=f'Fold {idx}')
# plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title("Normalized Pearson Correlation (Pearsonr_norm)")
# plt.xlabel("Supervised")
# plt.ylabel("Self-Supervised")

# # Scatter plot for R2
# plt.subplot(2, 2, 3)
# for idx in range(8):
#     plt.scatter(x=supervised_data["r2"][idx], y=self_supervised_data["r2"][idx], color=colors[idx], label=f'Fold {idx}')
# plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title("R-squared (R2)")
# plt.xlabel("Supervised")
# plt.ylabel("Self-Supervised")

# # Scatter plot for R2_norm
# plt.subplot(2, 2, 4)
# for idx in range(8):
#     plt.scatter(x=supervised_data["r2_norm"][idx], y=self_supervised_data["r2_norm"][idx], color=colors[idx], label=f'Fold {idx}')
# plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title("Normalized R-squared (R2_norm)")
# plt.xlabel("Supervised")
# plt.ylabel("Self-Supervised")

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # # Add the legend below all plots
# plt.legend(loc='lower right', ncol=8, bbox_to_anchor=(1, -0.5), prop={'size': 9})

# # Ensure the legend is fully included
# plt.subplots_adjust(bottom=0.15)

# # Save the figure
# plt.savefig("comparison_scatter.png", dpi=300)
# plt.clf()
