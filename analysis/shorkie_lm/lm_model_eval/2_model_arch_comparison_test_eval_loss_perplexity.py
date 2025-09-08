import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# ---------------------------------------------------
# 1. Define the datasets, model names, and paths
# ---------------------------------------------------
datasets = ["r64", "strains", "saccharomycetales", "fungi_1385"]
model_names = ["Conv_Small", "Conv_Big", "U-Net_Small", "U-Net_Big"]
# model_names = ["U-Net_Small"]

# Map internal dataset names to display labels
dataset_label_map = {
    "r64": "R64_yeast",
    "strains": "80_Strains",
    "saccharomycetales": "165_Saccharomycetales",
    "fungi_1385": "1342_Fungus",
}

model_file_map = {
    "Conv_Small": "small",
    "Conv_Big": "big",
    "U-Net_Small": "unet_small",
    "U-Net_Big": "unet_big",
}

# ---------------------------------------------------
# 2. Parse test files for overall and region-specific metrics
# ---------------------------------------------------
# Dictionaries to store parsed metrics for each dataset
overall_metrics_dict = {}  # keys: dataset, values: {"loss": float, "perplexity": float}
region_metrics_dict = {}   # keys: dataset, values: DataFrame (region metrics)
test_losses_dict = {}      # we use overall loss as the test loss for the grouped bar chart

for ds in datasets:
    # Initialize variables for the current dataset
    overall_loss_val = None
    overall_perplexity_val = None
    region_table_lines = []
    in_region_table = False

    for model_name in model_names:
        model_suffix = model_file_map[model_name]
        print(f"Processing {ds} with model {model_name} ({model_suffix})")
        # Construct the file path based on dataset and model
        if ds == "saccharomycetales":
            if model_name == "Conv_Small":
                model_suffix_fix = "small"
            elif model_name == "Conv_Big":
                model_suffix_fix = "big"
            elif model_name == "U-Net_Small":
                model_suffix_fix = "unet_small_bert_drop"
            elif model_name == "U-Net_Big":
                model_suffix_fix = "unet_big_bert_drop"
        elif ds == "fungi_1385":
            if model_name == "Conv_Small":
                model_suffix_fix = "small_bert"
            elif model_name == "Conv_Big":
                model_suffix_fix = "big_bert"
            elif model_name == "U-Net_Small":
                model_suffix_fix = "unet_small_bert_drop"
            elif model_name == "U-Net_Big":
                model_suffix_fix = "unet_big_bert_drop"
        else:
            model_suffix_fix = model_suffix
            
        if ds == "saccharomycetales":            
            test_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_{ds}_gtf/lm_{ds}_gtf_{model_suffix_fix}/test_testset_perplexity_region/test_testset_perplexity_region.out"
        elif ds == "fungi_1385":
            test_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_{ds}_gtf/lm_{ds}_gtf_{model_suffix_fix}/test_testset_perplexity_region/test_testset_perplexity_region.out"
        else:
            test_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/lm_{ds}_gtf/lm_{ds}_gtf_{model_suffix_fix}/test_testset_perplexity_region/test_testset_perplexity_region.out"
        print("test_file: ", test_file)
        if not os.path.exists(test_file):
            print(f"Warning: test file not found: {test_file}")
            overall_loss_val = np.nan
            overall_perplexity_val = np.nan
            break

        with open(test_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Look for overall metrics
            if line.startswith("Average Categorical Cross-Entropy loss"):
                try:
                    overall_loss_val = float(line.split("=")[-1].strip())
                except ValueError:
                    print(f"Error parsing overall loss in {test_file}")
            elif line.startswith("Overall Perplexity"):
                try:
                    overall_perplexity_val = float(line.split("=")[-1].strip())
                except ValueError:
                    print(f"Error parsing overall perplexity in {test_file}")
            # Check for the start of region-specific metrics table
            elif line.startswith("Region-specific metrics:"):
                in_region_table = True
                continue
            # Once in the region table, collect non-empty lines
            elif in_region_table:
                if line != "":
                    region_table_lines.append(line)

    # Store the parsed overall metrics and test loss (using overall loss)
    overall_metrics_dict[ds] = {"loss": overall_loss_val, "perplexity": overall_perplexity_val}
    test_losses_dict[ds] = overall_loss_val

    # Parse region metrics table if any lines were found
    if region_table_lines:
        table_text = "\n".join(region_table_lines)
        try:
            # Attempt to parse the table. If the header row is present, pandas will use it.
            region_df = pd.read_csv(StringIO(table_text), delim_whitespace=True)
            # If the first column appears to be an index (e.g. unnamed or numeric), drop it.
            first_col = region_df.columns[0].lower()
            if first_col.startswith("unnamed") or region_df[first_col].dtype in [np.int64, np.float64]:
                region_df = region_df.iloc[:, 1:]
        except Exception as e:
            print(f"Error parsing region metrics table in {test_file}: {e}")
            region_df = pd.DataFrame()
        region_metrics_dict[ds] = region_df
    else:
        region_metrics_dict[ds] = pd.DataFrame()

# ---------------------------------------------------
# 3. Plot Grouped Bar Chart for Test Losses by Dataset
# ---------------------------------------------------
plot_data = np.array([test_losses_dict[ds] for ds in datasets])
# Compute y-axis limits on valid (non-NaN) values
valid_data = plot_data[~np.isnan(plot_data)]
if valid_data.size > 0:
    lowest_val = np.min(valid_data)
    highest_val = np.max(valid_data)
else:
    lowest_val, highest_val = 0, 1
x_lim_ratio = 0.003
y_lower_limit = lowest_val * (1 - x_lim_ratio)
y_upper_limit = highest_val * (1 + x_lim_ratio)

n_groups = len(datasets)
n_models = len(model_names)  # Now we have 4 models
x = np.arange(n_groups)
bar_width = 0.18  # Adjusted to fit multiple models

fig, ax = plt.subplots(figsize=(9, 5))  # Increased figure width for clarity
for i, model_name in enumerate(model_names):
    offset = (i - (n_models / 2 - 0.5)) * bar_width  # Centering the bars
    ax.bar(
        x + offset,
        plot_data,
        width=bar_width,
        label=model_name
    )

ax.set_xticks(x)
ax.set_xticklabels([dataset_label_map[ds] for ds in datasets], fontsize=12)
ax.set_ylabel("Test Loss (Cross Entropy)", fontsize=12)
ax.set_title("Comparison of Test Losses by Dataset and Model", fontsize=14)

# Setting the y-axis limits based on valid data
ax.set_ylim([y_lower_limit, y_upper_limit])

# Adding a legend and tighter layout
ax.legend(title="Model", fontsize=10)
plt.tight_layout()
plt.savefig("viz/model_arch_comparison_test_eval_updated.png", dpi=300)
# plt.show()

# ---------------------------------------------------
# 4. Plot Overall Avg CE Loss by Dataset
# ---------------------------------------------------
overall_loss_values = [overall_metrics_dict[ds]["loss"] for ds in datasets]

fig_loss, ax_loss = plt.subplots(figsize=(2.8, 1.6))
x = np.arange(len(datasets))
ax_loss.bar(x, overall_loss_values, width=0.5, color="blue")
ax_loss.set_xticks(x)
ax_loss.set_xticklabels([dataset_label_map[ds] for ds in datasets], fontsize=10)
ax_loss.set_ylabel("Avg Categorical Cross-Entropy Loss", fontsize=10)
ax_loss.set_title("Avg CE Loss by Dataset", fontsize=12)

min_loss_val = np.nanmin(overall_loss_values)
max_loss_val = np.nanmax(overall_loss_values)
margin_loss = (max_loss_val - min_loss_val) * 0.1  # 10% margin
ax_loss.set_ylim([min_loss_val - margin_loss, max_loss_val + margin_loss])

for i, v in enumerate(overall_loss_values):
    if v is not None and not np.isnan(v):
        ax_loss.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("viz/overall_avg_ce_loss.png", dpi=300)
# plt.show()

# ---------------------------------------------------
# 5. Plot Overall Perplexity by Dataset
# ---------------------------------------------------
overall_perplexity_values = [overall_metrics_dict[ds]["perplexity"] for ds in datasets]

fig_perp, ax_perp = plt.subplots(figsize=(2.8, 1.6))
ax_perp.bar(x, overall_perplexity_values, width=0.5, color="green")
ax_perp.set_xticks(x)
ax_perp.set_xticklabels([dataset_label_map[ds] for ds in datasets], fontsize=10)
ax_perp.set_ylabel("Overall Perplexity", fontsize=10)
ax_perp.set_title("Overall Perplexity by Dataset", fontsize=12)

min_perp_val = np.nanmin(overall_perplexity_values)
max_perp_val = np.nanmax(overall_perplexity_values)
margin_perp = (max_perp_val - min_perp_val) * 0.1  # 10% margin
ax_perp.set_ylim([min_perp_val - margin_perp, max_perp_val + margin_perp])

for i, v in enumerate(overall_perplexity_values):
    if v is not None and not np.isnan(v):
        ax_perp.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("viz/overall_perplexity.png", dpi=300)
# plt.show()

# ---------------------------------------------------
# 6. Plot Region-Specific Metrics (gene vs intergenic) for all datasets
# ---------------------------------------------------
# Aggregate region metrics from all datasets and filter for gene and intergenic regions.
region_all_list = []
for ds in datasets:
    region_df = region_metrics_dict[ds]
    if not region_df.empty:
        # Filter rows to only include "gene" and "intergenic" regions (case sensitive)
        region_df = region_df[region_df["region"].isin(["gene", "intergenic"])]
        if not region_df.empty:
            region_df['dataset'] = dataset_label_map[ds]
            region_all_list.append(region_df)

if region_all_list:
    combined_region_df = pd.concat(region_all_list, ignore_index=True)
    
    # Define the desired order for datasets
    desired_order = ["R64_yeast", "80_Strains", "165_Saccharomycetales", "1342_Fungus"]
    
    # ---- Plot Grouped Bar Chart for Avg Loss ----
    # Pivot the DataFrame: rows = dataset, columns = region, values = avg_loss
    pivot_loss = combined_region_df.pivot(index='dataset', columns='region', values='avg_loss')
    # Reindex rows to match the desired order
    pivot_loss = pivot_loss.reindex(desired_order)
    
    fig_group_loss, ax_group_loss = plt.subplots(figsize=(6.3, 3.2))
    x = np.arange(len(pivot_loss.index))
    bar_width = 0.35  # Adjust width for 2 bars per group
    regions = pivot_loss.columns.tolist()
    for i, region in enumerate(regions):
        offset = (i - len(regions)/2) * bar_width + bar_width/2
        ax_group_loss.bar(x + offset, pivot_loss[region], width=bar_width, label=region)
    ax_group_loss.set_xticks(x)
    ax_group_loss.set_xticklabels(pivot_loss.index, fontsize=10)
    ax_group_loss.set_ylabel("Avg Loss", fontsize=10)
    ax_group_loss.set_title("Region-specific Average Loss (Gene vs Intergenic)", fontsize=14)
    
    # Set y-axis limits based on overall min and max values with a margin
    min_loss = np.nanmin(pivot_loss.values)
    max_loss = np.nanmax(pivot_loss.values)
    margin_loss = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
    ax_group_loss.set_ylim([min_loss - margin_loss, max_loss + margin_loss])
    
    ax_group_loss.legend()
    plt.tight_layout()
    plt.savefig("viz/grouped_region_avg_loss.png", dpi=300)
    # plt.show()

    # ---- Plot Grouped Bar Chart for Perplexity ----
    # Pivot the DataFrame: rows = dataset, columns = region, values = perplexity
    pivot_perp = combined_region_df.pivot(index='dataset', columns='region', values='perplexity')
    # Reindex rows to match the desired order
    pivot_perp = pivot_perp.reindex(desired_order)
    
    fig_group_perp, ax_group_perp = plt.subplots(figsize=(6.3, 3.2))
    for i, region in enumerate(pivot_perp.columns.tolist()):
        offset = (i - len(pivot_perp.columns)/2) * bar_width + bar_width/2
        ax_group_perp.bar(x + offset, pivot_perp[region], width=bar_width, label=region)
    ax_group_perp.set_xticks(x)
    ax_group_perp.set_xticklabels(pivot_perp.index, fontsize=10)
    ax_group_perp.set_ylabel("Perplexity", fontsize=10)
    ax_group_perp.set_title("Region-specific Perplexity (Gene vs Intergenic)", fontsize=14)
    
    # Set y-axis limits based on overall min and max values with a margin
    min_perp = np.nanmin(pivot_perp.values)
    max_perp = np.nanmax(pivot_perp.values)
    margin_perp = (max_perp - min_perp) * 0.1 if max_perp > min_perp else 0.1
    ax_group_perp.set_ylim([min_perp - margin_perp, max_perp + margin_perp])
    
    ax_group_perp.legend()
    plt.tight_layout()
    plt.savefig("viz/grouped_region_perplexity.png", dpi=300)
    # plt.show()
else:
    print("No region-specific metrics available for gene and intergenic regions.")
