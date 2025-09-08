import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Define the datasets, model names, and paths
# ---------------------------------------------------

datasets = ["r64", "strains", "saccharomycetales", "fungi_1385"]
model_names = ["Conv_Small", "Conv_Big", "U-Net_Small", "U-Net_Big"]
# model_names = ["U-Net_Small"]
# model_names = ["U-Net_Big"]

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
# 2. Collect test losses for each dataset & model
# ---------------------------------------------------

test_losses_dict = {ds: [] for ds in datasets}

for ds in datasets:
    for model_name in model_names:
        # Construct the file path based on dataset and model
        model_suffix = model_file_map[model_name]
        print(f"Processing {ds} with model {model_name} ({model_suffix})")

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
            test_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_{ds}_gtf/lm_{ds}_gtf_{model_suffix_fix}/test_testset/test_testset.out"
        elif ds == "fungi_1385":
            test_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_{ds}_gtf/lm_{ds}_gtf_{model_suffix_fix}/test_testset/test_testset.out"
        else:
            test_file = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/lm_{ds}_gtf/lm_{ds}_gtf_{model_suffix_fix}/test/test.out"
        print("test_file: ", test_file)
        if not os.path.exists(test_file):
            print(f"Warning: test file not found: {test_file}")
            test_losses_dict[ds].append(np.nan)
            continue
        
        # Read the final line and parse out the CE value
        with open(test_file, 'r') as f:
            line_raw = f.readlines()[-1].strip()
            ce_val = float(line_raw.split("=")[-1])
        
        test_losses_dict[ds].append(ce_val)

# ---------------------------------------------------
# 3. Prepare data for plotting
# ---------------------------------------------------

plot_data = np.array([test_losses_dict[ds] for ds in datasets])

# Mask NaNs if present and compute min only on valid data
valid_data = plot_data[~np.isnan(plot_data)]
lowest_val = np.min(valid_data)
highest_val = np.max(valid_data)

x_lim_ratio = 0.003
y_lower_limit = lowest_val * (1 - x_lim_ratio)
y_upper_limit = highest_val * (1 + x_lim_ratio)

# ---------------------------------------------------
# 4. Plot a grouped bar chart
# ---------------------------------------------------

n_groups = len(datasets)
n_models = len(model_names)

x = np.arange(n_groups)
bar_width = 0.18  # Reduced bar width for clarity

fig, ax = plt.subplots(figsize=(10, 5))  # Increased figure width for better legibility
# fig, ax = plt.subplots(figsize=(5,3))

for i in range(n_models):
    offset = (i - (n_models / 2 - 0.5)) * bar_width  # Centering the bars
    ax.bar(
        x + offset,
        plot_data[:, i],
        width=bar_width,
        label=model_names[i]
    )

# ---------------------------------------------------
# 5. Final formatting of the plot
# ---------------------------------------------------

# Use the mapped labels for the x-ticks
ax.set_xticks(x)
ax.set_xticklabels([dataset_label_map[ds] for ds in datasets], fontsize=12)

ax.set_ylabel("Test Loss (Cross Entropy)", fontsize=12)
ax.set_title("Comparison of Test Losses by Dataset and Model", fontsize=14)

# Set the y-limits to make differences clearer
ax.set_ylim([y_lower_limit, y_upper_limit])

# Position the legend outside the plot for clarity
ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)

plt.tight_layout()
plt.savefig("viz/model_arch_comparison_test_eval_updated.png", dpi=300)
# plt.show()
