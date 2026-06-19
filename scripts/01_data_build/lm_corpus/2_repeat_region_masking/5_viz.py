import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from shorkie import config

CORPUS_BUILD_RESULTS_ROOT = str(config.path('corpus_build_results_root'))

data_type = sys.argv[1]
print("data_type: ", data_type)

df = pd.read_csv("masked_genome_stats.tsv", delimiter="\t")
print("df: ", df)

# Create the scatter plots
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Define colors for each genome file for consistent coloring
colors = plt.cm.tab20(range(len(df)))

# First scatter plot
for i, (x, y, label) in enumerate(zip(df["Percentage Masked SM"], df["Percentage Masked My Mask"], df["Genome File"])):
    label = label.split(".")[0]
    ax[0].scatter(x, y, color=colors[i], label=label)
ax[0].set_xlabel("Percentage Masked SM")
ax[0].set_ylabel("Percentage Masked My Mask")
ax[0].set_title("Percentage Masked Comparison")

# Add legend outside the plot
ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# Second scatter plot
for i, (x, y, label) in enumerate(zip(df["Overlapping Ratio SM"], df["Overlapping Ratio My Mask"], df["Genome File"])):
    label = label.split(".")[0]
    ax[1].scatter(x, y, color=colors[i], label=label)
ax[1].set_xlabel("Overlapping Ratio SM")
ax[1].set_ylabel("Overlapping Ratio My Mask")
ax[1].set_title("Overlapping Ratio Comparison")

# Add legend outside the plot
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()
# rect=[0, 0, 0.75, 1])
os.makedirs(f"{CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/{data_type}/repeat_eval", exist_ok=True)
plt.savefig(f"{CORPUS_BUILD_RESULTS_ROOT}/ensembl_fungi_59/{data_type}/repeat_eval/masked_genome_stats.png", dpi=300)