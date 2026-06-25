import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse

# ---------------------------
# 1. Configuration
# ---------------------------
parser = argparse.ArgumentParser(description="Merge plots BD")
parser.add_argument("--out_dir", default="./results")
parser.add_argument("--model_arch", default="unet_small_bert_drop")
args = parser.parse_args()

results_dir = os.path.join(args.out_dir, f"{args.model_arch}_various_mlm")

# Image Paths
# We only want the current B and D.
# B: comparison_valid_r.png
# D: plot_best_valid_r_vs_perplexity.png
img_paths = {
    "A": os.path.join(results_dir, "comparison_valid_r.png"),
    "B": os.path.join(args.out_dir, "plot_best_valid_r_vs_perplexity.png")
}

output_file = os.path.join(args.out_dir, "merged_plots_BD.png")

# ---------------------------
# 2. Plotting
# ---------------------------
# Create a 1x2 grid with constrained_layout for proper spacing
fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                         gridspec_kw={'wspace': 0.05})

# Map A and B to the two axes
plot_mapping = [
    ("A", axes[0]),
    ("B", axes[1])
]

for label, ax in plot_mapping:
    path = img_paths[label]

    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img, aspect='equal')
        ax.axis('off')

        # Add Label (A, B) — top-left, just above the image
        ax.text(0.0, 1.02, label, transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='bottom', ha='left')
    else:
        print(f"Warning: Image for {label} not found at {path}")
        ax.text(0.5, 0.5, f"Image {label} not found", ha='center', va='center')
        ax.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Merged plot saved to {output_file}")
