import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse

# ---------------------------
# 1. Configuration
# ---------------------------
parser = argparse.ArgumentParser(description="Merge plots")
parser.add_argument("--out_dir", default="./results")
parser.add_argument("--model_arch", default="unet_small_bert_drop")
args = parser.parse_args()

results_dir = os.path.join(args.out_dir, f"{args.model_arch}_various_mlm")

# Image Paths
img_paths = {
    "A": os.path.join(results_dir, "comparison_valid_loss.png"),
    "B": os.path.join(results_dir, "comparison_valid_r.png"),
    "C": os.path.join(args.out_dir, "plot_best_valid_loss_vs_perplexity.png"),
    "D": os.path.join(args.out_dir, "plot_best_valid_r_vs_perplexity.png")
}

output_file = os.path.join(args.out_dir, "merged_plots_A_B_C_D.png")

# ---------------------------
# 2. Plotting
# ---------------------------
# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# Flatten axes for easy iteration if needed, but we map explicitly
# A (0,0), B (0,1)
# C (1,0), D (1,1)

plot_mapping = [
    ("A", axes[0, 0]),
    ("B", axes[0, 1]),
    ("C", axes[1, 0]),
    ("D", axes[1, 1])
]

for label, ax in plot_mapping:
    path = img_paths[label]
    
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis('off')  # Turn off axis lines/ticks for the image container
        
        # Add Label (A, B, C, D)
        # Position: Upper Left, slightly outside
        # transform=ax.transAxes makes (0,0) bottom-left and (1,1) top-right of the axes
        ax.text(-0.05, 1.05, label, transform=ax.transAxes, 
                fontsize=40, fontweight='bold', va='bottom', ha='right')
    else:
        print(f"Warning: Image for {label} not found at {path}")
        ax.text(0.5, 0.5, f"Image {label} not found", ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Merged plot saved to {output_file}")
