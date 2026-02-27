import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge PNG files")
    parser.add_argument("--out_dir", default="./results", help="Output directory")
    args = parser.parse_args()

    root_dir = os.path.join(args.out_dir, "model_variants_comparison")
    img_path_a = os.path.join(root_dir, "compare_avg_valid_loss.png") # A
    img_path_b = os.path.join(root_dir, "compare_avg_valid_r.png")    # B
    img_path_c = os.path.join(root_dir, "compare_avg_valid_r2.png")   # C
    
    out_path = os.path.join(root_dir, "merged_loss_r_r2.png")

    for p in [img_path_a, img_path_b, img_path_c]:
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            return
        
    img_a = mpimg.imread(img_path_a)
    img_b = mpimg.imread(img_path_b)
    img_c = mpimg.imread(img_path_c)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot A (Top-Left)
    axes[0, 0].imshow(img_a)
    axes[0, 0].axis('off')
    axes[0, 0].text(-0.05, 1.05, 'A', transform=axes[0, 0].transAxes, 
                 fontsize=36, fontweight='bold', va='top', ha='right')

    # Plot B (Top-Right)
    axes[0, 1].imshow(img_b)
    axes[0, 1].axis('off')
    axes[0, 1].text(-0.05, 1.05, 'B', transform=axes[0, 1].transAxes, 
                 fontsize=36, fontweight='bold', va='top', ha='right')

    # Plot C (Bottom-Left)
    axes[1, 0].imshow(img_c)
    axes[1, 0].axis('off')
    axes[1, 0].text(-0.05, 1.05, 'C', transform=axes[1, 0].transAxes, 
                 fontsize=36, fontweight='bold', va='top', ha='right')
    
    # Hide Bottom-Right
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.1, hspace=0.1)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved merged plot to: {out_path}")

if __name__ == "__main__":
    main()
