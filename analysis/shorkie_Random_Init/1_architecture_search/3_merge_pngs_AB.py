import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge AB PNG files")
    parser.add_argument("--out_dir", default="./results", help="Output directory")
    args = parser.parse_args()

    root_dir = os.path.join(args.out_dir, "model_variants_comparison")
    
    # Old B becomes A, Old C becomes B
    img_path_a = os.path.join(root_dir, "compare_avg_valid_r.png")    # New A (was B)
    img_path_b = os.path.join(root_dir, "compare_avg_valid_r2.png")   # New B (was C)
    
    out_path = os.path.join(root_dir, "merged_r_r2_AB.png")

    for p in [img_path_a, img_path_b]:
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            return
        
    img_a = mpimg.imread(img_path_a)
    img_b = mpimg.imread(img_path_b)
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot A (Left)
    axes[0].imshow(img_a)
    axes[0].axis('off')
    axes[0].text(-0.05, 1.05, 'A', transform=axes[0].transAxes, 
                 fontsize=36, fontweight='bold', va='top', ha='right')

    # Plot B (Right)
    axes[1].imshow(img_b)
    axes[1].axis('off')
    axes[1].text(-0.05, 1.05, 'B', transform=axes[1].transAxes, 
                 fontsize=36, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    # Adjust spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.1)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved merged plot to: {out_path}")

if __name__ == "__main__":
    main()
