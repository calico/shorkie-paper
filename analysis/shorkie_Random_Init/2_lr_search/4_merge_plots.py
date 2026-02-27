import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse

def merge_plots():
    parser = argparse.ArgumentParser(description="Merge PNG files")
    parser.add_argument("--out_dir", default="./results", help="Output directory")
    args = parser.parse_args()

    base_dir = os.path.join(args.out_dir, "lr_comparison")
    
    # Files mapping to layout
    # We want: 
    # Row 1: A (Loss Avg), B (R Avg)
    # Row 2: C (Loss Bar), D (R Bar)
    
    files = [
        "valid_r_avg.png",     # A (was B)
        "valid_r_bar.png"      # B (was D)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes = axes.flatten()
    
    # Add titles A, B? Or just plot them. User didn't ask for labels, just merge.
    
    labels = ['A', 'B']
    
    for i, fname in enumerate(files):
        ax = axes[i]
        path = os.path.join(base_dir, fname)
        
        if os.path.exists(path):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis('off')
            
            # Add label
            # Position relative to axes: x=0, y=1 (top-left)
            # Adjust slightly if needed. Since axis is off, 0,1 is corner of the image area.
            # Using transform=ax.transAxes ensures relative positioning.
            # -0.05, 1.05 puts it slightly outside.
            ax.text(-0, 1.08, labels[i], transform=ax.transAxes, 
                    fontsize=50, fontweight='bold', va='top', ha='right')
        else:
            print(f"Warning: File not found {path}")
            ax.text(0.5, 0.5, f"File not found:\n{fname}", ha='center')
            ax.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(base_dir, "merged_valid_metrics.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved merged plot to {out_path}")

if __name__ == "__main__":
    merge_plots()
