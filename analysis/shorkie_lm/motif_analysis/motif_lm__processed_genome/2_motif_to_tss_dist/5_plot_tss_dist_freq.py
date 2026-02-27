import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_motif_and_background_distributions(true_csv, bg_csv, output_dir="distance_histograms"):
    """
    Plot histogram distributions of TSS distances for each motif, overlaying the true motif distribution
    and the background distribution. For visualization, the sign of the distance is flipped:
      - Negative values indicate upstream (5') of the TSS.
      - Positive values indicate downstream (3') of the TSS.
    
    Saves individual plots as PNG files in the specified directory.

    Parameters:
      true_csv: Path to CSV file with true motif distances (e.g., motif_tss_distances.csv)
      bg_csv:   Path to CSV file with background distances (e.g., background_tss_distances.csv)
      output_dir: Directory to save the plots.
    """
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data.
    df_true = pd.read_csv(true_csv)
    df_bg = pd.read_csv(bg_csv)
    
    # Get list of unique motifs based on the true motif data.
    motifs = df_true['motif_name'].unique()
    
    # Set up consistent plotting parameters
    sns.set_style("whitegrid")  # Set white background with grid lines
    common_bins = 50
    xlim = (-2500, 2500)  # Updated x-axis limits
    
    for motif in motifs:
        # Filter data for the current motif from both datasets
        true_data = df_true[df_true['motif_name'] == motif].copy()
        bg_data = df_bg[df_bg['motif_name'] == motif].copy()
        
        # Flip the sign for visualization
        true_data['flip_distance'] = -true_data['distance']
        bg_data['flip_distance'] = -bg_data['distance']
        
        plt.figure(figsize=(8, 3.6))
        
        # Generate bins based on new xlim
        bin_edges = np.linspace(xlim[0], xlim[1], common_bins + 1)

        # Plot histograms with new colors and no KDE
        sns.histplot(true_data, x='flip_distance', bins=bin_edges,
                     color='#2ca02c', label='True',  # Green color
                     stat='count', alpha=0.7, kde=False)
        
        sns.histplot(bg_data, x='flip_distance', bins=bin_edges,
                     color='#d62728', label='Background',  # Red color
                     stat='count', alpha=0.5, kde=False)
        
        # Add a vertical line at the TSS (0)
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        
        # Customize the plot
        plt.title(f'Distance Distribution for {motif}\n(True: n={len(true_data)}, Background: n={len(bg_data)})')
        plt.xlabel('Distance from TSS (bp)\nNegative = upstream, Positive = downstream')
        plt.ylabel('Count')
        plt.legend()
        plt.xlim(xlim)
        
        # Set facecolor for the figure and axes
        plt.gca().set_facecolor('white')
        plt.gcf().set_facecolor('white')
        
        # Save the plot
        safe_name = motif.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(output_dir, f"{safe_name}_distance_distribution.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()

# Example usage
if __name__ == "__main__":
    exp_species = "strains_select"   # or "schizosaccharomycetales"
    # exp_species = "schizosaccharomycetales"

    root_dir = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/2_motif_to_tss_dist/results/{exp_species}_motif_hits"
    true_csv = f"{root_dir}/motif_tss_distances.csv"          # CSV with true motif TSS distances
    bg_csv = f"{root_dir}/background_tss_distances.csv"         # CSV with background distances
    output_dir=f"results/{exp_species}_motif_hits/distance_histograms"
    plot_motif_and_background_distributions(true_csv, bg_csv, output_dir)