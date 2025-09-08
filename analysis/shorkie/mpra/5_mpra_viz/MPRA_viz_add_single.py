import os
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # for 3D scatter plot
from pandas.plotting import parallel_coordinates
from scipy.stats import pearsonr, spearmanr

########################################
# 1. Data Loading Functions
########################################
def load_ground_truth(root_dir):
    """
    Load ground truth expression values from a TSV file.
    Assumes the file has at least two columns with expression values in column 1.
    """
    filename = os.path.join(root_dir, 'filtered_test_data_with_MAUDE_expression.txt')
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    ground_truth = np.array([float(line[1]) for line in lines])
    print("Loaded ground truth, total sequences:", len(ground_truth))
    return ground_truth

def load_reference(root_dir):
    """
    Load reference model predictions from a submission file.
    Assumes the file is a TSV with predictions in column 1.
    """
    filename = os.path.join(root_dir, 'sample_submission.txt')
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    reference = np.array([float(line[1]) for line in lines])
    print("Loaded reference predictions, total predictions:", len(reference))
    return reference

def load_predicted_scores(pickle_file, seq_type='yeast_seqs'):
    """
    Load aggregated NPZ-based predicted scores from a pickle file.
    This pickle file is assumed to be created by your analysis script (aggregated_MPRA_data.pkl).
    The function selects only the gene stats for the given sequence type and aggregates their scores.
    """
    with open(pickle_file, 'rb') as f:
        aggregated_data = pickle.load(f)
    predicted_scores_list = []
    # aggregated_data is a dict with keys for strands: '+' and '-'
    for strand in aggregated_data:
        for gene_stats in aggregated_data[strand]:
            if gene_stats['seq_type'] == seq_type:
                predicted_scores_list.append(gene_stats['average_predicted_scores'])
    if len(predicted_scores_list) == 0:
        raise ValueError(f"No predicted scores found for sequence type: {seq_type}")
    predicted_scores_array = np.array(predicted_scores_list)
    # Aggregate by averaging over genes
    aggregated_pred = np.mean(predicted_scores_array, axis=0)
    print(f"Loaded and aggregated predicted scores for {seq_type} from {len(predicted_scores_list)} genes.")
    return aggregated_pred

def load_csv_indices(csv_file):
    """
    Load the CSV file that contains indices (assumed to be in the column named 'pos').
    """
    df = pd.read_csv(csv_file)
    indices = df['pos'].values
    print(f"Loaded {len(indices)} indices from {csv_file}")
    return indices

########################################
# 2. Visualization Functions
########################################
def create_pairplot(df, output_path):
    sns.pairplot(df)
    plt.suptitle("Pairwise Scatter Matrix", y=1.02)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved pairplot to", output_path)

def create_3d_scatter(gt, pred, ref, output_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt, pred, ref, alpha=0.5)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted Score')
    ax.set_zlabel('Reference Model')
    plt.title("3D Scatter Plot of Scores")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved 3D scatter plot to", output_path)

def plot_residuals(gt, pred, ref, output_path):
    # Compute residuals (difference from ground truth)
    residual_pred = pred - gt
    residual_ref = ref - gt

    plt.figure(figsize=(8, 6))
    plt.hist(residual_pred, bins=50, alpha=0.5, label='Predicted Residuals')
    plt.hist(residual_ref, bins=50, alpha=0.5, label='Reference Residuals')
    plt.xlabel("Residual (Score - Ground Truth)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution Comparison")
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved residual distribution plot to", output_path)

def plot_correlation_heatmap(df, output_path):
    corr = df.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved correlation heatmap to", output_path)

def plot_parallel_coordinates(df, output_path):
    # Create a copy and add an identifier column for parallel_coordinates
    df_parallel = df.copy()
    df_parallel['Index'] = df_parallel.index.astype(str)
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df_parallel, 'Index', cols=["Ground Truth", "Predicted Score", "Reference Model"], colormap=plt.get_cmap("Set1"))
    plt.title("Parallel Coordinates Plot")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved parallel coordinates plot to", output_path)

def bland_altman_plot(data1, data2, title, output_path):
    """
    Create a Bland–Altman plot for two sets of measurements.
    data1: method 1 (e.g. predicted)
    data2: method 2 (e.g. ground truth)
    """
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff)
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color='gray', linestyle='--', label=f"Mean Diff: {md:.3f}")
    plt.axhline(md + 1.96*sd, color='red', linestyle='--', label="±1.96 SD")
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')
    plt.xlabel("Mean of Two Methods")
    plt.ylabel("Difference (Method1 - Method2)")
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Bland–Altman plot to", output_path)

########################################
# 3. Main Processing and Plotting
########################################
def main():
    # Set file paths (adjust these paths as needed)
    root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA'
    predicted_pickle_file = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded/viz/aggregated_MPRA_data.pkl'
    yeast_csv = os.path.join(root_dir, 'test_subset_ids/yeast_seqs.csv')

    # Load scores
    ground_truth = load_ground_truth(root_dir)
    reference = load_reference(root_dir)
    predicted = load_predicted_scores(predicted_pickle_file, seq_type='yeast_seqs')
    yeast_indices = load_csv_indices(yeast_csv)

    # Subset ground truth and reference scores using the yeast indices.
    gt_subset = ground_truth[yeast_indices]
    ref_subset = reference[yeast_indices]
    # We assume that the predicted scores from the aggregated NPZ file are already aligned with the CSV order.
    pred_subset = predicted

    # Create a DataFrame for pairwise comparisons.
    df = pd.DataFrame({
        'Ground Truth': gt_subset,
        'Predicted Score': pred_subset,
        'Reference Model': ref_subset
    })

    # Create an output directory for plots.
    output_dir = os.path.join(root_dir, 'viz_comparison')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Pairwise Scatter Matrix (Pairplot)
    pairplot_path = os.path.join(output_dir, "pairplot.png")
    create_pairplot(df, pairplot_path)

    # 2. 3D Scatter Plot
    scatter3d_path = os.path.join(output_dir, "3d_scatter.png")
    create_3d_scatter(gt_subset, pred_subset, ref_subset, scatter3d_path)

    # 3. Residual/Error Distribution Plot
    residuals_path = os.path.join(output_dir, "residuals.png")
    plot_residuals(gt_subset, pred_subset, ref_subset, residuals_path)

    # 4. Correlation Heatmap
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(df, heatmap_path)

    # 5. Parallel Coordinates Plot
    parallel_coords_path = os.path.join(output_dir, "parallel_coordinates.png")
    plot_parallel_coordinates(df, parallel_coords_path)

    # 6. Bland–Altman Plots for Predicted vs. Ground Truth and Reference vs. Ground Truth
    bland_altman_pred_path = os.path.join(output_dir, "bland_altman_pred_vs_gt.png")
    bland_altman_plot(pred_subset, gt_subset, "Bland–Altman: Predicted vs. Ground Truth", bland_altman_pred_path)
    bland_altman_ref_path = os.path.join(output_dir, "bland_altman_ref_vs_gt.png")
    bland_altman_plot(ref_subset, gt_subset, "Bland–Altman: Reference vs. Ground Truth", bland_altman_ref_path)

    print("All plots generated and saved in:", output_dir)

if __name__ == "__main__":
    main()
