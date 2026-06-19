import os, sys
import matplotlib.pyplot as plt

def read_aln_stats(aln_stats_file):
    """
    Read alignment statistics from a file.

    Parameters:
    aln_stats_file (str): Path to the file containing alignment statistics.

    Returns:
    dict: A dictionary of alignment statistics.
    """
    ids = []
    cov = []
    with open(aln_stats_file, "r") as file:
        # aln_stats = {line.split()[0]: int(line.split()[1]) for line in file}
        lines = file.read().splitlines()
        # Skip the header lines
        for line in lines:
            parts = line.split(' ')
            ids.append(float(parts[0]))
            cov.append(float(parts[1]))
    return ids, cov

def main():
    data_type = sys.argv[1]
    print("data_type: ", data_type)
    x_threshold = 80
    y_threshold = 10
    for target in ["test", "valid"]:
        aln_fn = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_type}/dataset_stats/dataset_similarity/mummer/5_viz_target_{data_type}_train_{target}.txt"
        print("aln_fn: ", aln_fn)
        ids, cov = read_aln_stats(aln_fn)
        # Create a square plot
        plt.figure(figsize=(6, 6))  # 6x6 inches for a square figure
        plt.scatter(ids, cov, s=10)
        # Set the x and y axis limits from 0 to 1
        plt.xlim(0, 105)
        plt.ylim(0, 105)
        plt.vlines(80, 0, 105, colors='r', linestyles='dashed')
        plt.hlines(10, 0, 105, colors='r', linestyles='dashed')
        plt.xlabel(f"Query ({target}) Alignment Identity (%)")
        plt.ylabel(f"Query ({target}) Coverage (%)")
        plt.title(f"Alignments between train and ({target}) datasets", fontsize=15)
        # Add a filled region for x > 80 and y > 10
        plt.fill_betweenx([y_threshold, 105], x_threshold, 105, color='red', alpha=0.2)
        plt.fill_betweenx([0, y_threshold], x_threshold, 105, color='green', alpha=0.2)
        plt.fill_betweenx([0, 105], 0, x_threshold, color='green', alpha=0.2)
        plt.savefig(f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_type}/dataset_stats/dataset_similarity/mummer/6_viz_target_{data_type}_train_{target}.png", dpi=300)
        plt.clf()


if __name__ == "__main__":  
    main()