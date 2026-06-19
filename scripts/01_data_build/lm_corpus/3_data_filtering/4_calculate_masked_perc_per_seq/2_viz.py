import sys
import matplotlib.pyplot as plt
import pandas as pd

data_target = sys.argv[1]

for data_type in ["train", "test", "valid"]:
    # Data from the provided file
    fname = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_target}/dataset_stats/removed_repeats/{data_type}/removal_ratios.txt"
    fout = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_target}/dataset_stats/removed_repeats/{data_type}/removal_ratios.png"
    df = pd.read_csv(fname, sep='\t', skiprows=1, header=None, names=['Threshold', 'Ratio'])

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df.iloc[:,0]*100, df.iloc[:,1], color='blue')
    plt.title("Scatter Plot of Thresholds vs Removal Ratios")
    plt.xlabel("Masked Ratio Thresholds (%)")
    plt.ylabel("Removal Ratio")
    plt.tight_layout()
    # plt.axis('square')
    plt.grid(True)
    plt.savefig(fout, dpi=300)
