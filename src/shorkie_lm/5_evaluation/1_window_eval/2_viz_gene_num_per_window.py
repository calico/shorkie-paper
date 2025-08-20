import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os, sys

median_ovps = []
mean_ovps = []
def viz(input_file, output_dir, select_name):
    # Create a unique output filename for each input
    filename = os.path.splitext(os.path.basename(input_file))[0] 
    output_path = os.path.join(output_dir, f"{filename}_overlap_counts.png")

    # Read the CSV file
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    df = pd.read_csv(input_file, sep=',', header=0)
    
    ovp_median = df['complete_within'].median()
    ovp_mean = df['complete_within'].mean()
    print(f"{select_name}: ", ovp_median)
    print(f"{select_name}: ", ovp_mean)
    median_ovps.append(ovp_median)
    mean_ovps.append(ovp_mean)


def main(data_type):
    # Directory containing the fasta files
    FASTA_DIR = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{data_type}/fasta"
    OUTPUT_DIR = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_type}/window_eval"
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    species_csv = f'/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/species_{data_type}.cleaned.csv'
    species_df = pd.read_csv(species_csv)
    accession_df = species_df[["Name", "Accession"]]
    accession_df["Accession"] = accession_df["Accession"].str.replace(".", "_")
    print(accession_df)


    for fasta_file in os.listdir(FASTA_DIR):
        if fasta_file.endswith(".cleaned.fasta.masked.dust.softmask"):
            # print(fasta_file)
            basename = fasta_file.replace(".cleaned.fasta.masked.dust.softmask", "")
            # print("basename: ", basename)
            select_name = list(accession_df[accession_df["Accession"] == basename]["Name"])[0]

            ovp_input_f = os.path.join(OUTPUT_DIR, f"{basename}_ovp.txt")
            ovp_output_dir = OUTPUT_DIR

            # Create output directory if it doesn't exist
            if not os.path.exists(ovp_output_dir):
                os.makedirs(ovp_output_dir)

            # print("ovp_input_f: ", ovp_input_f)
            # print("ovp_output_dir: ", ovp_output_dir)

            viz(ovp_input_f, ovp_output_dir, select_name)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_type>")
        sys.exit(1)
    
    data_type = sys.argv[1]
    main(data_type)
    # Plot histogram of repeat ratios
    plt.figure(figsize=(5, 3))
    print("median_ovps: ", len(median_ovps))
    median_ovps.remove(0)
    plt.hist(median_ovps, bins=6, color='skyblue', edgecolor='black')
    plt.title('Histogram of Repeat Ratios')
    plt.xlabel('Repeat Ratio')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{data_type}_ovp_median.png", dpi=300)

