import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os, sys

coding_ratio = []

def viz(input, output, select_name):
    # Read the input data
    if not os.path.exists(input):
        print(f"File not found: {input}")
        return
    df = pd.read_csv(input, sep=',', header=0)
    
    # Calculate the percentages of coding and non-coding nucleotides
    df['total_nucleotides'] = df['nucleotides_coding'] + df['nucleotides_non_coding']
    df['coding_percentage'] = df['nucleotides_coding'] / df['total_nucleotides']
    df['non_coding_percentage'] = df['nucleotides_non_coding'] / df['total_nucleotides']

    # Print the dataframe for debugging
    # print("Dataframe:\n", df)

    # Calculate the overall coding percentage
    total_coding_nucleotides = df['nucleotides_coding'].sum()
    total_nucleotides = df['total_nucleotides'].sum()
    overall_coding_percentage = (total_coding_nucleotides / total_nucleotides) * 100

    print(f"{select_name}: {overall_coding_percentage:.2f}%")
    if overall_coding_percentage > 5:
        coding_ratio.append(overall_coding_percentage)


    
    # # Group by chromosome and sort by start position within each group
    # grouped = df.groupby('chrom')
    
    # # Determine the number of chromosomes
    # num_chroms = len(grouped)
    
    # # Create subplots with an appropriate layout
    # fig, axes = plt.subplots(nrows=num_chroms, ncols=1, figsize=(28, 6*num_chroms))
    
    # # If there's only one chromosome, axes will not be an array, so we convert it to a list
    # if num_chroms == 1:
    #     axes = [axes]
    
    # for ax, (chrom, group) in zip(axes, grouped):
    #     group_sorted = group.sort_values(by='start')
        
    #     # Plot stacked bar plots
    #     ax.bar(range(len(group_sorted)), group_sorted['coding_percentage'], label='Coding', alpha=0.7)
    #     ax.bar(range(len(group_sorted)), group_sorted['non_coding_percentage'], bottom=group_sorted['coding_percentage'], label='Non-Coding', alpha=0.7)
        
    #     ax.set_title(f'Coding/Non-Coding Percentage for Chromosome {chrom}')
    #     ax.set_xlabel('Entries (sorted by start position)')
    #     ax.set_ylabel('Percentage')
    #     ax.set_xticks(ticks=range(len(group_sorted)))
    #     ax.set_xticklabels(group_sorted['start'], rotation=90)
    #     ax.grid(axis='y', linestyle='--', alpha=0.7)
    #     ax.legend()
    
    # plt.tight_layout()
    # plt.savefig(output, dpi=300)
    # plt.close()


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
    # print(accession_df)

    for fasta_file in os.listdir(FASTA_DIR):
        if fasta_file.endswith(".cleaned.fasta.masked.dust.softmask"):
            # print(fasta_file)
            basename = fasta_file.replace(".cleaned.fasta.masked.dust.softmask", "")
            # print("basename: ", basename)
            select_name = list(accession_df[accession_df["Accession"] == basename]["Name"])[0]

            ovp_input_f = os.path.join(OUTPUT_DIR, f"{basename}_ovp.txt")
            ovp_output_dir = OUTPUT_DIR

            # print("ovp_input_f: ", ovp_input_f) 
            # print("ovp_output_dir: ", ovp_output_dir)

            # Create output directory if it doesn't exist
            if not os.path.exists(ovp_output_dir):
                os.makedirs(ovp_output_dir)

            # print("ovp_input_f: ", ovp_input_f)
            # print("ovp_output_dir: ", ovp_output_dir)

            viz(ovp_input_f, ovp_output_dir, select_name)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Count overlaps between GFF and BED intervals.')
    # parser.add_argument('--input', type=str, help='Path to the input file', default='GCA_000146045_2_ovp.txt')
    # parser.add_argument('--output', type=str, help='Path to the output file (default: overlap_counts.png)', default='overlap_counts.png')
    # args = parser.parse_args()
    # viz(args.input, args.output)
    print("sys.argv: ", sys.argv)   
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_type>")
        sys.exit(1)
    
    data_type = sys.argv[1]
    main(data_type)
    # Plot histogram of repeat ratios
    plt.figure(figsize=(5, 3))
    plt.hist(coding_ratio, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Coding Region Ratios')
    plt.xlabel('Coding Region Ratio (%)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{data_type}_coding_ratio.png", dpi=300)

