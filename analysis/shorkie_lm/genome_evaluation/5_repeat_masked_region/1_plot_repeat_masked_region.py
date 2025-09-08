import matplotlib.pyplot as plt
from Bio import SeqIO
import sys, os
import pandas as pd

repeat_ratios = []

def plot_lowercase_regions(fasta_file, basename, window_size=500):
    # Read the FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    
    # Count the number of sequences
    num_sequences = len(sequences)
    
    # Initialize variables for overall masked ratio
    total_length = 0
    total_lowercase = 0
    
    # Set up the figure and subplots
    fig, axes = plt.subplots(nrows=num_sequences, ncols=1, figsize=(15, num_sequences * 3))
    
    if num_sequences == 1:
        axes = [axes]
    
    for ax, record in zip(axes, sequences):
        chromosome = record.id
        sequence = str(record.seq)
        
        # Update overall counts
        total_length += len(sequence)
        total_lowercase += sum(1 for c in sequence if c.islower())
        
        # Calculate the proportion of lowercase characters in sliding windows
        lower_ratios = []
        positions = range(0, len(sequence) - window_size + 1, window_size)
        for start in positions:
            window = sequence[start:start + window_size]
            lower_count = sum(1 for c in window if c.islower())
            lower_ratios.append(lower_count / window_size)
        
    # Calculate and print the overall masked ratio
    overall_masked_ratio = (total_lowercase / total_length) * 100
    repeat_ratios.append(overall_masked_ratio)
    print(f"{basename}: {overall_masked_ratio*100:.2f}%")

# Usage example
data_type = sys.argv[1]
print("data_type: ", data_type)
fasta_dir = f'/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{data_type}/fasta/'
species_csv = f'/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/species_{data_type}.cleaned.csv'
species_df = pd.read_csv(species_csv)
accession_df = species_df[["Name", "Accession"]]
accession_df["Accession"] = accession_df["Accession"].str.replace(".", "_")
print(accession_df)

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".cleaned.fasta.masked.dust.softmask"):
        basename = fasta_file.replace(".cleaned.fasta.masked.dust.softmask", "")
        select_name = list(accession_df[accession_df["Accession"] == basename]["Name"])[0]
        plot_lowercase_regions(os.path.join(fasta_dir, fasta_file), select_name)

# Plot histogram of repeat ratios
plt.figure(figsize=(5, 3))
plt.hist(repeat_ratios, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Repeat Ratios')
plt.xlabel('Repeat Ratio (%)')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{data_type}repeat_masked_frequency.png", dpi=300)
