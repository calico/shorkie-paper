from Bio import SeqIO
import numpy as np
import os, sys
import pandas as pd
import subprocess

def run_bash_command(command):
    """Runs a Bash command and returns the output and error (if any)."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    output = result.stdout.strip()  # Remove trailing whitespace
    error = result.stderr.strip()   # Remove trailing whitespace
    return output, error

# Function to find masked regions
def find_masked_regions(original_seq, masked_seq):
    masked_regions = []
    in_masked_region = False
    start = 0
    
    for i in range(len(original_seq)):
        if original_seq[i] != masked_seq[i]:
            if not in_masked_region:
                in_masked_region = True
                start = i
        else:
            if in_masked_region:
                in_masked_region = False
                masked_regions.append((start, i))
    if in_masked_region:
        masked_regions.append((start, len(original_seq)))
    
    return masked_regions

data_type = sys.argv[1]
# Load the sequences from the FASTA files
masked_files_dir = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta/"  # Replace with the actual directory path

# Get masked files
masked_files = [f for f in os.listdir(masked_files_dir) if f.endswith(".rm.fasta")]
results = []

# Compare sequences and find masked regions for each masked file
for masked_file in masked_files:
    masked_file_path = os.path.join(masked_files_dir, masked_file)
    # masked_seqs = SeqIO.to_dict(SeqIO.parse(masked_file_path, "fasta"))

    original_file = masked_file_path.replace(".rm.fasta", ".fasta")
    
    # Calculate total chromosome length
    total_length = 0
    for record in SeqIO.parse(original_file, "fasta"):
        total_length += len(record.seq)

    print("masked_file_path: ", masked_file_path)
    # Your Bash command
    rm_bash_command = f"grep -o -i N {masked_file_path} | wc -l"
    # Execute and get results
    rm_output, rm_error = run_bash_command(rm_bash_command)
    print("original_file: ", original_file)
    # Your Bash command
    origin_bash_command = f"grep -o -i N {original_file} | wc -l"
    # Execute and get results
    origin_output, origin_error = run_bash_command(origin_bash_command)
    
    masked_n = int(rm_output) - int(origin_output)
    masked_n_ratio = masked_n / total_length
    print("masked_n: ", masked_n)
    print("masked_n_ratio: ", masked_n_ratio)

    # Store results
    file_results = {
        "filename": masked_file,
        "total_length": total_length,
        "ns_in_original": int(origin_output),
        "ns_in_masked": int(rm_output),
        "masked_n": masked_n,
        "masked_n_ratio (%)": masked_n_ratio*100
    }
    results.append(file_results)

# Create DataFrame and print results
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv(f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/masked_n_results.csv", index=False)