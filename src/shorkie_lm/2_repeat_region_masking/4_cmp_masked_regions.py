from Bio import SeqIO
import numpy as np
import os
import pandas as pd
import sys

data_type = sys.argv[1]
print("data_type: ", data_type)

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

# Function to count masked regions
def count_masked(seq):
    return sum(1 for base in seq if base.islower())

# Function to calculate percentage masked
def calculate_percentage_masked(total_masked_length, total_length):
    return (total_masked_length / total_length) * 100 if total_length > 0 else 0

# Load the sequences from the FASTA files
masked_files_dir = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta/"  # Replace with the actual directory path

# Get masked files
cleaned_files = [f for f in os.listdir(masked_files_dir) if f.endswith(".cleaned.fasta")]
results = []
tsv_results = []

# Compare sequences and find masked regions for each masked file
for cleaned_file in cleaned_files:
    cleaned_file_path = os.path.join(masked_files_dir, cleaned_file)
    
    original_file = cleaned_file.replace(".cleaned.fasta", ".fasta")
    rm_file = cleaned_file.replace(".cleaned.fasta", ".rm.fasta")
    sm_file = cleaned_file.replace(".cleaned.fasta", ".sm.fasta")
    my_mask_file = cleaned_file.replace(".cleaned.fasta", ".cleaned.fasta.masked.dust.softmask")

    original_seqs = SeqIO.to_dict(SeqIO.parse(cleaned_file_path, "fasta"))
    
    # for masked_file in [rm_file, sm_file, my_mask_file]:
    for masked_file in [sm_file]:
        print("masked_file: ", masked_file)
        masked_file_path = os.path.join(masked_files_dir, masked_file)
        masked_seqs = SeqIO.to_dict(SeqIO.parse(masked_file_path, "fasta"))

        # Statistics (reset for each file)
        total_length_original = 0
        total_length_masked = 0
        total_masked_regions = 0
        total_masked_length = 0
        total_overlap_length = 0
        masked_lengths = []
        
        # Compare sequences and find masked regions
        for seq_id in original_seqs:
            original_seq = str(original_seqs[seq_id].seq)
            masked_seq = None
            if seq_id in masked_seqs:
                masked_seq = str(masked_seqs[seq_id].seq)
            elif seq_id[3:] in masked_seqs:
                masked_seq = str(masked_seqs[seq_id[3:]].seq)

            if len(original_seq) != len(masked_seq):
                print(f"Sequence length mismatch for {seq_id}")
                continue

            total_length_original += len(original_seq)
            total_length_masked += len(masked_seq)
            
            original_masked_regions = find_masked_regions(original_seq, original_seq)
            masked_regions = find_masked_regions(original_seq, masked_seq)
            total_masked_regions += len(masked_regions)
            for region in masked_regions:
                start, end = region
                region_length = end - start
                total_masked_length += region_length
                masked_lengths.append(region_length)

        # Calculate percentage_masked for the current masked file
        percentage_masked = calculate_percentage_masked(total_masked_length, total_length_original)
        results.append((masked_file, percentage_masked))

        # Create DataFrame for visualization
        df = pd.DataFrame(results, columns=["Masked Genome File", "Percentage Masked"])

        print(df)
        # Output statistics
        print("\n--- Genome Comparison Statistics ---")
        print(f"Total length of original genome: {total_length_original} bases")
        print(f"Total length of masked genome: {total_length_masked} bases")
        print(f"Total number of masked regions: {total_masked_regions}")
        print(f"Total length of masked regions: {total_masked_length} bases")
        print(f"Percentage of genome masked: {percentage_masked:.2f}%")
        print("\n")


    # # Initialize variables
    # total_length_sm = 0
    # total_length_my_mask = 0
    # total_masked_sm = 0
    # total_masked_my_mask = 0
    # total_overlapping_masked = 0

    # sm_file = cleaned_file.replace(".cleaned.fasta", ".sm.fasta")
    # my_mask_file = cleaned_file.replace(".cleaned.fasta", ".cleaned.fasta.masked.dust.softmask")

    # sm_file_path = os.path.join(masked_files_dir, sm_file)
    # sm_seqs = SeqIO.to_dict(SeqIO.parse(sm_file_path, "fasta"))

    # my_mask_file_path = os.path.join(masked_files_dir, my_mask_file)
    # my_mask_seqs = SeqIO.to_dict(SeqIO.parse(my_mask_file_path, "fasta"))

    # # Compare sequences and find masked regions
    # for seq_id in sm_seqs:
    #     try:
    #         original_seq = str(sm_seqs[seq_id].seq)
    #         my_mask_seq = str(my_mask_seqs["chr"+seq_id].seq)
    #     except KeyError:
    #         continue

    #     if len(original_seq) != len(my_mask_seq):
    #         print(f"Sequence length mismatch for {seq_id}")
    #         continue

    #     total_length_sm += len(original_seq)
    #     total_length_my_mask += len(my_mask_seq)

    #     masked_sm = count_masked(original_seq)
    #     masked_my_mask = count_masked(my_mask_seq)
    #     overlapping_masked = sum(1 for a, b in zip(original_seq, my_mask_seq) if a.islower() and b.islower())

    #     total_masked_sm += masked_sm
    #     total_masked_my_mask += masked_my_mask
    #     total_overlapping_masked += overlapping_masked

    # # Store results in a list for writing to TSV
    # percentage_masked_sm = calculate_percentage_masked(total_masked_sm, total_length_sm)
    # percentage_masked_my_mask = calculate_percentage_masked(total_masked_my_mask, total_length_my_mask)

    # # Calculate the overlapping ratio
    # if total_masked_sm > 0 and total_masked_my_mask > 0:
    #     overlapping_ratio_sm = total_overlapping_masked / total_masked_sm
    #     overlapping_ratio_my_mask = total_overlapping_masked / total_masked_my_mask
    #     overall_overlapping_ratio = total_overlapping_masked / (total_masked_sm + total_masked_my_mask - total_overlapping_masked)
        
    #     print(f"Total masked length in sm file: {total_masked_sm}")
    #     print(f"Total masked length in my mask file: {total_masked_my_mask}")
    #     print(f"Total overlapping masked length: {total_overlapping_masked}")
    #     print(f"Overlapping ratio (sm): {overlapping_ratio_sm:.4f}")
    #     print(f"Overlapping ratio (my mask): {overlapping_ratio_my_mask:.4f}")
        
    #     tsv_results.append((cleaned_file, percentage_masked_sm, percentage_masked_my_mask, overlapping_ratio_sm, overlapping_ratio_my_mask))
    # else:
    #     print("No masked regions found in one or both files.")
    #     tsv_results.append((cleaned_file, percentage_masked_sm, percentage_masked_my_mask, 0, 0))
    # print("tsv_results: ", tsv_results)
    # print("\n\n")

# Convert results to a DataFrame and save as TSV
tsv_df = pd.DataFrame(tsv_results, columns=["Genome File", "Percentage Masked SM", "Percentage Masked My Mask", "Overlapping Ratio SM", "Overlapping Ratio My Mask"])
os.makedirs(f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/{data_type}/repeat_eval", exist_ok=True)
tsv_df.to_csv(f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/{data_type}/repeat_eval/masked_genome_stats.tsv", sep="\t", index=False)
