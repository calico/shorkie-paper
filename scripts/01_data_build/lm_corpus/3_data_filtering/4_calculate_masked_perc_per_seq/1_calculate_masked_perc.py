import os
import sys
from Bio import SeqIO

def check_nucleotide_ratio(fasta_file, thresholds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total_sequences = 0
    total_nucleotides_all = 0
    lower_case_and_N_count_all = 0
    removal_ratios = {}
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    total_sequences = len(sequences)

    # Calculate overall counts
    for record in sequences:
        sequence = str(record.seq)
        total_nucleotides = len(sequence)
        lower_case_and_N_count = sum(1 for char in sequence if char.islower() or char == 'N')
        
        total_nucleotides_all += total_nucleotides
        lower_case_and_N_count_all += lower_case_and_N_count

    # Calculate removal ratios and store sequence IDs for each threshold
    for threshold in thresholds:
        removed_sequences = 0
        removed_sequence_ids = []
        for record in sequences:
            sequence = str(record.seq)
            total_nucleotides = len(sequence)
            lower_case_and_N_count = sum(1 for char in sequence if char.islower() or char == 'N')
            
            ratio = lower_case_and_N_count / total_nucleotides
            if ratio > threshold:
                removed_sequences += 1
                removed_sequence_ids.append(record.id)
        
        removal_ratio = removed_sequences / total_sequences
        removal_ratios[threshold] = removal_ratio

        # Write removed sequence IDs to a file for the current threshold
        output_file = f"{output_dir}/removed_sequences_threshold_{threshold:.3f}.txt"
        with open(output_file, 'w') as f:
            for seq_id in removed_sequence_ids:
                f.write(f"{seq_id}\n")

    overall_ratio = lower_case_and_N_count_all / total_nucleotides_all

    # Write overall and removal ratios to a file
    ratio_output_file = f"{output_dir}/removal_ratios.txt"
    with open(ratio_output_file, 'w') as f:
        f.write(f"Overall ratio of lowercase or 'N' nucleotides: {overall_ratio:.4f}\n")
        for threshold, removal_ratio in removal_ratios.items():
            f.write(f"{threshold:.3f}\t{removal_ratio:.4f}\n")

    # Print the overall ratio and removal ratios
    print(f"Overall ratio of lowercase or 'N' nucleotides: {overall_ratio:.4f}")
    for threshold, removal_ratio in removal_ratios.items():
        print(f"Threshold: {threshold:.3f}, Sequence removal ratio: {removal_ratio:.4f}")

# Example usage:
data_target = sys.argv[1]
# for data_type in ["train", "test", "valid"]:
for data_type in ["test", "valid"]:
    fasta_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{data_target}/extracted_fasta/sequences_{data_type}.fasta"
    print(f"Calculating masked nucleotide ratios for {data_type} data in {data_target} dataset...")
    thresholds = [i * 0.005 for i in range(1, 16)]
    # thresholds = [0.070]
    output_dir = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_target}/dataset_stats/removed_repeats/{data_type}/"  # Replace with the actual path to your output directory
    check_nucleotide_ratio(fasta_file, thresholds, output_dir)
