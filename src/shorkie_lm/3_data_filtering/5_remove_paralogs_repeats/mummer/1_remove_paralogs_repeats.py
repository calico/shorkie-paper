import sys
from Bio import SeqIO

def read_ids_to_remove(ids_file):
    """
    Read sequence IDs from a file.

    Parameters:
    ids_file (str): Path to the file containing the IDs to remove.

    Returns:
    set: A set of sequence IDs to remove.
    """
    with open(ids_file, "r") as file:
        ids_to_remove = {line.strip() for line in file}
    return ids_to_remove


def remove_sequences(fasta_file, ids_to_remove, output_file):
    """
    Remove sequences with specified IDs from a FASTA file.

    Parameters:
    fasta_file (str): Path to the input FASTA file.
    ids_to_remove_file (str): Path to the file containing sequence IDs to remove.
    output_file (str): Path to the output FASTA file.
    """
    seq_count = 0
    species_set = set()
    with open(fasta_file, "r") as input_handle, open(output_file, "w") as output_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            if record.id not in ids_to_remove:
                species_set.add(record.id.split("|")[1])
                SeqIO.write(record, output_handle, "fasta-2line")
                seq_count += 1
    return seq_count, len(species_set)


dataset = sys.argv[1]
threshold = "0.070"
out_dir = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/"
test_seqs = 0
valid_seqs = 0
for data_type in ['test', 'valid']:
    # Example usage
    fasta_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/extracted_fasta/sequences_{data_type}.fasta"
    repeat_to_remove_file = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{dataset}/dataset_stats/removed_repeats/{data_type}/removed_sequences_threshold_{threshold}.txt"
    ids_to_remove = read_ids_to_remove(repeat_to_remove_file)    
    print(f"Removing paralogs from {data_type} data in {dataset} dataset...")
    output_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/extracted_fasta/sequences_{data_type}.cleaned.fasta"
    train_seq_count, _ = remove_sequences(fasta_file, ids_to_remove, output_file)
    if data_type == 'test':
        test_seqs = train_seq_count
    elif data_type == 'valid':
        valid_seqs = train_seq_count


for data_type in ['train']:
    # Example usage
    fasta_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/extracted_fasta/sequences_{data_type}.fasta"
    paralog_to_remove_test_file = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{dataset}/dataset_stats/dataset_similarity/mummer/4_filter_aln_{dataset}_train_test_paralogs_for_{data_type}.txt"
    paralog_to_remove_valid_file = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{dataset}/dataset_stats/dataset_similarity/mummer/4_filter_aln_{dataset}_train_valid_paralogs_for_{data_type}.txt"
    repeat_to_remove_file = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{dataset}/dataset_stats/removed_repeats/{data_type}/removed_sequences_threshold_{threshold}.txt"

    print("paralog_to_remove_test_file: ", paralog_to_remove_test_file)
    print("paralog_to_remove_valid_file: ", paralog_to_remove_valid_file)
    print("repeat_to_remove_file : ", repeat_to_remove_file)

    paralog_to_remove_test = read_ids_to_remove(paralog_to_remove_test_file)
    print(len(paralog_to_remove_test))    
    paralog_to_remove_valid = read_ids_to_remove(paralog_to_remove_valid_file)    
    print(len(paralog_to_remove_valid))    
    repeat_to_remove = read_ids_to_remove(repeat_to_remove_file)    
    print(len(repeat_to_remove))    

    merged_set = paralog_to_remove_test.union(paralog_to_remove_valid, repeat_to_remove)
    print("merged_set: ", len(merged_set))
    print(f"Removing paralogs from {data_type} data in {dataset} dataset...")
    output_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/extracted_fasta/sequences_{data_type}.cleaned.fasta"
    train_seq_count, species_count = remove_sequences(fasta_file, merged_set, output_file)
    print("train_seq_count: ", train_seq_count)
    print("species_count: ", species_count)
seq_length = 16384
train_seqs = train_seq_count
num_species = species_count


#Write statistics file
stats_file = out_dir + '/statistics.json'
stats_out = open(stats_file, 'w')
print("{", file=stats_out)
print("\t\"seq_length\": " + str(seq_length) + ",", file=stats_out)
print("\t\"seq_1hot\": true,", file=stats_out)
print("\t\"test_seqs\": " + str(test_seqs) + ",", file=stats_out)
print("\t\"valid_seqs\": " + str(valid_seqs) + ",", file=stats_out)
print("\t\"train_seqs\": " + str(train_seqs) + ",", file=stats_out)
print("\t\"num_species\": " + str(num_species), file=stats_out)
print("}", file=stats_out)
stats_out.close()


#Write dummy targets file
targets_file = out_dir + '/targets.txt'
targets_out = open(targets_file, 'w')
print("\tidentifier\tfile\tclip\tclip_soft\tscale\tsum_stat\tstrand_pair\tdescription", file=targets_out)
print("0\tRNA-MISSI.1\t/home/jlinder/tillage/datasets/yeast/rna/RNA-MISSI.1/coverage.w5\t1\t1\t1.0\tsum_sqrt\t0\tRNA:missing", file=targets_out)
targets_out.close()
