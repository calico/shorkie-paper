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
    with open(fasta_file, "r") as input_handle, open(output_file, "w") as output_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            if record.id not in ids_to_remove:
                SeqIO.write(record, output_handle, "fasta-2line")


dataset = sys.argv[1]
threshold = 0.035
for data_type in ['test', 'valid']:
    # Example usage
    fasta_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/extracted_fasta/sequences_{data_type}.fasta"
    # repeat_to_remove_file = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{dataset}/dataset_stats/removed_repeats/{data_type}/removed_sequences_threshold_{threshold}.txt"
    # ids_to_remove = read_ids_to_remove(repeat_to_remove_file)    
    ids_to_remove = {}
    print(f"Removing paralogs from {data_type} data in {dataset} dataset...")
    output_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/extracted_fasta/sequences_{data_type}.cleaned.fasta"
    remove_sequences(fasta_file, ids_to_remove, output_file)


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
    # repeat_to_remove = read_ids_to_remove(repeat_to_remove_file)    
    # print(len(repeat_to_remove))    

    merged_set = paralog_to_remove_test.union(paralog_to_remove_valid)
    print("merged_set: ", len(merged_set))
    print(f"Removing paralogs from {data_type} data in {dataset} dataset...")
    output_file = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}/extracted_fasta/sequences_{data_type}.cleaned.fasta"
    remove_sequences(fasta_file, merged_set, output_file)
