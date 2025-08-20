import csv
import random
import json

def reverse_complement(seq):
    """
    Return the reverse complement of a DNA sequence.
    Uses str.maketrans for efficient translation.
    """
    complement_table = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement_table)[::-1]

def sample_csv_rows(input_file, sample_size=1000, remove_ids=set()):
    """
    Read the CSV file and store each row along with its original row number.
    Skip rows whose 'pos' (converted to int) is in remove_ids.
    Then, if there are more than sample_size rows remaining, randomly sample sample_size rows.
    Returns a list of sampled rows.
    """
    with open(input_file, 'r') as fin:
        reader = csv.DictReader(fin, delimiter=',')
        rows = []
        # Annotate each row with its original row number (starting at 1)
        for i, row in enumerate(reader, start=1):
            # Assumes a 'pos' column is present with a numerical value.
            if int(row['pos']) in remove_ids:
                # Optionally, print or log the skipped rows.
                print(f"Skipping row {i} with pos {row['pos']}")
            else:
                row['_orig_row'] = i
                rows.append(row)
        
        # Sample the rows if necessary.
        if len(rows) > sample_size:
            sampled_rows = random.sample(rows, sample_size)
        else:
            sampled_rows = rows
        print(f"Sampled {len(sampled_rows)} rows")
    return sampled_rows

def write_original_sequences(sampled_rows, output_file):
    """
    Write a TSV file with columns:
    id, seq
    using the sampled rows with the sequence in its original orientation.
    """
    with open(output_file, 'w') as fout:
        fout.write("id\tseq\n")
        for counter, row in enumerate(sampled_rows, start=1):
            sequence = row['sequence']
            fout.write(f"seq{counter}\t{sequence}\n")

def write_reverse_complement_sequences(sampled_rows, output_file):
    """
    Write a TSV file with columns:
    id, seq
    where the sequence is reverse complemented.
    """
    with open(output_file, 'w') as fout:
        fout.write("id\tseq\n")
        for counter, row in enumerate(sampled_rows, start=1):
            rev_comp = reverse_complement(row['sequence'])
            fout.write(f"seq{counter}\t{rev_comp}\n")

def write_sampled_row_ids(sampled_rows, output_file):
    """
    Write a TSV file mapping the new sample IDs to the original CSV row numbers.
    """
    with open(output_file, 'w') as fout:
        fout.write("sample_id\toriginal_row_id\n")
        for counter, row in enumerate(sampled_rows, start=1):
            orig_id = row['_orig_row']
            fout.write(f"seq{counter}\t{orig_id}\n")

# Load public leaderboard indices from JSON files.
# These JSON files are assumed to have keys that can be converted to integers.
with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/high_exp_indices.json', 'r') as f:
    public_high = [int(ind) for ind in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/low_exp_indices.json', 'r') as f:
    public_low = [int(ind) for ind in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/yeast_exp_indices.json', 'r') as f:
    public_yeast = [int(ind) for ind in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/random_exp_indices.json', 'r') as f:
    public_random = [int(ind) for ind in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/challenging_exp_indices.json', 'r') as f:
    public_challenging = [int(ind) for ind in list(json.load(f).keys())]

# Combine all public leaderboard indices into a single set.
remove_ids = set(public_high + public_low + public_yeast + public_random + public_challenging)

print(f"Number of public leaderboard ids to remove: {len(remove_ids)}")
print(f"Example public leaderboard ids: {list(remove_ids)[:5]}")

# List of target file names for sequence-only CSV files.
# target_ls = ["all_random_seqs", "challenging_seqs", "high_exp_seqs", "low_exp_seqs", "yeast_seqs"]
target_ls = ["all_random_seqs", "challenging_seqs"]

for target in target_ls:
    print(f"Processing target: {target}")
    input_csv = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/{target}.csv"
    output_tsv_original = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_fix.csv"
    output_tsv_reversed = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_fix_rev.csv"
    output_sample_ids = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_sample_ids.tsv"

    # Sample rows from the CSV file (removing rows with public leaderboard indices).
    sampled_rows = sample_csv_rows(input_csv, sample_size=1000, remove_ids=remove_ids)
    
    # Write the output files.
    write_original_sequences(sampled_rows, output_tsv_original)
    write_reverse_complement_sequences(sampled_rows, output_tsv_reversed)
    write_sampled_row_ids(sampled_rows, output_sample_ids)
