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

def sample_csv_rows(input_file, sample_size=1000, remove_ids=[]):
    """
    Read the CSV file and store each row along with its original row number.
    Then remove rows whose original row numbers are in remove_ids and sample sample_size rows (if there are more than sample_size).
    Returns a list of sampled rows.
    """
    # print("remove_ids: ", remove_ids)
    with open(input_file, 'r') as fin:
        reader = csv.DictReader(fin, delimiter=',')
        rows = []
        # Annotate each row with its original row number (starting at 1)
        for i, row in enumerate(reader, start=1):
            if int(row['alt_pos']) in remove_ids or int(row['ref_pos']) in remove_ids:
                print("row", row, "; i", i)
                print("row['alt_pos']: ", row['alt_pos'])
                print("row['ref_pos']: ", row['ref_pos'])
            else:
                row['_orig_row'] = i
                rows.append(row)
        
        # Perform sampling after reading all rows
        if len(rows) > sample_size:
            sampled_rows = random.sample(rows, sample_size)
        else:
            sampled_rows = rows
    return sampled_rows

def write_original_sequences(sampled_rows, output_file):
    """
    Write a TSV file with columns:
    id, alt_sequence, ref_sequence, alt_exp, ref_exp
    using the sampled rows with sequences in their original orientation.
    """
    with open(output_file, 'w') as fout:
        fout.write("id\talt_sequence\tref_sequence\talt_exp\tref_exp\n")
        for counter, row in enumerate(sampled_rows, start=1):
            alt_seq = row['alt_sequence']
            ref_seq = row['ref_sequence']
            fout.write(f"seq{counter}\t{alt_seq}\t{ref_seq}\n")
            # alt_exp = row['alt_exp']
            # ref_exp = row['ref_exp']
            # fout.write(f"seq{counter}\t{alt_seq}\t{ref_seq}\t{alt_exp}\t{ref_exp}\n")

def write_reverse_complement_sequences(sampled_rows, output_file):
    """
    Write a TSV file with columns:
    id, alt_sequence, ref_sequence, alt_exp, ref_exp
    where the sequences are reverse complemented.
    """
    with open(output_file, 'w') as fout:
        fout.write("id\talt_sequence\tref_sequence\talt_exp\tref_exp\n")
        for counter, row in enumerate(sampled_rows, start=1):
            alt_seq = reverse_complement(row['alt_sequence'])
            ref_seq = reverse_complement(row['ref_sequence'])
            fout.write(f"seq{counter}\t{alt_seq}\t{ref_seq}\n")
            # alt_exp = row['alt_exp']
            # ref_exp = row['ref_exp']
            # fout.write(f"seq{counter}\t{alt_seq}\t{ref_seq}\t{alt_exp}\t{ref_exp}\n")

def write_sampled_row_ids(sampled_rows, output_file):
    """
    Write a file that maps the new sample IDs to the original CSV row numbers.
    This helps you track which original rows were extracted.
    """
    with open(output_file, 'w') as fout:
        fout.write("sample_id\toriginal_row_id\n")
        for counter, row in enumerate(sampled_rows, start=1):
            orig_id = row['_orig_row']
            fout.write(f"seq{counter}\t{orig_id}\n")

# Load public leaderboard ids
with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/high_exp_indices.json', 'r') as f:
    public_high = [int(indice) for indice in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/low_exp_indices.json', 'r') as f:
    public_low = [int(indice) for indice in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/yeast_exp_indices.json', 'r') as f:
    public_yeast = [int(indice) for indice in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/random_exp_indices.json', 'r') as f:
    public_random = [int(indice) for indice in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/challenging_exp_indices.json', 'r') as f:
    public_challenging = [int(indice) for indice in list(json.load(f).keys())]
    
with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/SNVs_exp_indices.json', 'r') as f:
    public_SNVs = [(int(indice.split(',')[0]), int(indice.split(',')[1])) for indice in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/motif_perturbation_exp_indices.json', 'r') as f:
    public_motif_perturbation = [(int(indice.split(',')[0]), int(indice.split(',')[1])) for indice in list(json.load(f).keys())]

with open('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/public_leaderboard_ids/motif_tiling_exp_indices.json', 'r') as f:
    public_motif_tiling = [(int(indice.split(',')[0]), int(indice.split(',')[1])) for indice in list(json.load(f).keys())]

# print("public_SNVs: ", public_SNVs)

# public_high + public_low + public_yeast + public_random + public_challenging + 
# Combine all public leaderboard ids into a single set
remove_ids = set([item[0] for item in public_SNVs] + 
                 [item[0] for item in public_motif_perturbation] + 
                 [item[0] for item in public_motif_tiling])

print(f"Number of public leaderboard ids: {len(remove_ids)}")
print(f"Example public leaderboard ids: {list(remove_ids)[:5]}")
# List of target file names.
targets = ["all_SNVs_seqs", "motif_perturbation", "motif_tiling_seqs"]

for target in targets:
    print(f"Processing target: {target}")
    input_csv = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/{target}.csv"
    output_tsv_original = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_fix.csv"
    output_tsv_reversed = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_fix_rev.csv"
    output_sample_ids = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_sample_ids.tsv"

    # Sample rows once from the CSV, removing the public leaderboard ids
    sampled_rows = sample_csv_rows(input_csv, sample_size=1000, remove_ids=remove_ids)
    
    # Write the output files.
    write_original_sequences(sampled_rows, output_tsv_original)
    write_reverse_complement_sequences(sampled_rows, output_tsv_reversed)
    write_sampled_row_ids(sampled_rows, output_sample_ids)
