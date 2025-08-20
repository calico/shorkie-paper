import csv
import random

def reverse_complement(seq):
    """
    Return the reverse complement of a DNA sequence.
    Uses str.maketrans for efficient translation.
    """
    complement_table = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement_table)[::-1]

def sample_csv_rows(input_file, sample_size=1000):
    """
    Read the CSV file and store each row along with its original row number.
    Then randomly sample sample_size rows (if there are more than sample_size).
    Returns a list of sampled rows.
    """
    with open(input_file, 'r') as fin:
        reader = csv.DictReader(fin, delimiter=',')
        rows = []
        # Annotate each row with its original row number (starting at 1)
        for i, row in enumerate(reader, start=1):
            row['_orig_row'] = i
            rows.append(row)
        
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

# List of target file names.
# targets = ["all_SNVs_seqs", "motif_perturbation", "motif_tiling_seqs"]
targets = ["motif_perturbation", "motif_tiling_seqs"]

for target in targets:
    input_csv = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/{target}.csv"
    output_tsv_original = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_fix.csv"
    output_tsv_reversed = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_fix_rev.csv"
    output_sample_ids = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/test_subset_ids/fix/{target}_sample_ids.tsv"

    # Sample rows once from the CSV.
    sampled_rows = sample_csv_rows(input_csv, sample_size=1000)
    
    # Write the output files.
    write_original_sequences(sampled_rows, output_tsv_original)
    write_reverse_complement_sequences(sampled_rows, output_tsv_reversed)
    write_sampled_row_ids(sampled_rows, output_sample_ids)
