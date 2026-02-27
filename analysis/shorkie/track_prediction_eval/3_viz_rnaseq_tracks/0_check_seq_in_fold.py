import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Check sequence overlaps in sequence folds.")
parser.add_argument("--root_dir", default="../../..", help="Root directory pointing to Yeast_ML")
args = parser.parse_args()

# Load the data into a pandas DataFrame
sequence_fold_bed = f"{args.root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_model/data/sequences.bed"
df = pd.read_csv(sequence_fold_bed, sep="\t", header=None, names=["chrom", "start", "end", "fold"])

# Convert the data into a pandas DataFrame
from io import StringIO


# Define the target sequence
target_chrom = "chrVII"
target_start = 495207
target_end = 509543
# chrVII:489,580-507,696

# target_chrom = "chrIII"
# target_start = 100051
# target_end = 114387

# chrIII:100,051-114,387
# 497,046-498,113

# Function to check overlap between two intervals
def overlap(row):
    # Check if the chroms match and if the intervals overlap
    return (row["chrom"] == target_chrom) and not (row["end"] < target_start or row["start"] > target_end)

# Filter rows that overlap with the target sequence
overlapping_rows = df[df.apply(overlap, axis=1)]

print(overlapping_rows)

# # Display the overlapping rows
# import ace_tools as tools; tools.display_dataframe_to_user(name="Overlapping Rows", dataframe=overlapping_rows)

# print(overlapping_rows)
