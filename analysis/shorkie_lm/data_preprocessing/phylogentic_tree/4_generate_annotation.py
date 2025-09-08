#!/usr/bin/env python3
import csv
import sys

# # Usage: python filter_saccharomycetales.py input.csv output.txt
# if len(sys.argv) < 3:
#     print("Usage: {} input.csv output.txt".format(sys.argv[0]))
#     sys.exit(1)

input_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/species_saccharomycetales_gtf.cleaned.csv"
output_file = "species_saccharomycetales_gtf.txt"

# Open the CSV file and read it as a dictionary.
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    # Filter rows where the "Classification" column is exactly "Saccharomycetales"
    species_list = [
        row["Name"].strip()
        for row in reader
        if row["Classification"].strip() == "Saccharomycetales"
    ]

# Write the filtered species names (one per line) to the output file.
with open(output_file, 'w', encoding='utf-8') as outfile:
    for species in species_list:
        outfile.write(species + "\n")

print("Filtered species names written to", output_file)




input_file ="species_saccharomycetales_gtf.txt"
output_file = "annotations.txt"

# Read species names from the input file (one per line)
with open(input_file, 'r', encoding='utf-8') as infile:
    species = [line.strip() for line in infile if line.strip()]

# Optionally deduplicate the list while preserving order
seen = set()
unique_species = []
for sp in species:
    if sp not in seen:
        seen.add(sp)
        unique_species.append(sp)

# Write the iTOL annotation dataset file
with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write("DATASET_COLORSTRIP\n")
    outfile.write("SEPARATOR COMMA\n")
    outfile.write("DATASET_LABEL,Saccharomycetales Highlight\n")
    outfile.write("COLOR,#00ff00\n")
    outfile.write("DATA\n")
    for sp in unique_species:
        outfile.write(f"{sp},#00ff00\n")

print("Annotation dataset file written to", output_file)
