#!/usr/bin/env python3
import csv
import sys

def get_taxon_ids(filename):
    taxon_ids = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract the value from the "Taxon ID" column
            taxon_id = row.get("Taxon ID")
            if taxon_id:
                taxon_ids.append(taxon_id)
    return taxon_ids

if __name__ == "__main__":
    filename = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/species_fungi_1385_gtf.cleaned.csv"
    taxon_ids = get_taxon_ids(filename)
    # Print each Taxon ID on its own line
    for taxon in taxon_ids:
        print(taxon)

