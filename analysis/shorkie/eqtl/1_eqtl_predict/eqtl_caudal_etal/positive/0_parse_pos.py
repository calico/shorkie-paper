#!/usr/bin/env python3
import argparse
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        description="Extract GENES, CENTER_POS, POS, and ALTS arrays from a TSV"
    )
    # to take the TSV from the command line again, uncomment:
    # parser.add_argument("tsv", help="input TSV file")
    # args = parser.parse_args()
    # tsv = args.tsv

    tsv = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/selected_eQTL/intersected_data_CIS.tsv"
    df = pd.read_csv(tsv, sep="\t", dtype=str)

    # parse numeric positions
    center_series = pd.to_numeric(df["ChrPos"], errors="coerce").astype("Int64")
    pos_series    = center_series.copy()

    genes_out  = []
    center_out = []
    pos_out    = []
    alts_out   = []

    for idx, raw_alt in df["Alternate"].fillna("").iteritems():
        # split on commas, strip whitespace
        alleles = [a.strip() for a in raw_alt.split(",") if a.strip()]
        # filter only the single‐nucleotide ones
        single_nts = [a for a in alleles if len(a) == 1]
        # if none, skip this row
        if not single_nts:
            continue

        # for each single‐nt allele, make a separate entry
        for alt in single_nts:
            genes_out.append(df.at[idx, "Pheno"])
            center_out.append(center_series.iloc[idx])
            pos_out.append(pos_series.iloc[idx])
            alts_out.append(alt)

    print(f"Extracted {len(genes_out)} total single‐nt entries.")
    print(f"genes_out: {len(genes_out)} ...")
    print(f"center_out: {len(center_out)} ...")
    print(f"pos_out: {len(pos_out)} ...")
    print(f"alts_out: {len(alts_out)} ...")

    def bash_array(name, items):
        items = [str(x) for x in items]
        return f'{name}=({" ".join(items)})'

    outfile = "results/parsed_pos_arrays.sh"
    with open(outfile, "w") as f:
        f.write(bash_array("GENES",      genes_out)  + "\n")
        f.write(bash_array("CENTER_POS", center_out) + "\n")
        f.write(bash_array("POS",         pos_out)   + "\n")
        f.write(bash_array("ALTS",        alts_out)  + "\n")

    print(f"Wrote arrays to {outfile}")

if __name__ == "__main__":
    main()
