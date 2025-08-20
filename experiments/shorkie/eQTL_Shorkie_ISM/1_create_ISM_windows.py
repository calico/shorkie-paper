#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a variant TSV into an 80bp-window BED centered on each variant."
    )
    p.add_argument("input_tsv",
                   help="Path to TSV of variants (must have columns 'variant_id', 'chromosome', 'position').")
    p.add_argument("output_bed",
                   help="Path to write BED file.")
    p.add_argument("--window", type=int, default=80,
                   help="Total window size in bp (must be even). Default: 80")
    p.add_argument("--id-col", default="SNP",
                   help="TSV column for variant ID. Default: %(default)s")
    p.add_argument("--chr-col", default="Chr",
                   help="TSV column for chromosome. Default: %(default)s")
    p.add_argument("--pos-col", default="ChrPos",
                   help="TSV column for variant position. Default: %(default)s")
    return p.parse_args()

def main():
    args = parse_args()

    if args.window % 2 != 0:
        sys.exit("Error: --window must be an even integer.")

    half = args.window // 2

    # 1) Read the TSV
    df = pd.read_csv(args.input_tsv, sep="\t", comment="#", dtype={args.chr_col: str})

    # Verify required columns
    for col in (args.id_col, args.chr_col, args.pos_col):
        if col not in df.columns:
            sys.exit(f"Error: column '{col}' not found in {args.input_tsv}")

    # Map chromosome names from 'chromosome1'..'chromosome16' to 'chrI'..'chrXVI'
    roman_numerals = [
        "I","II","III","IV","V","VI","VII","VIII","IX","X",
        "XI","XII","XIII","XIV","XV","XVI"
    ]
    chr_map = {f"chromosome{idx+1}": f"chr{roman}" for idx, roman in enumerate(roman_numerals)}
    # Apply mapping; leave others unchanged
    df[args.chr_col] = df[args.chr_col].map(chr_map).fillna(df[args.chr_col])

    # 3) Compute bed start/end
    #    If your TSV 'position' is 1-based, you may want to subtract 1 here:
    df["start"] = df[args.pos_col] - half
    df["end"]   = df[args.pos_col] + half

    #  Avoid negative starts
    df["start"] = df["start"].clip(lower=0).astype(int)
    df["end"]   = df["end"].astype(int)

    # 4) Select and write BED (no header, zero-based half-open)
    out_df = df[[args.chr_col, "start", "end", args.id_col]]
    out_df.to_csv(
        args.output_bed,
        sep="\t",
        header=False,
        index=False
    )

    print(f"Wrote {len(out_df)} windows to {args.output_bed}")

if __name__ == "__main__":
    main()
