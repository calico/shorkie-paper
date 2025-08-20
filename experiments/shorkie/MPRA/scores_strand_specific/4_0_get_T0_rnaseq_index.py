'''
Script to find positions of T0 RNA-Seq tracks in the full RNA-Seq track sheet.

Usage:
    python find_t0_indices.py <all_tracks_file> <t0_tracks_file> [output_file]

Reads two tab-delimited files, identifies rows of T0 tracks in the full sheet by 'identifier',
and outputs original index values and row positions (0-based) either to stdout or to an output_file.
'''
import sys
import pandas as pd

def find_indices(all_file, t0_file):
    # Read sheets as strings to preserve all columns
    df_all = pd.read_csv(all_file, sep='\t', dtype=str)
    df_t0 = pd.read_csv(t0_file, sep='\t', dtype=str)

    # Validate presence of identifier column
    if 'identifier' not in df_all.columns or 'identifier' not in df_t0.columns:
        raise ValueError("Both files must contain an 'identifier' column.")

    # Convert original index column to numeric if it exists
    if 'index' in df_all.columns:
        df_all['index'] = pd.to_numeric(df_all['index'], errors='coerce')

    # Filter full sheet for T0 identifiers
    mask = df_all['identifier'].isin(df_t0['identifier'])
    common = df_all[mask].copy()

    # Extract positions and, if available, original index values
    positions = common.index.tolist()
    result = pd.DataFrame({
        'identifier': common['identifier'],
        'position_0based': positions
    })
    if 'index' in common.columns:
        result['original_index'] = common['index'].astype(int).tolist()
    return result


def main():
    # if len(sys.argv) < 3:
    #     print("Usage: python find_t0_indices.py <all_tracks_file> <t0_tracks_file> [output_file]")
    #     sys.exit(1)

    # all_file = sys.argv[1]
    # t0_file = sys.argv[2]
    # output_file = sys.argv[3] if len(sys.argv) >= 4 else None
    all_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_all_RNA-Seq_strand.txt"
    t0_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq_T0.txt"
    output_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/t0_indices.tsv"

    res = find_indices(all_file, t0_file)
    if output_file:
        res.to_csv(output_file, sep='\t', index=False)
        print(f"Wrote results to {output_file}")
    else:
        # Default behavior: write to 't0_indices.tsv'
        default_path = 't0_indices.tsv'
        res.to_csv(default_path, sep='\t', index=False)
        print(f"Wrote results to {default_path}")

if __name__ == '__main__':
    main()
