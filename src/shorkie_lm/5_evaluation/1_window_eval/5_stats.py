import pandas as pd
import argparse

def generate_report(input, output):
    # Read the input data
    df = pd.read_csv(input, sep=',', header=0)
    
    # Debug: Print the column names and the first few rows
    print("Columns in the dataframe:", df.columns)
    print("First few rows of the dataframe:\n", df.head())
    
    # Ensure column names have no leading or trailing spaces
    df.columns = df.columns.str.strip()
    
    # Check if 'chromosome' column exists
    if 'chrom' not in df.columns:
        raise KeyError("The input file does not contain a 'chrom' column.")
    
    # Calculate the percentages of coding and non-coding nucleotides
    df['total_nucleotides'] = df['nucleotides_coding'] + df['nucleotides_non_coding']
    df['coding_percentage'] = df['nucleotides_coding'] / df['total_nucleotides']
    df['non_coding_percentage'] = df['nucleotides_non_coding'] / df['total_nucleotides']
    
    # Calculate overall statistics
    overall_avg_overlap = df['total_overlap'].mean()
    overall_median_overlap = df['total_overlap'].median()
    overall_avg_coding_ratio = df['coding_percentage'].mean()

    # Prepare the overall report
    report_lines = [
        f"Overall Average Overlapped Gene Count: {overall_avg_overlap:.2f}",
        f"Overall Median Overlapped Gene Count: {overall_median_overlap:.2f}",
        f"Overall Average Coding Ratio: {overall_avg_coding_ratio:.2f}\n"
    ]
    
    # Calculate statistics for each chromosome
    chromosome_stats = df.groupby('chrom').agg(
        avg_overlap=('total_overlap', 'mean'),
        median_overlap=('total_overlap', 'median'),
        avg_coding_ratio=('coding_percentage', 'mean')
    ).reset_index()
    
    # Add chromosome-specific stats to the report
    for _, row in chromosome_stats.iterrows():
        report_lines.append(
            f"Chromosome {row['chrom']}:\n"
            f"  Average Overlapped Gene Count: {row['avg_overlap']:.2f}\n"
            f"  Median Overlapped Gene Count: {row['median_overlap']:.2f}\n"
            f"  Average Coding Ratio: {row['avg_coding_ratio']:.2f}\n"
        )
    
    # Write the report to the output file
    with open(output, 'w') as f:
        for line in report_lines:
            f.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a summary report of overlaps and coding ratios.')
    parser.add_argument('--input', type=str, help='Path to the input file', default='GCA_000146045_2_ovp.txt')
    parser.add_argument('--output', type=str, help='Path to the output file (default: report.txt)', default='report.txt')
    args = parser.parse_args()
    generate_report(args.input, args.output)
