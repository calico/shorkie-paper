import pandas as pd
import argparse

def generate_report(input):
    # Read the input data
    df = pd.read_csv(input, sep=',', header=0)
    
    # Calculate the percentages of coding and non-coding nucleotides
    df['total_nucleotides'] = df['nucleotides_coding'] + df['nucleotides_non_coding']
    df['coding_percentage'] = df['nucleotides_coding'] / df['total_nucleotides']
    df['non_coding_percentage'] = df['nucleotides_non_coding'] / df['total_nucleotides']
    
    # Calculate overall statistics
    overall_avg_overlap = df['total_overlap'].mean()
    overall_median_overlap = df['total_overlap'].median()
    overall_avg_coding_ratio = df['coding_percentage'].mean()

    # Prepare the report
    report_lines = [
        f"{overall_avg_overlap:.2f}",
        f"{overall_median_overlap:.2f}",
        f"{overall_avg_coding_ratio:.2f}"
    ]
    
    # Print the report lines so they can be captured in bash
    print(",".join(report_lines))
    return report_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a summary report of overlaps and coding ratios.')
    parser.add_argument('--input', type=str, help='Path to the input file', default='GCA_000146045_2_ovp.txt')

    args = parser.parse_args()
    generate_report(args.input)
