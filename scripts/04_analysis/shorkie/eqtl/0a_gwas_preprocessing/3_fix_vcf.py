import pandas as pd
import os
import argparse

# Mapping dictionary from chromosome names to Roman numerals
chromosome_map = {
    'chromosome1': 'I', 'chromosome2': 'II', 'chromosome3': 'III',
    'chromosome4': 'IV', 'chromosome5': 'V', 'chromosome6': 'VI',
    'chromosome7': 'VII', 'chromosome8': 'VIII', 'chromosome9': 'IX',
    'chromosome10': 'X', 'chromosome11': 'XI', 'chromosome12': 'XII',
    'chromosome13': 'XIII', 'chromosome14': 'XIV', 'chromosome15': 'XV',
    'chromosome16': 'XVI'
}

def main():
    parser = argparse.ArgumentParser(description="Fix VCF chromosome names")
    parser.add_argument("--input_dir", required=True, help="Directory containing intersected_data_CIS.vcf etc.")
    parser.add_argument("--output_dir", required=True, help="Directory to save updated VCFs")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for distance_type in ["CIS", "TRANS"]:  
        # Define file paths
        intersected_tsv_file = os.path.join(input_dir, f"intersected_data_{distance_type}.vcf")
        output_vcf_file = os.path.join(output_dir, f"updated_intersected_data_{distance_type}.vcf")
        
        if not os.path.exists(intersected_tsv_file):
            continue
    # Open the VCF file and read the headers
    with open(intersected_tsv_file, 'r') as file:
        headers = []
        for line in file:
            if line.startswith("##"):
                headers.append(line.strip())
            elif line.startswith("#CHROM"):
                break

    # Read the VCF file, skipping the metadata lines (lines starting with '##')
    vcf_df = pd.read_csv(intersected_tsv_file, sep='\t', comment='#', header=None)
    # Assuming the VCF columns are in standard order
    vcf_df.columns = [
        '#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'
    ] + [f'SAMPLE{i+1}' for i in range(vcf_df.shape[1] - 9)]

    # Map the column values using the dictionary
    vcf_df['#CHROM'] = vcf_df['#CHROM'].map(chromosome_map)
    # print("vcf_df['#CHROM']: ", vcf_df['#CHROM'])    
    vcf_df.sort_values(by=['#CHROM', 'POS'], inplace=True)

        # Save the modified and sorted VCF file with headers
        with open(output_vcf_file, 'w') as file:
            # Write the headers
            for header in headers:
                file.write(header + '\n')
            # Write the sorted data without the index column
            vcf_df.to_csv(file, sep='\t', index=False)

if __name__ == "__main__":
    main()