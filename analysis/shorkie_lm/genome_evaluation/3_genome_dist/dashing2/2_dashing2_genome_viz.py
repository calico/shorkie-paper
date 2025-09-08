import os, sys
import pandas as pd
import matplotlib.pyplot as plt

# Sample data to mimic the file content and directory structure
def is_number(s):
    try:
        float(s)  # Check if it can be converted to a float
        if '.' in s or 'e' in s or 'E' in s:  # Check if it's explicitly a float
            return "float"
        else:
            return "int"
    except ValueError:
        return None


# Define a function to parse the file and extract the number
def parse_file_content(fname):
    with open(fname, 'r') as f:
        lines = file_content = f.read().splitlines()
        # lines = file_content.strip().split('\n')
        for line in lines:
            # print("line: ", line)
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                print("parts: ", parts)
                if len(parts) > 2 and (is_number(parts[2])):
                    return float(parts[2])
    return None


def main(data_type):
    # Directory containing the fasta files
    REF_FASTA = "/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"
    REF_GTF = "/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_r64_gtf/gtf/GCA_000146045_2.59.gtf"
    ref_base_name = os.path.basename(REF_FASTA).replace(".cleaned.fasta", "")

    FASTA_DIR = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta"
    GTF_DIR = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/gtf"
    OUTPUT_DIR = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/{data_type}/genome_dist/{data_type}/dashing2"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_files = []
    output_targets = []
    # 1. Collect GTF and FASTA file paths
    for fasta_file in os.listdir(FASTA_DIR):
        if fasta_file.endswith(".cleaned.fasta"):
            base_name = fasta_file.replace(".cleaned.fasta", "")
            gtf_file = os.path.join(GTF_DIR, f"{base_name}.59.gtf")
            target_fasta = os.path.join(FASTA_DIR, fasta_file)
            output_targets.append(base_name)
            
    # Visualize the results
    data = {}
    for output_target in output_targets:
        output_file = os.path.join(OUTPUT_DIR, f"{data_type}_{ref_base_name}_{output_target}.txt")
        number = parse_file_content(output_file)
        if number is not None:
            data[output_target] = number

    # Convert to DataFrame for visualization
    df = pd.DataFrame(list(data.items()), columns=['File', 'Value'])
    print("df: ", df)
    df_sorted = df.sort_values(by='Value', ascending=False)

    # Plotting the results
    plt.figure(figsize=(30, 6))
    plt.bar(df_sorted['File'], df_sorted['Value'], color='skyblue')
    plt.xlabel('Target Genomes')
    plt.ylabel('Dashing2 Similarity Score')
    plt.title('Dashing2 Similarity Score for Yeast Reference Genome (R64) and Target Genomes')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"{OUTPUT_DIR}/{data_type}dashing2_viz.png", dpi=300)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_type>")
        sys.exit(1)
    data_type = sys.argv[1]
    main(data_type)