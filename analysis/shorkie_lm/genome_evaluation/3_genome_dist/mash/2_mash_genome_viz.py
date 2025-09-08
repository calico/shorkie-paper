import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Define a function to parse the file and extract the last third column value
def parse_file_content(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                # print("parts: ", parts)
                if len(parts) >= 3:
                    try:
                        return float(parts[-3])
                    except ValueError:
                        pass
    return None

def main(data_type):
    # Directory containing the fasta files
    REF_FASTA = "/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"
    REF_GTF = "/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_r64_gtf/gtf/GCA_000146045_2.59.gtf"
    ref_base_name = os.path.basename(REF_FASTA).replace(".cleaned.fasta", "")

    FASTA_DIR = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta"
    GTF_DIR = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/gtf"
    OUTPUT_DIR = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/{data_type}/genome_dist/{data_type}/mash"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_targets = []
    # 1. Collect GTF and FASTA file paths
    for fasta_file in os.listdir(FASTA_DIR):
        if fasta_file.endswith(".cleaned.fasta"):
            base_name = fasta_file.replace(".cleaned.fasta", "")
            gtf_file = os.path.join(GTF_DIR, f"{base_name}.59.gtf")

            print("Reference fasta: ", REF_FASTA)
            print("Reference gtf  : ", REF_GTF)

            target_fasta = os.path.join(FASTA_DIR, fasta_file)
            print("Target fasta: ", target_fasta)
            print("Target gtf  : ", gtf_file)

            output_targets.append(base_name)
    print("output_targets: ", output_targets)

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
    df_sorted = df.sort_values(by='Value', ascending=True)

    # Plotting the results
    plt.figure(figsize=(20, 6))
    plt.bar(df_sorted['File'], df_sorted['Value'], color='skyblue')
    plt.xlabel('Target Genomes')
    plt.ylabel('Mash Distance Score')
    plt.title('Mash Distance Score for Yeast Reference Genome (R64) and Target Genomes')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"{OUTPUT_DIR}/{data_type}mash_viz.png", dpi=300)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_type>")
        sys.exit(1)
    data_type = sys.argv[1]
    main(data_type)
