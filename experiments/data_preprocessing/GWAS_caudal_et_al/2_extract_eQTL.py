import pandas as pd
import pysam
import os 

def read_gvcf_to_df(file_path):
    records = []
    # Open the GVCF file with pysam
    with pysam.VariantFile(file_path) as gvcf:
        print(f"Reading GVCF file: {file_path}")
        counter = 0
        for record in gvcf:
            variant_info = {
                "Chr": record.chrom,
                "ChrPos": int(record.pos),      # Ensure ChrPos is int for compatibility
                "Reference": record.ref,
                "Alternate": ','.join(record.alts) if record.alts else None,
                "Quality": record.qual
            }
            records.append(variant_info)
            counter += 1
            # if counter > 30000:
            #     break
    gvcf_df = pd.DataFrame(records)
    # Explicitly convert ChrPos to int64
    gvcf_df["ChrPos"] = gvcf_df["ChrPos"].astype("int64")
    return gvcf_df


def write_df_to_vcf(df, output_file, template_file):
    # Open a template GVCF file to copy its header
    with pysam.VariantFile(template_file) as template:
        # Get the list of valid chromosomes/contigs from the template header
        valid_contigs = set(template.header.contigs)
        # Filter the DataFrame to include only valid chromosomes
        df_filtered = df[df["Chr"].isin(valid_contigs)]
        with pysam.VariantFile(output_file, "w", header=template.header) as vcf_out:
            for _, row in df_filtered.iterrows():
                # Check for valid Reference and Alternate alleles
                if pd.isna(row["Reference"]) or pd.isna(row["Alternate"]):
                    print(f"Skipping invalid entry at {row['Chr']}:{row['ChrPos']}")
                    continue  # Skip rows with missing allele information
                # Create a new record
                record = vcf_out.new_record(
                    contig=row["Chr"],
                    start=row["ChrPos"] - 1,  # VCF is 0-based
                    stop=row["ChrPos"],
                    alleles=(str(row["Reference"]), str(row["Alternate"])),  # Convert to strings
                    qual=row.get("Quality", None)
                )
                vcf_out.write(record)
    print(f"Saved VCF file: {output_file}")


gvcf_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/1011Matrix.gvcf"
# Read GVCF file and convert to DataFrame
gvcf_df = read_gvcf_to_df(gvcf_file)
print("GVCF DataFrame:")
print(gvcf_df.head())
print(gvcf_df.tail())
print("\tlen(gvcf_df): ", len(gvcf_df))   

output_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/selected_eQTL/"
os.makedirs(output_dir, exist_ok=True)
for distance_type in ["CIS", "TRANS"]:  
    # Define file paths
    GWAS_output_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/GWAS/"
    eqtl_file = f"{GWAS_output_dir}GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207_cleaned_{distance_type}.tab"

    # Read the eQTL data
    eqtl_df = pd.read_csv(eqtl_file, sep='\t')
    # Ensure Chr and ChrPos in eQTL DataFrame are of the correct type
    eqtl_df["ChrPos"] = pd.to_numeric(eqtl_df["ChrPos"],
            errors='coerce').fillna(0).astype("int64") 
    eqtl_df["Chr"] = "chromosome" + eqtl_df["Chr"].astype(str)
    eqtl_df["Chr"] = eqtl_df["Chr"].astype(str)
    print("eQTL DataFrame:")

    print("\tlen(eqtl_df): ", len(eqtl_df))   

    # Write intersected and unique data to VCF files
    intersected_tsv_file = f"{output_dir}intersected_data_{distance_type}.tsv"
    only_eqtl_tsv_file = f"{output_dir}only_eqtl_data_{distance_type}.tsv"

    # Intersect GVCF DataFrame with eQTL DataFrame on Chr and ChrPos
    merged_df = pd.merge(eqtl_df, gvcf_df, how="inner", on=["Chr", "ChrPos"])
    print("Intersected DataFrame (shared between eQTL and GVCF):")
    print(merged_df.head()) 
    print("\tlen(merged_df):", len(merged_df))
    merged_df.to_csv(intersected_tsv_file, sep='\t', index=False)

    # Find entries only in eQTL DataFrame
    only_eqtl_df = eqtl_df.merge(gvcf_df, on=["Chr", "ChrPos"], how="left", indicator=True) 
    only_eqtl_df = only_eqtl_df[only_eqtl_df["_merge"]== "left_only"].drop(columns="_merge") 
    print("Entries only in eQTL DataFrame (not in GVCF):")
    print(only_eqtl_df.head())
    only_eqtl_df.to_csv(only_eqtl_tsv_file, sep='\t', index=False)

    # Write intersected and unique data to VCF files
    intersected_vcf_file = f"{output_dir}intersected_data_{distance_type}.vcf"
    only_eqtl_vcf_file = f"{output_dir}only_eqtl_data_{distance_type}.vcf"
    write_df_to_vcf(merged_df, intersected_vcf_file, gvcf_file)
    write_df_to_vcf(only_eqtl_df, only_eqtl_vcf_file, gvcf_file)
