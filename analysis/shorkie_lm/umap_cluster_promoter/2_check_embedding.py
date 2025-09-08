import h5py
import numpy as np
import pandas as pd
import pyranges as pr

##############################
# 1) Load Metadata from H5
##############################
h5_path = "embeddings_LM_sequence/embeddings_chrI.h5"
with h5py.File(h5_path, "r") as h5f:
    meta_array = np.array(h5f['metadata'])
    # Decode byte strings to UTF-8
    meta_str = np.char.decode(meta_array.astype(np.bytes_), 'utf-8')

# Create a DataFrame: columns [chrom, start, end, strand, feature, gene_id]
meta_df = pd.DataFrame(meta_str, columns=["chrom","start","end","strand","feature","gene_id"])
print("Unique features in metadata:")
print(meta_df['feature'].unique())
print(f"Metadata shape: {meta_df.shape}")

##############################
# 2) Load GTF and Extract Biotype
##############################
gtf_path = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf"
gtf_pr = pr.read_gtf(gtf_path)
gtf_df = gtf_pr.as_df()

# Check what columns are available:
print("\nColumns in GTF DataFrame:")
print(gtf_df.columns)

# We assume 'gene_biotype' is present. If not, you may need to parse 'Attributes'.
if "gene_biotype" not in gtf_df.columns:
    print("No 'gene_biotype' column found. You may need to parse the 'Attributes' field.")
    # e.g., if your GTF has something like: gene_id "XYZ"; gene_biotype "protein_coding"; ...
    # you'd parse gtf_df['Attributes'] strings to extract 'gene_biotype'.
    # For demonstration, let's assume it's already extracted.

# We'll keep only [gene_id, gene_biotype] (and drop duplicates if needed).
gene_info_df = gtf_df[["gene_id", "gene_biotype"]].drop_duplicates()

##############################
# 3) Merge on gene_id
##############################
merged_df = meta_df.merge(gene_info_df, on="gene_id", how="left")

# Now, 'merged_df' will have a new column 'gene_biotype'
print("\nAfter merging with gene_biotype:")
print(merged_df.head(10))
print("Unique biotypes found:")
print(merged_df["gene_biotype"].unique())

# You can filter out 'intergenic' or other features that do not have gene_biotype
# if you want only gene features:
gene_only = merged_df.dropna(subset=["gene_biotype"])
print(f"\nNumber of intervals with a known gene_biotype: {gene_only.shape[0]}")
