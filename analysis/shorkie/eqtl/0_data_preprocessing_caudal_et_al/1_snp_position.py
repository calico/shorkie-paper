import pandas as pd
import matplotlib.pyplot as plt
import os

gwas_input = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207.tab'
output_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/GWAS/'
gwas_output = f'{output_dir}GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207_cleaned.tab'
gwas_CIS_output = f'{output_dir}GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207_cleaned_CIS.tab'
gwas_TRANS_output = f'{output_dir}GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207_cleaned_TRANS.tab'

os.makedirs(output_dir)
# Load the GWAS data (modify the file path as needed)
gwas_df = pd.read_csv(gwas_input)

gwas_df = gwas_df[gwas_df['ld_mask'] != "masked"]
print(gwas_df.head())
print(len(gwas_df))
gwas_df.to_csv(gwas_output, sep='\t', index=False)


# Normalize the gene positions within their respective chromosomes
chromosome_max_pos_gene = gwas_df.groupby('Pheno_chr')['Pheno_pos'].max()
gwas_df['Gene_relative_position'] = gwas_df.apply(lambda row: row['Pheno_pos'] / chromosome_max_pos_gene[row['Pheno_chr']], axis=1)

# Normalize the SNP positions within their respective chromosomes
chromosome_max_pos = gwas_df.groupby('Chr')['ChrPos'].max()
gwas_df['SNP_relative_position'] = gwas_df.apply(lambda row: row['ChrPos'] / chromosome_max_pos[row['Chr']], axis=1)

# Define the local threshold (25 kb each side of the gene position)
local_threshold = 25000

# Check if an eQTL is local
gwas_df['is_local'] = gwas_df.apply(lambda row: abs(row['ChrPos'] - row['Pheno_pos']) <= local_threshold if row['Chr'] == row['Pheno_chr'] else False, axis=1)

# Split the data into SNV and CNV subtypes
snv_data = gwas_df[gwas_df['subtype'] == 'SNP']
cnv_data = gwas_df[gwas_df['subtype'] == 'CNV']

snv_data_CIS = snv_data[snv_data['type'] == 'CIS']
snv_data_TRANS = snv_data[snv_data['type'] == 'TRANS']


snv_data_CIS.to_csv(gwas_CIS_output, sep='\t', index=False)
snv_data_TRANS.to_csv(gwas_TRANS_output, sep='\t', index=False)

print("len full gwas_df: ", len(gwas_df)) 
print("len cnv_data: ", len(cnv_data))
print("len snv_data: ", len(snv_data))
print("   - len snv_data_CIS: ", len(snv_data_CIS))
print("   - len snv_data_TRANS: ", len(snv_data_TRANS))
