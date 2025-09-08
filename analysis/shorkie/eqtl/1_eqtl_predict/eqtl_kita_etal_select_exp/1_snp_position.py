import pandas as pd
import matplotlib.pyplot as plt
import os

root_dir='/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL_kita_etal'
eqtl_input = f'{root_dir}/pnas.1717421114.sd01.fix.txt'
output_dir = f'{root_dir}/fix/'
eqtl_output = f'{output_dir}pnas.1717421114.sd01_cleaned.txt'
# eqtl_CIS_output = f'{output_dir}pnas.1717421114.sd01_cleaned_CIS.tab'
# eqtl_TRANS_output = f'{output_dir}pnas.1717421114.sd01_cleaned_TRANS.tab'

os.makedirs(output_dir, exist_ok=True)
# Load the eqtl data (modify the file path as needed)
eqtl_df = pd.read_csv(eqtl_input, sep='\t', low_memory=False)

# eqtl_df = eqtl_df[eqtl_df['ld_mask'] != "masked"]
print(eqtl_df.head())
print(len(eqtl_df))

# Define the local threshold (25 kb each side of the gene position)
local_threshold = 8000

# Check if an eQTL is local
eqtl_df['distance_to_TSS'] = eqtl_df.apply(lambda row: abs(int(row['position']) - int(row['TSS'])), axis=1)
eqtl_df['is_local'] = eqtl_df.apply(lambda row: abs(int(row['position']) - int(row['TSS'])) <= local_threshold if row['chr'] == row['chr'] else False, axis=1)

eqtl_df.to_csv(eqtl_output, sep='\t', index=False)

print("len full eqtl_df: ", len(eqtl_df)) 
