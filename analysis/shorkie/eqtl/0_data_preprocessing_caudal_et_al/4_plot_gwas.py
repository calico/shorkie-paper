import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and filter data ---
# Load the GWAS data (modify the file path as needed)
gwas_df = pd.read_csv('/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207.tab')

gwas_df = gwas_df[gwas_df['ld_mask'] != "masked"]
print(gwas_df.head())
print(len(gwas_df))
# Filter out missing or infinite p-values
gwas_df = gwas_df.dropna(subset=['PValue'])
gwas_df = gwas_df[gwas_df['PValue'] > 0]

# compute –log10(p)
gwas_df['minus_log10_p'] = -np.log10(gwas_df['PValue'])

# ensure chromosome column is string
gwas_df['Chr'] = gwas_df['Chr'].astype(str)


# --- 2. Define chromosome ordering and offsets ---
# numeric chromosomes first, then any others
chromosomes = sorted(
    gwas_df['Chr'].unique(),
    key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'))
)

# get maximum base‐pair position per chromosome in that order
chr_max = gwas_df.groupby('Chr')['ChrPos'].max().reindex(chromosomes)

# compute cumulative offset for each chromosome
offset = chr_max.cumsum().shift(fill_value=0).to_dict()

# add genome‐wide position
gwas_df['cum_pos'] = (
    gwas_df['ChrPos'] + gwas_df['Chr'].map(offset)
)


# --- 3. Plotting ---
# fig, ax = plt.subplots(figsize=(12, 6))
fig, ax = plt.subplots(figsize=(8, 4))

# alternating colors by chromosome
colors = ['#1f77b4', '#ff7f0e']

for i, chr_name in enumerate(chromosomes):
    sub = gwas_df[gwas_df['Chr'] == chr_name]
    ax.scatter(
        sub['cum_pos'],
        sub['minus_log10_p'],
        c=colors[i % 2],
        s=3,
        label=chr_name
    )

# genome‐wide significance line (p = 5e-8)
ax.axhline(-np.log10(5e-8), color='red', linestyle='--', linewidth=1)

# set x‐ticks at the midpoint of each chromosome
tick_locs = [
    (offset[chr_name] + chr_max[chr_name] / 2)
    for chr_name in chromosomes
]
ax.set_xticks(tick_locs)
ax.set_xticklabels(chromosomes, rotation=45, ha='right')

# labels and title
ax.set_xlabel('Chromosome')
ax.set_ylabel('-log10(P-value)')
ax.set_title('Manhattan Plot')

plt.tight_layout()
plt.savefig('manhattan_plot.png', dpi=300)
plt.show()
