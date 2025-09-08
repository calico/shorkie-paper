import pandas as pd
import matplotlib.pyplot as plt
import sys

data_type = sys.argv[1]

x_threshold = 80.0
y_threshold = 10.0

for target in ["test", "valid"]:
    alns = f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_type}/dataset_stats/dataset_similarity/minimap2/overlaps_ratio_{target}.txt"
    alns_df = pd.read_csv(alns, sep=' ', header=None)
    alns_df = alns_df[(alns_df[2].astype(float) > x_threshold) | (alns_df[3].astype(float) > y_threshold)]
    print("alns_df: ", alns_df)

    removed_dict = {}
    for i in range(0, len(alns_df)):
        # print("i: ", i)
        seq_id, species = alns_df.iloc[i][0].split('|') 
        chrom = seq_id.split(':')[0]
        # print("seq_id: ", seq_id, "species: ", species, "chrom: ", chrom)
        if species not in removed_dict:
            removed_dict[species] = {}
            removed_dict[species][chrom] = set()
            removed_dict[species][chrom].add(seq_id)
        else:
            if chrom not in removed_dict[species]:
                removed_dict[species][chrom] = set()
                removed_dict[species][chrom].add(seq_id)
            else:
                removed_dict[species][chrom].add(seq_id)

    for species, chroms in removed_dict.items():
        for chrom, seq_ids in chroms.items():
            if chrom == "chrXV" or chrom == "chrXVI":
                print(f"species: {species}, chrom: {chrom}, len(seq_ids): {len(seq_ids)}")
                # for seq_id in seq_ids:
                #     print(f"\tseq_id: {seq_id}")

    all_dict = {}
    train_bed = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{data_type}/sequences.bed"
    print("train_bed: ", train_bed)
    train_bed_df = pd.read_csv(train_bed, sep='\t', header=None)
    print("train_bed_df: ", train_bed_df)
    for i in range(0, len(train_bed_df)):
        species = train_bed_df.iloc[i][4]
        chrom = train_bed_df.iloc[i][0]
        seq_id = train_bed_df.iloc[i][0] + ":" + str(train_bed_df.iloc[i][1]) + "-" + str(train_bed_df.iloc[i][2]) 

        # if chrom == "chrXV" or chrom == "chrXVI":
        print("species: ", species, "chrom: ", chrom, "seq_id: ", seq_id)
        if species not in all_dict:
            all_dict[species] = {}
            all_dict[species][chrom] = set()
            all_dict[species][chrom].add(seq_id)
        else:
            if chrom not in all_dict[species]:
                all_dict[species][chrom] = set()
                all_dict[species][chrom].add(seq_id)
            else:
                all_dict[species][chrom].add(seq_id)

    for species, chroms in all_dict.items():
        for chrom, seq_ids in chroms.items():
            if chrom == "chrXVI":
                print(f"species: {species}, chrom: {chrom}, len(seq_ids): {len(seq_ids)}")


    # Calculate removal ratios
    removal_ratios = {}

    for species in all_dict.keys():
        removal_ratios[species] = {}
        for chrom in all_dict[species].keys():
            total_count = len(all_dict[species][chrom])
            removed_count = len(removed_dict.get(species, {}).get(chrom, set()))
            removal_ratio = removed_count / total_count if total_count > 0 else 0
            removal_ratios[species][chrom] = removal_ratio
            print(f"species: {species}, chrom: {chrom}, removal_ratio: {removal_ratio}")

    # Prepare data for visualization
    species_list = []
    chrom_list = []
    ratio_list = []

    for species, chroms in removal_ratios.items():
        for chrom, ratio in chroms.items():
            species_list.append(species)
            chrom_list.append(chrom)
            ratio_list.append(ratio)

    # Create a DataFrame for easy plotting
    plot_df = pd.DataFrame({
        'Species': species_list,
        'Chromosome': chrom_list,
        'Removal Ratio': ratio_list
    })

    # Sort the DataFrame by removal ratio in descending order
    plot_df = plot_df.sort_values('Removal Ratio', ascending=False)

    # # Create the bar chart
    # if data_type == "r64_gtf":
    #     plt.figure(figsize=(8, 4))
    # elif data_type == "strains_gtf":
    #     plt.figure(figsize=(100, 4))
    # elif data_type == "fungi":
    #     plt.figure(figsize=(75, 24))
    # elif data_type == "saccharomycetales_gtf":
    #     plt.figure(figsize=(500, 80))

    plt.figure(figsize=(8, 4))

    bars = plt.bar(range(len(plot_df)), plot_df['Removal Ratio'])

    # Customize the chart
    plt.xlabel('Species - Chromosome')
    plt.ylabel('Removal Ratio')
    plt.title('Removal Ratio by Species and Chromosome')
    # plt.xticks(range(len(plot_df)), [f"{s} - {c}" for s, c in zip(plot_df['Species'], plot_df['Chromosome'])], rotation=90)

    # Set y-axis limits from 1 to 0
    plt.ylim(0, 1)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        # plt.text(bar.get_x() + bar.get_width()/2., height,
        #         f'{height:.2f}',
        #         ha='center', va='bottom')

    # Adjust layout and display the chart
    plt.tight_layout()
    plt.savefig(f"/scratch4/khc/yeast_ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{data_type}/dataset_stats/dataset_similarity/minimap2/removal_ratios_{target}.png")

    plt.clf()
    # Print the removal ratios
    for species, chroms in removal_ratios.items():
        for chrom, ratio in chroms.items():
            print(f"Species: {species}, Chromosome: {chrom}, Removal Ratio: {ratio:.2f}")