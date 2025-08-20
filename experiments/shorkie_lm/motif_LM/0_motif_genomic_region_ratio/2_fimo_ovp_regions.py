import os
from pybedtools import BedTool

for motif_idx in range(100):
    motif_bed=f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/fimo_out/pos_patterns_pattern_{motif_idx}_fwd/motifs.bed"
    if not os.path.exists(motif_bed):
        continue
    # Load the BED files
    motifs = BedTool(motif_bed)
    genes = BedTool("genes.bed")

    # Convert motifs to a DataFrame
    motifs_df = motifs.to_dataframe()

    # Filter rows where the 5th column (score) > 1
    filtered_motifs_df = motifs_df[motifs_df.iloc[:, 4] > 5]

    # Convert the filtered DataFrame back to a BedTool object
    filtered_motifs = BedTool.from_dataframe(filtered_motifs_df)

    # Calculate intersections
    motifs_in_genes = filtered_motifs.intersect(genes, u=True)
    motifs_in_intergenic = filtered_motifs.intersect(genes, v=True)

    # Count the filtered motifs
    n_in_genes = len(motifs_in_genes)
    n_in_intergenic = len(motifs_in_intergenic)

    # Calculate the genic ratio
    ratio_genic = n_in_genes / (n_in_genes + n_in_intergenic)
    ratio_intergenic = n_in_intergenic / (n_in_genes + n_in_intergenic)
    print(f"Motif {motif_idx}:")
    print("\tGenic ratio:", ratio_genic)
    print("\tIntergenic ratio:", ratio_intergenic)
    print("\n\n")
