import os
import h5py
import numpy as np

########################
# Adjustable parameters #
########################

# Thresholding for trimming
trim_threshold = 0.3  
trim_min_length = 3   
flank_size = 4

# A small pseudocount added to each position when converting to PFM
pseudocount = 1e-4

########################
# Helper Functions     #
########################
def cwm_to_pfm(cwm, pseudocount=1e-4):
    """
    Converts a (trimmed) CWM (which can have positive and negative values)
    into a pseudo–position frequency matrix by:
      1) Taking absolute values
      2) Normalizing so each position sums to 1
      3) Adding a small pseudocount
    """
    cwm_abs = np.abs(cwm)
    row_sums = np.sum(cwm_abs, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-8  # safeguard against divide-by-zero
    pfm = (cwm_abs + pseudocount) / (row_sums + 4 * pseudocount)
    return pfm

def write_meme(pfm, out_file, motif_name="motif"):
    """
    Writes a single motif in MEME format to 'out_file'.
    The pfm should be shape (length, 4) with rows summing to 1.
    """
    length = pfm.shape[0]
    with open(out_file, 'w') as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= A C G T\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        f.write(f"MOTIF {motif_name}\n\n")
        f.write(f"letter-probability matrix: alength= 4 w= {length} nsites= 20 E= 0\n")
        for row in pfm:
            f.write(" ".join([f"{val:.5f}" for val in row]) + "\n")

########################
# Main Extraction Code #
########################
def main():

    models = ["unet_small_bert_drop", "unet_small_bert_drop_retry_1", "unet_small_bert_drop_retry_2"]
    datasets = ["schizosaccharomycetales_viz_seq", "strains_select_viz_seq"]
    for dataset in datasets:
        for model in models:
            modisco_h5py = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_unseen_species/{dataset}/{model}/modisco_results.h5"
            # Output directory where you want to store motifs
            out_dir = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_unseen_species/{dataset}/{model}/motifs_meme"
            os.makedirs(out_dir, exist_ok=True)

            modisco_results = h5py.File(modisco_h5py, 'r')

            # Typically, TF-MoDISco Lite results are under keys like 'pos_patterns' or 'neg_patterns'
            pattern_groups = ['pos_patterns', 'neg_patterns']

            for group_name in pattern_groups:
                if group_name not in modisco_results:
                    continue  # Skip if group doesn't exist

                print(f"Processing group: {group_name}")
                pattern_group = modisco_results[group_name]

                # Sort patterns by numeric index if they are named like 'pattern_0', 'pattern_1', etc.
                def sort_key(item):
                    # item is (pattern_name, pattern_dataset)
                    # pattern_name is typically 'pattern_#'
                    name = item[0]
                    # for example, 'pattern_12' -> integer 12
                    return int(name.split("_")[-1])  

                for pattern_name, pattern in sorted(pattern_group.items(), key=sort_key):
                    # Extract the forward CWM
                    cwm_fwd = np.array(pattern['contrib_scores'][:])  # shape: (L, 4)
                    cwm_rev = cwm_fwd[::-1, ::-1]  # quick way to get reverse CWM

                    # Sum across A,C,G,T to see where the signal is
                    score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
                    score_rev = np.sum(np.abs(cwm_rev), axis=1)

                    # Trimming: threshold is a fraction (trim_threshold) of the max
                    trim_thresh_fwd = np.max(score_fwd) * trim_threshold
                    trim_thresh_rev = np.max(score_rev) * trim_threshold

                    pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
                    pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

                    if len(pass_inds_fwd) == 0 or len(pass_inds_rev) == 0:
                        # If no positions pass threshold, skip
                        print(f"Skipping {group_name}/{pattern_name}: no positions pass trim.")
                        continue

                    # Start and end indices, with some flank
                    start_fwd = max(np.min(pass_inds_fwd) - flank_size, 0)
                    end_fwd   = min(np.max(pass_inds_fwd) + flank_size + 1, cwm_fwd.shape[0])
                    start_rev = max(np.min(pass_inds_rev) - flank_size, 0)
                    end_rev   = min(np.max(pass_inds_rev) + flank_size + 1, cwm_rev.shape[0])

                    # Trim the CWM
                    trimmed_fwd = cwm_fwd[start_fwd:end_fwd]
                    trimmed_rev = cwm_rev[start_rev:end_rev]

                    # Convert to PFM
                    pfm_fwd = cwm_to_pfm(trimmed_fwd, pseudocount=pseudocount)
                    pfm_rev = cwm_to_pfm(trimmed_rev, pseudocount=pseudocount)

                    # Write each to its own MEME file
                    motif_id_fwd = f"{group_name}_{pattern_name}_fwd"
                    motif_id_rev = f"{group_name}_{pattern_name}_rev"

                    out_file_fwd = os.path.join(out_dir, motif_id_fwd + ".meme")
                    out_file_rev = os.path.join(out_dir, motif_id_rev + ".meme")

                    write_meme(pfm_fwd, out_file_fwd, motif_name=motif_id_fwd)
                    write_meme(pfm_rev, out_file_rev, motif_name=motif_id_rev)

                    print(f"Wrote {out_file_fwd} and {out_file_rev}")

            modisco_results.close()

if __name__ == "__main__":
    main()
