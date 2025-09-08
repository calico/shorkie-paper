import h5py

def extract_seqlet_positions(h5_filepath):
    """
    Opens the TF-MoDISco results HDF5 file and extracts the start and end positions,
    along with example index and reverse complement flag, for each seqlet in each motif.

    Parameters:
        h5_filepath (str): Path to the modisco_results.h5 file.

    Returns:
        dict: A nested dictionary where the keys are the pattern types (e.g. 'pos_patterns'
              and 'neg_patterns'), then pattern names (e.g. 'pattern_0', 'pattern_1', etc.),
              and then a dictionary with keys 'start', 'end', 'example_idx', and 'is_revcomp'.
    """
    seqlet_positions = {}

    with h5py.File(h5_filepath, 'r') as f:
        # Loop over both positive and negative patterns (if present)
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type in f:
                seqlet_positions[pattern_type] = {}
                # Loop over each pattern (motif) in this group
                for pattern_name in f[pattern_type].keys():
                    pattern_group = f[pattern_type][pattern_name]
                    # Check if the "seqlets" subgroup is present
                    if "seqlets" in pattern_group:
                        seqlets_group = pattern_group["seqlets"]
                        # Extract the arrays for each attribute.
                        # These arrays are relative to your input sequences.
                        starts = seqlets_group["start"][:]
                        ends = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        
                        # Store the extracted arrays in a dictionary for this motif.
                        seqlet_positions[pattern_type][pattern_name] = {
                            "start": starts,
                            "end": ends,
                            "example_idx": example_idxs,
                            "is_revcomp": revcomps
                        }
    return seqlet_positions

if __name__ == "__main__":
    # Path to your modisco_results.h5 file
    # Paths to your modisco results file and the BED files in the same order
    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq_v1/unet_small_bert_drop/modisco_results.h5"

    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w500_n5000.h5"

    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w500_n20000.h5"

    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w2000_n5000.h5"

    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w2000_n20000.h5"

    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n20000.h5"
    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop_retry_1/modisco_results_w16384_n20000.h5"
    # modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop_retry_2/modisco_results_w16384_n20000.h5"


    modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5"

    positions = extract_seqlet_positions(modisco_h5)
    
    count = 0
    max_ex_idx = 0
    # Print out the results
    for pattern_type, patterns in positions.items():
        print(f"\nPattern type: {pattern_type}")
        for pattern_name, seqlet_info in patterns.items():
            print(f"  Pattern: {pattern_name}")
            num_seqlets = len(seqlet_info["start"])
            for i in range(num_seqlets):
                ex_idx = seqlet_info["example_idx"][i]
                start = seqlet_info["start"][i]
                end = seqlet_info["end"][i]
                rev = seqlet_info["is_revcomp"][i]
                print(f"    Seqlet {i}: example_idx={ex_idx}, start={start}, end={end}, is_revcomp={rev}")
                max_ex_idx = max(max_ex_idx, ex_idx)
                count += 1
    print(f"\nTotal number of seqlets: {count}")
    print(f"Max example index: {max_ex_idx}")
