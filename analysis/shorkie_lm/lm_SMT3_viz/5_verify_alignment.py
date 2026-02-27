import numpy as np
import os

# 1. SpeciesLM Data (The Target)
# Extracted from nb_dump.txt (compute_SMT3_gene.ipynb)
# This is the 1003 bp sequence used in the notebook.
SPECIES_SEQ_STRING = 'GCTTCCCTCATTATTCCGCCCATGGCGTCTATTACCAAGCGTCATAATGTGCAATATTTGATATTATATAAGCTACTTGAGAAAGCGATAGTTTTTTTTTCTTACACAAAAAAAAAAAAACATAAAGCACCTATAACTCTCAACTTTGAAGAAGCACGAAAGGAATATGTTTAAATCAACAGAAATGTGAAAAAAATCGGTTATATATACAGAATCCGATTCTTTCTAACATCAAAGAGGTGGGGGAAGAAGGGACTCAAAAAAGAAACGACACTGCACAACCCGAGCCAAACTGACATACGAACACTAAAACCGATTTCCGAAAAAAACTTCAAATTTACATTTCATTGTCCGTCTGCCATCGCATCATCGCCTTCATCTCTAAGAGTTGCCGTGCCTTTCCATCCGCTTTCTTTTCATGCGGCGTTATTCTTTTTTCCTATTTTTGATGGTCCCTGTGCCGTTTCTTTTTCATGTTCACCGGTTTTTGGCGCCGCATACCGTACGGCGGGGCACTTTTGAAACGTTTTTGTGCATCCTGATGCCGTTTTCAAGGATCGCAAGCACGTCGCATAATACGGTAATGCCGAATTAAGGCTACGTCGTCATAGTAGGTTAGTCATGCGCGTTGGAAAAAGAAATGACCAACGCGTTGATTACGTAGTCCCCAAGGAATAATGCTTTTGAAAGTGAAAAAAAAAAATAAAACTGAAAAAAGCCATGCTGTTTCCATCACGTGCATGTCACGTTTTTGCCGCCGAACTCTTTGATCATGTGATATGAATATGTTGGGTTACCCAGCTTTGCCAACACGCGCCGTCGGAAGGTGTTCAGGAAGCAGGAAAAGAGCAAAACACCAACAATCAAACAAACGAACACATTCTACTCTTTTAGTTGATTTTTCTTACCTTTTCCAAGCTCCCGTTTCTTGTTACCACCTGTAGCATATAGGACAGAAGGACCCAGTTCAGTTCTAGTTTTACAAATAAATACACGAGCGATG'

# 1_viz_dna_pwm_julien.py plots [690:986]
SPECIES_START = 690
SPECIES_END = 986

# 2. Load Shorkie_LM Data (My Output)
SHORKIE_FILE = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/Shorkie_LM_SMT3_viz/inference_smt3_output/preds_smt3_unmasked.npz"

def onehot_to_seq(onehot):
    # onehot: (L, 4)
    # 0:A, 1:C, 2:G, 3:T
    seq = []
    for row in onehot:
        idx = np.argmax(row)
        # Assuming probabilistic output, max prob is the base.
        if idx == 0: seq.append('A')
        elif idx == 1: seq.append('C')
        elif idx == 2: seq.append('G')
        elif idx == 3: seq.append('T')
    return "".join(seq)

def main():
    print("--- Verifying Sequence Alignment (String Based) ---")
    
    # Extract Target Substring
    species_target_sub = SPECIES_SEQ_STRING[SPECIES_START:SPECIES_END]
    print(f"SpeciesLM Target Substring (Length {len(species_target_sub)}):")
    print(f"{species_target_sub[:30]}...{species_target_sub[-30:]}")
    
    # Load Shorkie
    print(f"Loading Shorkie: {SHORKIE_FILE}")
    shorkie_data = np.load(SHORKIE_FILE)
    shorkie_x_true = shorkie_data['x_true'] 
    
    GENE_START = 1469400
    PAD_BP = 512
    # My Windows
    SMT3_WINDOWS = [
        ("chrIV", 1454592, 1470976),
        ("chrIV", 1458688, 1475072),
        ("chrIV", 1462784, 1479168),
        ("chrIV", 1466880, 1483264),
    ]
    
    for i, (chrom, start, end) in enumerate(SMT3_WINDOWS):
        win_seq_onehot = shorkie_x_true[i]
        win_seq = onehot_to_seq(win_seq_onehot)
        
        # We need to find the SpeciesLM target substring in this window
        idx = win_seq.find(species_target_sub)
        rel_start = GENE_START - start
        
        print(f"\nWindow {i}: RelStart of Gene={rel_start}")
        
        if idx != -1:
            print(f"Match FOUND at index {idx} in Window {i}!")
            
            # Now calculating the offset relative to the 512bp buffer
            # In 4_viz_smt3_logo_unmasked.py:
            # upstream_slice = x_true_seq[rel_start - PAD_BP : rel_start]
            
            upstream_buffer_start_idx = rel_start - PAD_BP
            
            # The match starts at 'idx' in the full window sequence.
            # We want to know where it starts in the 'upstream_slice'.
            # Offset = MatchStart - SliceStart
            
            offset_in_slice = idx - upstream_buffer_start_idx
            
            print(f"Slice Start Index in Window: {upstream_buffer_start_idx}")
            print(f"Match Start Index in Window: {idx}")
            print(f"Offset (PLOT_START) should be: {offset_in_slice}")
            print(f"End (PLOT_END) should be: {offset_in_slice + len(species_target_sub)}")
            
            # Validation
            extracted_slice = win_seq[upstream_buffer_start_idx : rel_start]
            extracted_plot_region = extracted_slice[offset_in_slice : offset_in_slice + len(species_target_sub)]
            
            if extracted_plot_region == species_target_sub:
                print("SUCCESS: Exact string match confirmed with calculated offsets.")
            else:
                print("FAILURE: Calculated offsets do not produce identical string.")
                print(f"Target: {species_target_sub[:20]}")
                print(f"Actual: {extracted_plot_region[:20]}")
                
            break
        else:
             print("Match NOT found in this window.")

if __name__ == "__main__":
    main()
