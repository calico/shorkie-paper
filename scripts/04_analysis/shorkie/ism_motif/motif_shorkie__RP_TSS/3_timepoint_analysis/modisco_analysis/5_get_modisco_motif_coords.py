import numpy as np
import h5py
import os
import pandas as pd


def extract_seqlet_positions(h5_filepath, tp):
    """
    Open the TF‑MoDISco HDF5 results file and extract seqlet positional data
    along with quality scores if available.
    Returns a nested dictionary:
       { pattern_type: { pattern_name: list_of_seqlets } }
    Each seqlet is a dict with keys: 'example_idx', 'start', 'end', 'is_revcomp',
    and optionally 'score' if present.
    """
    results = {}
    with h5py.File(h5_filepath, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type in f:
                results[pattern_type] = {}
                for pattern_name in f[pattern_type]:
                    pattern_group = f[pattern_type][pattern_name]
                    if "seqlets" in pattern_group:
                        # if tp == "T5":
                        #     if pattern_name != "pattern_2" or pattern_type != "neg_patterns":
                        #         continue
                        # elif tp == "T10":
                        #     if pattern_name != "pattern_4" or pattern_type != "neg_patterns":
                        #         continue
                        # elif tp == "T15":
                        #     if pattern_name != "pattern_4" or pattern_type != "neg_patterns":
                        #         continue
                        # elif tp == "T30":
                        #     if pattern_name != "pattern_4" or pattern_type != "neg_patterns":
                        #         continue
                        # elif tp == "T45":
                        #     if pattern_name != "pattern_3" or pattern_type != "neg_patterns":
                        #         continue
                        # elif tp == "T60":
                        #     if pattern_name != "pattern_2" or pattern_type != "neg_patterns":
                        #         continue
                        # elif tp == "T90":
                        #     if pattern_name != "pattern_4" or pattern_type != "neg_patterns":
                        #         continue
                        
                        if tp != "T180":
                            continue
                        elif tp == "T180":
                            if pattern_name != "pattern_1" or pattern_type != "pos_patterns":
                                continue


                        print("* Pattern name: ", pattern_name, "Pattern type: ", pattern_type, "TP: ", tp)

                        seqlets_group = pattern_group["seqlets"]
                        print(f"Processing {pattern_name} in {pattern_type} of {h5_filepath}")
                        # Load arrays from the file
                        starts = seqlets_group["start"][:]
                        ends = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        
                        # Check for an optional quality/score dataset
                        scores = None
                        # print("seqlets_group.keys(): ", seqlets_group.keys())
                        # print("seqlets_group['contrib_scores']: ", seqlets_group['contrib_scores'])
                        if "score" in seqlets_group:
                            scores = seqlets_group["score"][:]
                        
                        seqlets_list = []
                        for i in range(len(starts)):
                            seqlet = {
                                "example_idx": int(example_idxs[i]),
                                "start": int(starts[i]),
                                "end": int(ends[i]),
                                "is_revcomp": bool(revcomps[i])
                            }
                            if scores is not None:
                                seqlet["score"] = float(scores[i])
                            seqlets_list.append(seqlet)
                        results[pattern_type][pattern_name] = seqlets_list
                        print("\n\n")
    return results



# def extract_seqlet_coordinates(modisco_h5py_files, output_csv='motif_coordinates.csv'):
#     records = []
#     for modisco_file in modisco_h5py_files:
#         if not os.path.isfile(modisco_file):
#             print(f"[WARN] File not found: {modisco_file}")
#             continue
#         with h5py.File(modisco_file, 'r') as hf:
#             for group_name in ['pos_patterns', 'neg_patterns']:
#                 if group_name not in hf:
#                     continue
#                 metacluster = hf[group_name]
#                 for pattern_name, pattern in metacluster.items():
#                     if 'seqlets' in pattern:
#                         seqlets_item = pattern['seqlets']
#                         print(f"Processing {pattern_name} in {group_name} of {modisco_file}")
                        
#                         # Check if seqlets_item is a Dataset or a Group
#                         if isinstance(seqlets_item, h5py.Dataset):
#                             seqlets = seqlets_item[:]
#                         elif isinstance(seqlets_item, h5py.Group):
#                             seqlets = []
#                             for seqlet_key in seqlets_item.keys():
#                                 # Extract the dataset for each seqlet
#                                 seqlet_data = seqlets_item[seqlet_key][()]
#                                 seqlets.append(seqlet_data)
#                             seqlets = np.array(seqlets)
#                         else:
#                             print(f"Unexpected type for seqlets in pattern {pattern_name}")
#                             continue

#                     #     # Process each seqlet record.
#                     #     for seqlet in seqlets:
#                     #         record = {
#                     #             'modisco_file': modisco_file,
#                     #             'group': group_name,
#                     #             'pattern': pattern_name,
#                     #         }
#                     #         # Adjust the field names based on your TF‑MoDISco output.
#                     #         for field in ['example_idx', 'start', 'end', 'strand']:
#                     #             if field in seqlet.dtype.names:
#                     #                 record[field] = seqlet[field]
#                     #             else:
#                     #                 record[field] = None
#                     #         records.append(record)
#                     # else:
#                     #     print(f"No seqlets found for pattern {pattern_name} in {group_name}")
#     df = pd.DataFrame(records)
#     df.to_csv(output_csv, index=False)
#     print(f"Motif coordinates saved to {output_csv}")

if __name__ == '__main__':
    # Example definitions:
    # exp_dirs = ["gene_exp_motif_test_TSS/f0c0", "gene_exp_motif_test_RP/f0c0"]
    # target_tf = "MET4"
    target_tf = "SWI4"
    # List all directories in the results folder f"results/gene_exp_motif_test_{target_tf}targets/f0c0"
    base_dir = f"results/gene_exp_motif_test_{target_tf}_targets/f0c0/{target_tf}/"
    time_points = os.listdir(base_dir)
    # exp_dirs = [os.path.join(base_dir, exp_dir) for exp_dir in exp_dirs if os.path.isdir(os.path.join(base_dir, exp_dir))]
    print("time_points: ", time_points)

    # os.listdir(f"results/gene_exp_motif_test_{target_tf}targets/f0c0")
    # exp_dirs = [f"gene_exp_motif_test_{target_tf}targets/f0c0"]
    # target_tfs = ["MSN2_", "MSN4_"]
    # time_points = ["T0", "T5", "T10", "T15", "T30", "T45", "T60", "T90"]


    scores = ["logSED"]
    n = 10000
    w = 500

    # Build list of modisco .h5 files
    modisco_h5py_files = []
    for tp in time_points:
        score = scores[0]  # only one score in example
        # e.g. "results/gene_exp_motif_test_TSS/f0c0/MSN2_/T0/modisco_results_10000_500_diff.h5"
        path = f"{base_dir}{tp}/modisco_results_{n}_{w}_diff.h5"
        modisco_h5py_files.append(path)

        if not os.path.exists(path):
            continue

        print(f"Processing {path}")
        seqlet_positions = extract_seqlet_positions(path, tp)
        print(seqlet_positions)

    # print("modisco_h5py_files: ", modisco_h5py_files)
    # # # Build your list of modisco .h5 files as in your original code.
    # # modisco_files = [
    # #     "results/gene_exp_motif_test_TSS/f0c0/MSN2_/T0/modisco_results_10000_500_diff.h5",
    # #     # ... add the rest of your files
    # # ]
    # for modisco_file in modisco_h5py_files:
    #     print(f"Processing {modisco_file}")
    #     seqlet_positions = extract_seqlet_positions(modisco_file)
    #     print(seqlet_positions)
    # # extract_seqlet_coordinates(modisco_h5py_files)
