#!/usr/bin/env python3
import os
import h5py
import pandas as pd
import numpy as np
from collections import defaultdict

# ---- Helper functions for chromosome conversion ----

def roman_to_int(s):
    """
    Convert a Roman numeral string to an integer.
    """
    roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for char in s[::-1]:
        value = roman_dict.get(char, 0)
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total

def convert_chr_name(chrom):
    """
    Converts a chromosome name from Roman numeral format (e.g., "chrI", "chrII")
    to Arabic numeral format (e.g., "chr1", "chr2"). If already numeric, returns unchanged.
    """
    if chrom.startswith("chr"):
        suffix = chrom[3:]
        if suffix.isdigit():
            return chrom
        if all(c in "IVXLCDM" for c in suffix.upper()):
            num = roman_to_int(suffix.upper())
            return "chr" + str(num)
    return chrom

# ---- Functions for parsing modisco results ----

def parse_motifs_html(motifs_html_file, qval_threshold):
    """
    Parse the motifs.html report to extract motif mapping.
    Expects columns: 'pattern', 'match0', 'qval0'.
    Returns a dict mapping modisco pattern IDs to a dict with keys 'best_match' and 'qval'.
    """
    dfs = pd.read_html(motifs_html_file)
    if not dfs:
        print("No tables found in motifs.html")
        return {}
    df = dfs[0]
    mapping = {}
    for idx, row in df.iterrows():
        raw_pattern = row['pattern']
        modisco_pattern_id = raw_pattern.replace(".", "/")
        try:
            qval = float(row['qval0'])
        except (ValueError, TypeError):
            qval = None
        best_match = row['match0'] if pd.notnull(row['match0']) else raw_pattern
        mapping[modisco_pattern_id] = {'best_match': best_match, 'qval': qval}
    return mapping

def read_bed_files(bed_files):
    """
    Read one or more BED files and return a list of entries.
    Each entry is a tuple: (chrom, start, end, label, identifier).
    """
    bed_entries = []
    for bed_file in bed_files:
        with open(bed_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                fields = line.strip().split()
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                label = fields[3] if len(fields) > 3 else ""
                identifier = fields[4] if len(fields) > 4 else ""
                bed_entries.append((chrom, start, end, label, identifier))
    return bed_entries

def extract_seqlet_positions(h5_filepath):
    """
    Open the TF-MoDISco HDF5 file and extract seqlet positions.
    Returns a nested dict: { pattern_type: { pattern_name: list_of_seqlets } }.
    Each seqlet is a dict with keys: 'example_idx', 'start', 'end', 'is_revcomp'.
    """
    results = {}
    with h5py.File(h5_filepath, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type in f:
                results[pattern_type] = {}
                for pattern_name in f[pattern_type].keys():
                    pattern_group = f[pattern_type][pattern_name]
                    if "seqlets" in pattern_group:
                        seqlets_group = pattern_group["seqlets"]
                        starts = seqlets_group["start"][:]
                        ends = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        seqlets_list = []
                        for i in range(len(starts)):
                            seqlet = {
                                "example_idx": int(example_idxs[i]),
                                "start": int(starts[i]),
                                "end": int(ends[i]),
                                "is_revcomp": bool(revcomps[i])
                            }
                            seqlets_list.append(seqlet)
                        results[pattern_type][pattern_name] = seqlets_list
    return results

def get_trim_boundaries(modisco_h5, trim_threshold=0.3, pad=4):
    """
    Compute trimmed boundaries for each motif in the modisco HDF5 file.
    Returns a dict mapping (pattern_type, pattern_name) to (trim_start, trim_end, full_length).
    """
    boundaries = {}
    with h5py.File(modisco_h5, 'r') as f:
        for pattern_type in ['pos_patterns', 'neg_patterns']:
            if pattern_type not in f:
                continue
            for pattern_name in f[pattern_type].keys():
                pattern_group = f[pattern_type][pattern_name]
                if 'contrib_scores' not in pattern_group:
                    continue
                cwm_fwd = np.array(pattern_group['contrib_scores'][:])
                full_length = cwm_fwd.shape[0]
                score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
                trim_thresh_fwd = np.max(score_fwd) * trim_threshold
                pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
                if len(pass_inds_fwd) == 0:
                    boundaries[(pattern_type, pattern_name)] = (0, full_length, full_length)
                else:
                    trim_start = max(np.min(pass_inds_fwd) - pad, 0)
                    trim_end = min(np.max(pass_inds_fwd) + pad + 1, full_length)
                    boundaries[(pattern_type, pattern_name)] = (trim_start, trim_end, full_length)
    return boundaries

def map_seqlet_to_genome(seqlet, bed_entries, pattern_type, pattern_name, boundaries):
    """
    Map a seqlet to genomic coordinates using the trimming boundaries.
    Returns: (chrom, new_genome_start, new_genome_end, strand).
    """
    ex_idx = seqlet["example_idx"]
    try:
        chrom, bed_start, bed_end, label, identifier = bed_entries[ex_idx]
    except IndexError:
        raise ValueError(f"Example index {ex_idx} not found in BED entries.")
    if (pattern_type, pattern_name) in boundaries:
        trim_start, trim_end, full_length = boundaries[(pattern_type, pattern_name)]
    else:
        trim_start, trim_end = 0, seqlet["end"] - seqlet["start"]
        full_length = seqlet["end"] - seqlet["start"]
    if not seqlet["is_revcomp"]:
        new_genome_start = bed_start + seqlet["start"] + trim_start
        new_genome_end   = bed_start + seqlet["start"] + trim_end
    else:
        new_genome_start = bed_start + seqlet["start"] + (full_length - trim_end)
        new_genome_end   = bed_start + seqlet["start"] + (full_length - trim_start)
    strand = "-" if seqlet["is_revcomp"] else "+"
    return chrom, new_genome_start, new_genome_end, strand

def map_all_seqlets(h5_filepath, bed_files, boundaries):
    """
    Reads the BED files, loads the modisco results file, and maps all seqlets.
    Returns a list of hits (each is a dict with genomic coordinates).
    """
    bed_entries = read_bed_files(bed_files)
    print(f"Loaded {len(bed_entries)} BED entries from {len(bed_files)} files.")
    seqlet_results = extract_seqlet_positions(h5_filepath)
    hits = []
    for pattern_type, patterns in seqlet_results.items():
        for pattern_name, seqlets in patterns.items():
            for i, seqlet in enumerate(seqlets):
                chrom, new_start, new_end, strand = map_seqlet_to_genome(
                    seqlet, bed_entries, pattern_type, pattern_name, boundaries)
                hit = {
                    "pattern_type": pattern_type,
                    "pattern_name": pattern_name,
                    "seqlet_index": i,
                    "example_idx": seqlet["example_idx"],
                    "chrom": chrom,
                    "genome_start": new_start,
                    "genome_end": new_end,
                    "strand": strand
                }
                hits.append(hit)
    return hits

def write_hits_to_bed(hits, output_bed):
    """
    Write the mapped seqlet hits to a BED file.
    Format: chrom, start, end, name, score, strand.
    """
    with open(output_bed, 'w') as bf:
        for hit in hits:
            chrom = hit['chrom']
            start = hit['genome_start']
            end   = hit['genome_end']
            strand = hit['strand']
            name = hit['motif_name']['best_match'] if isinstance(hit['motif_name'], dict) else hit['motif_name']
            score = hit['motif_name']['qval'] if isinstance(hit['motif_name'], dict) else 0
            bf.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")

# ---- Overlap and Evaluation Functions ----

def intervals_overlap(start1, end1, start2, end2):
    """
    Returns True if the intervals (start1, end1) and (start2, end2) overlap.
    """
    return start1 < end2 and start2 < end1

def find_overlapping_hits(hits, tf_bed_entries):
    """
    Given a list of hits (with keys 'chrom', 'genome_start', 'genome_end')
    and a list of TF BED entries, return a list of overlaps.
    Chromosome names are converted for consistent comparison.
    """
    overlaps = []
    for hit in hits:
        hit_chr = convert_chr_name(hit['chrom'])
        hit_start = hit['genome_start']
        hit_end = hit['genome_end']
        for entry in tf_bed_entries:
            tf_chr = convert_chr_name(entry[0])
            tf_start = entry[1]
            tf_end = entry[2]
            tf_label = entry[3]
            tf_id = entry[4]
            if hit_chr == tf_chr and intervals_overlap(hit_start, hit_end, tf_start, tf_end):
                overlaps.append({
                    "hit": hit,
                    "tf_entry": {
                        "chrom": tf_chr,
                        "start": tf_start,
                        "end": tf_end,
                        "label": tf_label,
                        "identifier": tf_id
                    }
                })
    return overlaps

def filter_hits_by_motif(hits, motif_query):
    """
    Filter the list of TF-MoDISco hits to include only those whose assigned motif
    (given by 'best_match') contains the motif_query substring (case-insensitive).
    """
    filtered = []
    for hit in hits:
        motif = hit['motif_name']
        if isinstance(motif, dict):
            if motif_query.upper() in motif.get('best_match', "").upper():
                filtered.append(hit)
        else:
            if motif_query.upper() in motif.upper():
                filtered.append(hit)
    return filtered

def count_overlaps(predicted, ground_truth):
    """
    Count the number of predicted intervals that overlap any ground truth interval,
    and count the number of ground truth intervals overlapped by any prediction.
    Returns (predicted_true_positives, ground_truth_true_positives).
    """
    pred_tp = 0
    for pred in predicted:
        p_chrom = convert_chr_name(pred['chrom'])
        p_start = pred['genome_start']
        p_end = pred['genome_end']
        found = any(p_chrom == convert_chr_name(gt[0]) and intervals_overlap(p_start, p_end, gt[1], gt[2])
                    for gt in ground_truth)
        if found:
            pred_tp += 1

    gt_tp = 0
    for gt in ground_truth:
        gt_chrom = convert_chr_name(gt[0])
        gt_start, gt_end = gt[1], gt[2]
        found = any(gt_chrom == convert_chr_name(pred['chrom']) and intervals_overlap(pred['genome_start'], pred['genome_end'], gt_start, gt_end)
                    for pred in predicted)
        if found:
            gt_tp += 1

    return pred_tp, gt_tp

def merge_intervals(intervals):
    """
    Merge overlapping intervals given as a list of (start, end) tuples.
    """
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    current_start, current_end = sorted_intervals[0]
    for s, e in sorted_intervals[1:]:
        if s <= current_end:
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))
    return merged

def intersection_length(intervals1, intervals2):
    """
    Compute the total length of intersection between two lists of merged intervals.
    """
    i, j, total = 0, 0, 0
    while i < len(intervals1) and j < len(intervals2):
        start1, end1 = intervals1[i]
        start2, end2 = intervals2[j]
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        if overlap_start < overlap_end:
            total += overlap_end - overlap_start
        if end1 < end2:
            i += 1
        else:
            j += 1
    return total

def group_intervals_by_chrom(predicted, ground_truth):
    """
    Group predicted and ground truth intervals by chromosome.
    Returns two dicts: {chrom: list of (start, end)} for predicted and ground truth.
    """
    pred_dict = defaultdict(list)
    for pred in predicted:
        chrom = convert_chr_name(pred['chrom'])
        pred_dict[chrom].append((pred['genome_start'], pred['genome_end']))
    gt_dict = defaultdict(list)
    for gt in ground_truth:
        chrom = convert_chr_name(gt[0])
        gt_dict[chrom].append((gt[1], gt[2]))
    return pred_dict, gt_dict

def total_base_agreement(pred_dict, gt_dict):
    """
    Compute base-level agreement as the Jaccard index: intersection length / union length.
    """
    total_intersection = 0
    total_union = 0
    all_chroms = set(list(pred_dict.keys()) + list(gt_dict.keys()))
    for chrom in all_chroms:
        pred_intervals = merge_intervals(pred_dict.get(chrom, []))
        gt_intervals = merge_intervals(gt_dict.get(chrom, []))
        pred_length = sum(e - s for s, e in pred_intervals)
        gt_length = sum(e - s for s, e in gt_intervals)
        inter = intersection_length(pred_intervals, gt_intervals)
        union = pred_length + gt_length - inter
        total_intersection += inter
        total_union += union
    return total_intersection / total_union if total_union > 0 else 0

# ---- Functions for Bin-level Metrics ----

def is_bin_covered(bin_start, bin_end, intervals):
    """
    Return True if the bin [bin_start, bin_end) overlaps any interval in intervals.
    """
    for s, e in intervals:
        if intervals_overlap(s, e, bin_start, bin_end):
            return True
    return False

def compute_bin_level_metrics(pred_dict, gt_dict, bin_size=10):
    """
    Compute bin-level TP, FP, TN, FN over the union of predicted and ground truth intervals
    for each chromosome (using the provided bin_size), then return overall metrics.
    """
    total_TP = total_FP = total_TN = total_FN = 0
    for chrom in set(list(pred_dict.keys()) + list(gt_dict.keys())):
        pred_intervals = pred_dict.get(chrom, [])
        gt_intervals = gt_dict.get(chrom, [])
        if not pred_intervals and not gt_intervals:
            continue
        all_intervals = pred_intervals + gt_intervals
        region_start = min(s for s, e in all_intervals)
        region_end = max(e for s, e in all_intervals)
        for b in range(region_start, region_end, bin_size):
            bin_start = b
            bin_end = min(b + bin_size, region_end)
            pred_label = 1 if is_bin_covered(bin_start, bin_end, pred_intervals) else 0
            gt_label = 1 if is_bin_covered(bin_start, bin_end, gt_intervals) else 0
            if pred_label == 1 and gt_label == 1:
                total_TP += 1
            elif pred_label == 1 and gt_label == 0:
                total_FP += 1
            elif pred_label == 0 and gt_label == 1:
                total_FN += 1
            else:
                total_TN += 1
    return total_TP, total_FP, total_TN, total_FN

# ---- Main Execution ----

if __name__ == "__main__":
    # Define paths for modisco results, sequence BED files, and motifs HTML.
    modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5"
    bed_files = [
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_train_r64.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/sequences_test.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_valid.cleaned.bed"
    ]
    motifs_html = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/report_w16384_n100000/motifs.html"
    
    # Chip peaks directory: each BED file is named like "Abf1_CX.bed"
    chip_peak_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/chip_peaks"
    chip_bed_files = [os.path.join(chip_peak_dir, f) for f in os.listdir(chip_peak_dir) if f.endswith("_CX.bed")]
    
    # Create a base output directory
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Compute trimmed boundaries from the modisco HDF5 file.
    boundaries = get_trim_boundaries(modisco_h5, trim_threshold=0.3, pad=4)
    
    # Map all seqlets to genomic coordinates.
    hits = map_all_seqlets(modisco_h5, bed_files, boundaries)
    
    # Parse motif mapping (if available) and assign motif names.
    motif_mapping = {}
    if motifs_html:
        motif_mapping = parse_motifs_html(motifs_html, qval_threshold=0.5)
        mapping_file = os.path.join(output_dir, "motif_mapping.tsv")
        with open(mapping_file, "w") as f:
            f.write("modisco_pattern\tbest_match\tqval0\n")
            for k, v in motif_mapping.items():
                f.write(f"{k}\t{v['best_match']}\t{v['qval']}\n")
        print("Saved motif mapping to", mapping_file)
    else:
        print("No motifs_html provided; using original motif labels.")
    
    # Assign motif names to each hit using the mapping.
    for hit in hits:
        key = f"{hit['pattern_type']}/{hit['pattern_name']}"
        hit['motif_name'] = motif_mapping.get(key, key.replace('/', '_'))
    
    # Write all modisco hits to a BED file.
    all_hits_bed = os.path.join(output_dir, "tf_modisco_motif_hits_trimmed.bed")
    write_hits_to_bed(hits, all_hits_bed)
    print(f"Wrote all trimmed TF-MoDISco motif hits to {all_hits_bed}")
    
    # ---- Evaluate for each chip peak BED file and store outputs hierarchically ----
    summary_metrics = []
    bin_size = 10  # You can adjust the bin size as needed.
    print("\nEvaluating TF-Modisco predictions against chip peaks:")
    for chip_bed in chip_bed_files:
        base = os.path.basename(chip_bed)
        # Assume motif name is the first part before an underscore, e.g., "Abf1" from "Abf1_CX.bed"
        motif_query = base.split("_")[0]
        print(f"\nProcessing chip peak file: {base} (motif query: {motif_query})")
        
        # Create subdirectory for the motif
        motif_out_dir = os.path.join(output_dir, motif_query)
        if not os.path.exists(motif_out_dir):
            os.makedirs(motif_out_dir)
        
        # Filter modisco hits that contain the motif_query (e.g., "ABF1.1" will match "Abf1")
        filtered_hits = filter_hits_by_motif(hits, motif_query)
        print(f"  Number of modisco hits for motif '{motif_query}': {len(filtered_hits)}")
        
        # Read chip peak BED file entries.
        chip_entries = read_bed_files([chip_bed])
        print(f"  Number of chip peaks: {len(chip_entries)}")
        
        # Find overlapping hits.
        overlapping_hits = find_overlapping_hits(filtered_hits, chip_entries)
        print(f"  Overlapping hits (region-level): {len(overlapping_hits)}")
        
        # ---- Region-level Metrics ----
        pred_tp, gt_tp = count_overlaps(filtered_hits, chip_entries)
        region_precision = pred_tp / len(filtered_hits) if filtered_hits else 0
        region_recall = gt_tp / len(chip_entries) if chip_entries else 0
        # Define region-level Jaccard as TP/(#predicted + #ground_truth - TP)
        region_jaccard = pred_tp / (len(filtered_hits) + len(chip_entries) - pred_tp) if (len(filtered_hits) + len(chip_entries) - pred_tp) > 0 else 0
        # Define region-level accuracy as the F1 score.
        region_accuracy = (2 * region_precision * region_recall / (region_precision + region_recall)
                           if (region_precision + region_recall) > 0 else 0)
        
        # ---- Bin-level Metrics ----
        pred_dict, gt_dict = group_intervals_by_chrom(filtered_hits, chip_entries)
        TP_bin, FP_bin, TN_bin, FN_bin = compute_bin_level_metrics(pred_dict, gt_dict, bin_size=bin_size)
        bin_precision = TP_bin / (TP_bin + FP_bin) if (TP_bin + FP_bin) > 0 else 0
        bin_recall = TP_bin / (TP_bin + FN_bin) if (TP_bin + FN_bin) > 0 else 0
        bin_accuracy = (TP_bin + TN_bin) / (TP_bin + FP_bin + TN_bin + FN_bin) if (TP_bin + FP_bin + TN_bin + FN_bin) > 0 else 0
        bin_jaccard = TP_bin / (TP_bin + FP_bin + FN_bin) if (TP_bin + FP_bin + FN_bin) > 0 else 0
        
        print(f"  Region-level Precision: {region_precision:.3f}")
        print(f"  Region-level Recall: {region_recall:.3f}")
        print(f"  Region-level Accuracy (F1): {region_accuracy:.3f}")
        print(f"  Region-level Jaccard: {region_jaccard:.3f}")
        print(f"  Bin-level Precision: {bin_precision:.3f}")
        print(f"  Bin-level Recall: {bin_recall:.3f}")
        print(f"  Bin-level Accuracy: {bin_accuracy:.3f}")
        print(f"  Bin-level Jaccard: {bin_jaccard:.3f}")
        
        # Save overlapping hits to a BED file in the motif subdirectory.
        out_file = os.path.join(motif_out_dir, f"overlapping_hits_{motif_query}.bed")
        with open(out_file, 'w') as out_f:
            for overlap in overlapping_hits:
                hit = overlap["hit"]
                tf_entry = overlap["tf_entry"]
                motif_name = hit['motif_name']['best_match'] if isinstance(hit['motif_name'], dict) else hit['motif_name']
                out_f.write(f"{hit['chrom']}\t{hit['genome_start']}\t{hit['genome_end']}\t{motif_name}\t{tf_entry['label']}\t{hit['strand']}\n")
        print(f"  Wrote overlapping hits to {out_file}")
        
        # Append metrics to the summary.
        summary_metrics.append({
            "motif": motif_query,
            "modisco_hits": len(filtered_hits),
            "chip_peaks": len(chip_entries),
            "region_overlapping_hits": len(overlapping_hits),
            "region_precision": region_precision,
            "region_recall": region_recall,
            "region_accuracy_F1": region_accuracy,
            "region_jaccard": region_jaccard,
            "bin_precision": bin_precision,
            "bin_recall": bin_recall,
            "bin_accuracy": bin_accuracy,
            "bin_jaccard": bin_jaccard
        })
    
    # Create a summary DataFrame and save as CSV.
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv = os.path.join(output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\nSummary of evaluation metrics for all motifs:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary metrics to {summary_csv}")
