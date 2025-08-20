#!/usr/bin/env python3
import os
import h5py
import pandas as pd
import numpy as np

def parse_motifs_html(motifs_html_file, qval_threshold):
    """
    Parse the motifs.html report to extract motif mapping.
    Expects the table to have columns: 'pattern', 'match0', 'qval0'.
    Returns a dictionary mapping modisco pattern IDs (with "/" as a separator)
    to a dict with keys 'best_match' and 'qval'.
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
    Read multiple BED files and return a list of entries.
    Each entry is a tuple:
       (chrom, start, end, label, identifier)
    The ordering of entries is preserved.
    """
    bed_entries = []
    for bed_file in bed_files:
        with open(bed_file, 'r') as f:
            for line in f:
                if line.strip() == "":
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
    Open the TF-MoDISco HDF5 results file and extract seqlet positional data.
    Returns a nested dictionary:
       { pattern_type: { pattern_name: list_of_seqlets } }
    Each seqlet is a dict with keys: 'example_idx', 'start', 'end', 'is_revcomp'
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
    Compute trimmed boundaries for each motif pattern in the modisco H5 file.
    For each pattern, we compute the per–position sum of absolute contribution scores,
    then define the trimmed window as the positions where the score is at least
    (max(score) * trim_threshold) with an extra 'pad' added on either side.
    
    Returns a dictionary mapping (pattern_type, pattern_name) to (trim_start, trim_end, full_length).
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
                    # If no positions pass, do not trim (use full length)
                    boundaries[(pattern_type, pattern_name)] = (0, full_length, full_length)
                else:
                    trim_start = max(np.min(pass_inds_fwd) - pad, 0)
                    trim_end = min(np.max(pass_inds_fwd) + pad + 1, full_length)
                    boundaries[(pattern_type, pattern_name)] = (trim_start, trim_end, full_length)
    return boundaries

def map_seqlet_to_genome(seqlet, bed_entries, pattern_type, pattern_name, boundaries):
    """
    Given a seqlet and the BED entries, compute the genome coordinates for the trimmed motif hit.
    The trimming is applied using the boundaries computed from the motif's contribution scores.
    
    For a forward-strand seqlet, the new coordinates are:
       new_start = bed_start + seqlet["start"] + trim_start
       new_end   = bed_start + seqlet["start"] + trim_end
       
    For a reverse-strand seqlet, the trimming is applied in the reverse direction:
       new_start = bed_start + seqlet["start"] + (full_length - trim_end)
       new_end   = bed_start + seqlet["start"] + (full_length - trim_start)
    
    Returns:
         (chrom, new_genome_start, new_genome_end, strand)
    """
    ex_idx = seqlet["example_idx"]
    try:
        chrom, bed_start, bed_end, label, identifier = bed_entries[ex_idx]
    except IndexError:
        raise ValueError(f"Example index {ex_idx} not found in BED entries.")
    
    # Retrieve trimming boundaries for this motif pattern.
    if (pattern_type, pattern_name) in boundaries:
        trim_start, trim_end, full_length = boundaries[(pattern_type, pattern_name)]
    else:
        # If no boundaries were computed, do not trim.
        trim_start, trim_end = 0, seqlet["end"] - seqlet["start"]
        full_length = seqlet["end"] - seqlet["start"]
    
    if not seqlet["is_revcomp"]:
        new_genome_start = bed_start + seqlet["start"] + trim_start
        new_genome_end   = bed_start + seqlet["start"] + trim_end
    else:
        # For reverse strand, flip the trimming boundaries.
        new_genome_start = bed_start + seqlet["start"] + (full_length - trim_end)
        new_genome_end   = bed_start + seqlet["start"] + (full_length - trim_start)
    strand = "-" if seqlet["is_revcomp"] else "+"
    return chrom, new_genome_start, new_genome_end, strand

def map_all_seqlets(h5_filepath, bed_files, boundaries):
    """
    Reads the BED files, loads the modisco results file, and maps all seqlets to genomic
    coordinates using the trimmed motif boundaries. Returns a list of hits.
    
    Each hit is a dictionary with keys:
      - pattern_type (e.g., 'pos_patterns' or 'neg_patterns')
      - pattern_name (e.g., 'pattern_0', 'pattern_1', etc.)
      - seqlet_index (the index of the seqlet within the pattern)
      - example_idx (index in the BED file list)
      - chrom, genome_start, genome_end, strand (trimmed coordinates)
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
    Write the mapped and trimmed seqlet hits to a BED file.
    Format: chrom, start, end, name, score, strand
    """
    with open(output_bed, 'w') as bf:
        for hit in hits:
            chrom = hit['chrom']
            start = hit['genome_start']
            end   = hit['genome_end']
            strand = hit['strand']
            # Use motif name if available; otherwise combine pattern info.
            name = hit['motif_name']['best_match']
            score = hit['motif_name']['qval']

            bf.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")

if __name__ == "__main__":
    # Define paths for modisco results, BED files, and motifs HTML.
    modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5"
    
    bed_files = [
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_train_r64.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/sequences_test.cleaned.bed",
        "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/sequences_valid.cleaned.bed"
    ]
    motifs_html = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM/saccharomycetales_viz_seq/unet_small_bert_drop/report_w16384_n100000_high_conf/motifs.html"
    
    # Compute trimmed boundaries for each motif pattern.
    boundaries = get_trim_boundaries(modisco_h5, trim_threshold=0.3, pad=4)
    
    # Map all seqlets to genome coordinates using the trimmed boundaries.
    hits = map_all_seqlets(modisco_h5, bed_files, boundaries)
    
    # Parse motif mapping if a motifs_html file is provided.
    motif_mapping = {}
    if motifs_html:
        motif_mapping = parse_motifs_html(motifs_html, qval_threshold=0.5)
        mapping_file = os.path.join("motif_mapping.tsv")
        with open(mapping_file, "w") as f:
            f.write("modisco_pattern\tbest_match\tqval0\n")
            for k, v in motif_mapping.items():
                f.write(f"{k}\t{v['best_match']}\t{v['qval']}\n")
        print("Saved motif mapping to", mapping_file)
    else:
        print("No motifs_html provided; using original motif labels.")
    
    # Add motif names to each hit.
    for hit in hits:
        key = f"{hit['pattern_type']}/{hit['pattern_name']}"
        hit['motif_name'] = motif_mapping.get(key, key.replace('/', '_'))
    
    print("Mapped and trimmed TF-MoDISco motif hits on Yeast Genome:")
    print("Number of hits:", len(hits))
    for hit in hits:
        print(f"{hit['chrom']}:{hit['genome_start']}-{hit['genome_end']} "
              f"(strand {hit['strand']}) | {hit['pattern_type']}/{hit['pattern_name']} "
              f"example_idx={hit['example_idx']} seqlet_index={hit['seqlet_index']}")
    
    # Write all trimmed hits to a BED file.
    output_bed = "tf_modisco_motif_hits_trimmed.bed"
    write_hits_to_bed(hits, output_bed)
    print(f"Wrote all trimmed TF-MoDISco motif hits to {output_bed}")
