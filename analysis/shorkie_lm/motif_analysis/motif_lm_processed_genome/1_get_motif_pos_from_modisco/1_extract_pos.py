import h5py

def read_bed_files(bed_files):
    """
    Read multiple BED files and return a list of entries.
    Each entry is a tuple:
       (chrom, start, end, label, identifier)
    The ordering of entries is preserved (i.e. first file, then second, then third).
    """
    bed_entries = []
    for bed_file in bed_files:
        with open(bed_file, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                # Assuming BED file columns: chrom, start, end, label, identifier
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
                        seqlets_group = pattern_group["seqlets"]
                        # Load arrays from the file
                        starts = seqlets_group["start"][:]
                        ends = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        
                        # Check for an optional quality/score dataset
                        scores = None
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
    return results

def map_seqlet_to_genome(seqlet, bed_entries):
    """
    Given a seqlet (with keys: example_idx, start, end, is_revcomp) and the bed_entries list,
    compute the genome coordinates for the seqlet.
    
    Assumes the seqlet positions are relative to the sequence extracted from the BED file.
    Genome coordinate = BED start + seqlet offset.
    
    Returns:
         (chrom, genome_start, genome_end, strand, species_id)
    """
    ex_idx = seqlet["example_idx"]
    # Look up the corresponding BED entry by example_idx
    try:
        chrom, bed_start, bed_end, label, species_id = bed_entries[ex_idx]
    except IndexError:
        raise ValueError(f"Example index {ex_idx} not found in BED entries.")
    
    # Map seqlet relative coordinates to genome coordinates.
    genome_start = bed_start + seqlet["start"]
    genome_end   = bed_start + seqlet["end"]

    strand = "-" if seqlet["is_revcomp"] else "+"

    return chrom, genome_start, genome_end, strand, species_id

def map_all_seqlets(h5_filepath, bed_files):
    """
    Reads the BED files, loads the modisco_results.h5 file,
    maps all seqlets to genomic coordinates, and returns a list of hits.
    
    Each hit is a dictionary with keys:
      - pattern_type (e.g., 'pos_patterns' or 'neg_patterns')
      - pattern_name (e.g., 'pattern_0', 'pattern_1', etc.)
      - seqlet_index (the index of the seqlet within the pattern)
      - example_idx (index in the BED file list)
      - species_id
      - chrom, genome_start, genome_end, strand
      - score
    """
    # Read and combine the BED files
    bed_entries = read_bed_files(bed_files)
    print(f"Loaded {len(bed_entries)} BED entries from {len(bed_files)} files.")
    # Extract seqlet positions from the modisco results file
    seqlet_results = extract_seqlet_positions(h5_filepath)
    
    hits = []
    for pattern_type, patterns in seqlet_results.items():
        for pattern_name, seqlets in patterns.items():
            for i, seqlet in enumerate(seqlets):
                chrom, genome_start, genome_end, strand, species_id = map_seqlet_to_genome(seqlet, bed_entries)
                hit = {
                    "pattern_type": pattern_type,
                    "pattern_name": pattern_name,
                    "seqlet_index": i,
                    "example_idx": seqlet["example_idx"],
                    "species_id": species_id,       # <-- Include species ID
                    "chrom": chrom,
                    "genome_start": genome_start,
                    "genome_end": genome_end,
                    "strand": strand,
                    "score": seqlet.get("score", None)  # Include score if available
                }
                hits.append(hit)
    return hits

if __name__ == "__main__":
    exp_dataset = "schizosaccharomycetales"
    # exp_dataset = "strains_select"

    if exp_dataset == "schizosaccharomycetales":
        # Paths to your modisco results file and the BED file
        modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/schizosaccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5"
        bed_files = [
            "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_schizosaccharomycetales_gtf_full/sequences_train.cleaned.bed"
        ]
    elif exp_dataset == "strains_select":
        modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/strains_select_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5"
        bed_files = [
            "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_strains_select_gtf/sequences_train.cleaned.bed"
        ]
    
    # Map all seqlets to genome coordinates
    hits = map_all_seqlets(modisco_h5, bed_files)
    
    # Show a subset of mapped hits, or just summary
    print("Mapped Motif Hits on Yeast Genome:")
    print("Number of hits:", len(hits))    
    # Get a dictionary of example_idx counts
    example_idx_counts = {}
    max_example_id = 0
    for hit in hits:
        print(f"{hit['species_id']} | {hit['chrom']}:{hit['genome_start']}-{hit['genome_end']} "
              f"(strand {hit['strand']}) | {hit['pattern_type']}/{hit['pattern_name']} "
              f"example_idx={hit['example_idx']} seqlet_index={hit['seqlet_index']} "
              f"score={hit.get('score', None)}")
        example_idx_counts[hit['example_idx']] = example_idx_counts.get(hit['example_idx'], 0) + 1
        max_example_id = max(max_example_id, hit['example_idx'])
    print(f"Max example index: {max_example_id}")
    # print("Example index counts:", example_idx_counts)
