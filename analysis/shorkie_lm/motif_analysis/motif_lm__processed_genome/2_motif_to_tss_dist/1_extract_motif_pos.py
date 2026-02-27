import h5py
import os
import pandas as pd

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
    The ordering of entries is preserved (i.e. first file, then second, etc.).
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
                        # Load arrays from the file
                        starts = seqlets_group["start"][:]
                        ends = seqlets_group["end"][:]
                        example_idxs = seqlets_group["example_idx"][:]
                        revcomps = seqlets_group["is_revcomp"][:]
                        
                        # Create a list of dictionaries (one per seqlet)
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

def map_seqlet_to_genome(seqlet, bed_entries):
    """
    Given a seqlet (with keys: example_idx, start, end, is_revcomp) and the bed_entries list,
    compute the genome coordinates for the seqlet.
    
    Assumes the seqlet positions are relative to the sequence extracted from the BED file.
    Genome coordinate = BED start + seqlet offset.
    
    Returns:
         (chrom, genome_start, genome_end, strand)
    """
    ex_idx = seqlet["example_idx"]
    # Look up the corresponding BED entry by example_idx
    try:
        chrom, bed_start, bed_end, label, identifier = bed_entries[ex_idx]
    except IndexError:
        raise ValueError(f"Example index {ex_idx} not found in BED entries.")
    
    # Map seqlet relative coordinates to genome coordinates.
    genome_start = bed_start + seqlet["start"]
    genome_end   = bed_start + seqlet["end"]
    # Mark the strand
    strand = "-" if seqlet["is_revcomp"] else "+"
    return chrom, genome_start, genome_end, strand, identifier

def map_all_seqlets(h5_filepath, bed_files):
    """
    Reads the BED files, loads the modisco_results.h5 file,
    maps all seqlets to genomic coordinates, and returns a list of hits.
    
    Each hit is a dictionary with keys:
      - pattern_type (e.g., 'pos_patterns' or 'neg_patterns')
      - pattern_name (e.g., 'pattern_0', 'pattern_1', etc.)
      - seqlet_index (the index of the seqlet within the pattern)
      - example_idx (index in the BED file list)
      - chrom, genome_start, genome_end, strand
      - species (the genome species associated with the hit)
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
                chrom, genome_start, genome_end, strand, species= map_seqlet_to_genome(seqlet, bed_entries)
                hit = {
                    "pattern_type": pattern_type,
                    "pattern_name": pattern_name,
                    "seqlet_index": i,
                    "example_idx": seqlet["example_idx"],
                    "chrom": chrom,
                    "genome_start": genome_start,
                    "genome_end": genome_end,
                    "strand": strand,
                    "species": species   # add species information to the hit
                }
                hits.append(hit)
    return hits

def write_hits_to_bed(hits, output_bed):
    """
    Write the mapped seqlets (hits) to a BED file.
    Format: combined_chrom, start, end, name, score, strand
    The combined_chrom field is a concatenation of species and chromosome (e.g., species:chrom)
    to ensure each sequence is unique, even if different species share the same chromosome name.
    """
    with open(output_bed, 'w') as bf:
        for hit in hits:
            # Combine species with chrom, e.g. "strains_select:chrXI"
            species = hit.get('species', 'NA')
            chrom = hit['chrom']
            # combined_chrom = f"{species}:{chrom}"
            start = hit['genome_start']
            end   = hit['genome_end']
            strand = hit['strand']
            # Optionally, you can also include species in the name field if desired.
            name = f"{hit['motif_name']['best_match']}"
            score = "0"
            bf.write(f"{species}:{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")

if __name__ == "__main__":
    # Here exp_species holds the species (or experiment) identifier.
    # (For example, you might have different species or strain sets.)
    # exp_species = "strains_select"   # or "schizosaccharomycetales"
    exp_species = "schizosaccharomycetales"
    
    if exp_species == "schizosaccharomycetales":
        modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/schizosaccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5"
        motifs_html = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/schizosaccharomycetales_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000_high_conf/motifs.html"
        bed_files = [
            "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_schizosaccharomycetales_gtf_full/sequences_train.cleaned.bed"
        ]
    elif exp_species == "strains_select":
        modisco_h5 = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/strains_select_viz_seq/unet_small_bert_drop/modisco_results_w_16384_n_1000000.h5"
        motifs_html = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/strains_select_viz_seq/unet_small_bert_drop/report_w_16384_n_1000000/motifs.html"
        bed_files = [
            "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_strains_select_gtf/sequences_train.cleaned.bed"
        ]
        
    output_dir = f"results/{exp_species}_motif_hits"
    os.makedirs(output_dir, exist_ok=True)
    
    # Map all seqlets to genome coordinates and pass the species information.
    hits = map_all_seqlets(modisco_h5, bed_files)
    
    # Parse motif mapping if a motifs_html file is provided.
    motif_mapping = {}
    if motifs_html:
        motif_mapping = parse_motifs_html(motifs_html, qval_threshold=0.05)
        mapping_file = os.path.join(output_dir, "motif_mapping.tsv")
        with open(mapping_file, "w") as f:
            f.write("modisco_pattern\tbest_match\tqval0\n")
            for k, v in motif_mapping.items():
                f.write(f"{k}\t{v['best_match']}\t{v['qval']}\n")
        print("Saved motif mapping to", mapping_file)
    else:
        print("No motifs_html provided; using original motif labels.")
    
    # Add motif names to each hit.
    # The key is built from the pattern type and pattern name.
    for hit in hits:
        key = f"{hit['pattern_type']}/{hit['pattern_name']}"
        # If a mapping exists, use it; otherwise, default to a key where '/' is replaced by '_'
        hit['motif_name'] = motif_mapping.get(key, key.replace('/', '_'))
    
    print("Mapped Motif Hits on Yeast Genome:")
    print("Number of hits:", len(hits))
    
    # Print the first few hits as a check (including species info)
    for hit in hits[:5]:
        print(f"{hit['chrom']}:{hit['genome_start']}-{hit['genome_end']} "
              f"(strand {hit['strand']}) | {hit['species']} - {hit['pattern_type']}/{hit['pattern_name']} "
              f"example_idx={hit['example_idx']} seqlet_index={hit['seqlet_index']}")
    
    # Finally, write all hits to a BED file.
    output_bed = f"{output_dir}/tf_modisco_motif_hits.bed"
    write_hits_to_bed(hits, output_bed)
    print(f"Wrote all TF-MoDISco motif hits to {output_bed}")
