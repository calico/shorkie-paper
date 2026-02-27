import pandas as pd
import bisect
import random
from collections import defaultdict

def load_tss_data(tss_file):
    """
    Load TSS data into a dictionary by chromosome.
    Returns: { chrom: sorted_list_of_tss_entries }
    Each entry is a tuple (tss_position, strand, gene_name)

    Note: For '+' strand, the TSS is the start coordinate; for '-' strand, it's the end.
    """
    tss_by_chrom = defaultdict(list)
    with open(tss_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            gene_name = fields[3]
            strand = fields[5]
            
            # Determine TSS position based on strand:
            if strand == '+':
                tss_pos = start
            else:
                tss_pos = end
            
            tss_by_chrom[chrom].append((tss_pos, strand, gene_name))
    
    # Sort TSS entries by position for each chromosome.
    for chrom in tss_by_chrom:
        tss_by_chrom[chrom].sort(key=lambda x: x[0])
    
    return tss_by_chrom

def load_genome_sizes(genome_file):
    """
    Load genome sizes from a two-column file:
         chrom    size
    Returns a dictionary { chrom: size }
    """
    genome_sizes = {}
    with open(genome_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            fields = line.strip().split()
            chrom = fields[0]
            size = int(fields[1])
            genome_sizes[chrom] = size
    return genome_sizes

def find_closest_tss(tss_entries, pos):
    """
    Given a sorted list of TSS entries (each is (tss_position, strand, gene_name))
    and a genomic position pos, find the closest TSS and return:
        (tss_entry, signed_distance)
    where the signed distance is calculated as:
      - for '+' TSS: distance = (tss_position - pos)
      - for '-' TSS: distance = (pos - tss_position)
    
    Note: Using the motif midpoint to find the closest TSS is common,
    but if there are multiple TSS nearby (especially in dense clusters),
    this may not always pick the most biologically relevant one.
    """
    # Create a list of TSS positions for binary search.
    tss_positions = [entry[0] for entry in tss_entries]
    idx = bisect.bisect_left(tss_positions, pos)
    candidates = []
    if idx > 0:
        candidates.append(tss_entries[idx - 1])
    if idx < len(tss_entries):
        candidates.append(tss_entries[idx])
    
    best_entry = None
    best_distance = float('inf')
    for entry in candidates:
        distance = abs(entry[0] - pos)
        if distance < best_distance:
            best_distance = distance
            best_entry = entry
    
    if best_entry is None:
        return None, None
    
    tss_pos, tss_strand, gene_name = best_entry
    if tss_strand == '+':
        signed_distance = tss_pos - pos
    else:
        signed_distance = pos - tss_pos
    
    return best_entry, signed_distance

def calculate_closest_tss_distance(motif_bed, tss_file):
    """
    For each motif in the BED file, calculate the closest TSS distance.
    Returns a DataFrame with columns:
       ['motif_chrom', 'motif_start', 'motif_end', 'motif_name', 'motif_strand',
        'tss_position', 'tss_strand', 'gene_name', 'distance']
    
    Note: This approach uses the midpoint of each motif hit for the calculation.
    If the motif lengths vary or the binding footprint is not centered,
    there might be some bias.
    """
    # Load TSS data.
    tss_by_chrom = load_tss_data(tss_file)
    
    # Load motif data.
    motifs = []
    with open(motif_bed, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3]
            motif_strand = fields[5]
            motifs.append((chrom, start, end, name, motif_strand))
    
    results = []
    for motif in motifs:
        chrom, start, end, name, motif_strand = motif
        # Use the motif midpoint for the distance calculation.
        midpoint = (start + end) // 2  
        
        tss_entries = tss_by_chrom.get(chrom, [])
        if not tss_entries:
            continue
        
        best_entry, distance = find_closest_tss(tss_entries, midpoint)
        if best_entry is None:
            continue
        
        tss_pos, tss_strand, gene_name = best_entry
        
        results.append({
            'motif_chrom': chrom,
            'motif_start': start,
            'motif_end': end,
            'motif_name': name,
            'motif_strand': motif_strand,
            'tss_position': tss_pos,
            'tss_strand': tss_strand,
            'gene_name': gene_name,
            'distance': distance
        })
    
    return pd.DataFrame(results)

def calculate_background_tss_distance(motif_bed, tss_file, genome_file, num_background=1):
    """
    For each motif in the BED file, sample num_background random positions
    on the same chromosome (using genome sizes from genome_file) and calculate
    the closest TSS distance for each random position.
    
    Returns a DataFrame with columns:
      ['motif_chrom', 'motif_start', 'motif_end', 'motif_name', 'motif_strand',
       'bg_position', 'tss_position', 'tss_strand', 'gene_name', 'distance']
    
    Note: Random positions are sampled uniformly along the chromosome,
    which might not reflect the actual accessible regions.
    """
    # Load TSS data and genome sizes.
    tss_by_chrom = load_tss_data(tss_file)
    genome_sizes = load_genome_sizes(genome_file)
    
    # Load motif data.
    motifs = []
    with open(motif_bed, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3]
            motif_strand = fields[5]
            motifs.append((chrom, start, end, name, motif_strand))
    
    results = []
    for motif in motifs:
        chrom, start, end, name, motif_strand = motif
        # Check that genome size info is available.
        if chrom not in genome_sizes:
            continue
        chrom_length = genome_sizes[chrom]
        
        # Get TSS entries for this chromosome.
        tss_entries = tss_by_chrom.get(chrom, [])
        if not tss_entries:
            continue
        
        # For each motif, sample random positions.
        for i in range(num_background):
            random_position = random.randint(0, chrom_length - 1)
            best_entry, distance = find_closest_tss(tss_entries, random_position)
            if best_entry is None:
                continue
            tss_pos, tss_strand, gene_name = best_entry
            
            results.append({
                'motif_chrom': chrom,
                'motif_start': start,
                'motif_end': end,
                'motif_name': name,
                'motif_strand': motif_strand,
                'bg_position': random_position,
                'tss_position': tss_pos,
                'tss_strand': tss_strand,
                'gene_name': gene_name,
                'distance': distance
            })
    
    return pd.DataFrame(results)


def main():
    # exp_species = "strains_select"   # or "schizosaccharomycetales"
    exp_species = "schizosaccharomycetales"

    directory = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{exp_species}_gtf/gtf/"
    outfile = f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM__unseen_species/2_motif_to_tss_dist/results/{exp_species}_motif_hits/tss.bed"
    # Find files ending with .gtf (case sensitive; adjust if needed)
    gtf_files = glob.glob(os.path.join(directory, "*_2.59.gtf"))
    if not gtf_files:
        sys.exit("No GTF files found in the specified directory.")
    
    with open(outfile, 'w') as f:
        for gtf_file in gtf_files:
            # Get the base name of the file (without extension) to prepend to the chromosome.
            base = os.path.basename(gtf_file)
            name, _ = os.path.splitext(base)
            
            entries = process_gtf_file(gtf_file)
            for chrom, tss_start, tss_end, gene_name, dot, strand in entries:
                # Append the file name (without extension) to the chromosome, separated by an underscore.
                name = name.replace(".59", "")
                combined_chrom = f"{name}:chr{chrom}"
                print(f"{combined_chrom}\t{tss_start}\t{tss_end}\t{gene_name}\t{dot}\t{strand}")
                f.write(f"{combined_chrom}\t{tss_start}\t{tss_end}\t{gene_name}\t{dot}\t{strand}\n")

if __name__ == '__main__':
    main()
