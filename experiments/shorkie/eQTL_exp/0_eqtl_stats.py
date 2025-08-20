import sys
import re
import random

def map_chromosome_to_roman(chromosome):
    roman_mapping = {
        'chromosome1': 'I',
        'chromosome2': 'II',
        'chromosome3': 'III',
        'chromosome4': 'IV',
        'chromosome5': 'V',
        'chromosome6': 'VI',
        'chromosome7': 'VII',
        'chromosome8': 'VIII',
        'chromosome9': 'IX',
        'chromosome10': 'X',
        'chromosome11': 'XI',
        'chromosome12': 'XII',
        'chromosome13': 'XIII',
        'chromosome14': 'XIV',
        'chromosome15': 'XV',
        'chromosome16': 'XVI',
    }
    return roman_mapping.get(chromosome, chromosome)

def parse_gtf(gtf_file):
    """
    Parse the GTF file to extract:
      1) TSS positions (start of each gene).
      2) Coding intervals (e.g., from 'CDS', 'exon').

    Returns:
      tss_data: dict { gene_id: (chrom, tss_start, strand) }
      coding_intervals: list of tuples (chrom, start, end)
    """
    tss_data = {}
    coding_intervals = []

    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            chrom = fields[0]
            feature_type = fields[2].lower()  # e.g. "gene", "cds", "exon"
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]

            # Attempt to parse gene_id
            gene_match = re.search(r'gene_id\s+"([^"]+)"', attributes)
            gene_id = gene_match.group(1) if gene_match else None

            # If feature is a gene, record TSS
            if feature_type == 'gene' and gene_id:
                if strand == '+':
                    tss_data[gene_id] = (chrom, start, strand)
                else:
                    tss_data[gene_id] = (chrom, end, strand)

            # If feature is CDS or exon, store in coding_intervals
            if feature_type in ['cds', 'exon']:
                coding_intervals.append((chrom, start, end))

    return tss_data, coding_intervals


def is_in_coding_region(chrom, pos, coding_intervals):
    """Return True if (chrom,pos) is inside any coding interval."""
    for (c, s, e) in coding_intervals:
        if c == chrom and s <= pos <= e:
            return True
    return False


def parse_gvcf(gvcf_file, coding_intervals, af_threshold=0.05):
    """
    Parse the gVCF, extracting (chrom, pos, ref, alt) and AF.
    Skip variants that:
      - Are in coding regions
      - Have AF < af_threshold
    Return dict: {(chrom, pos, ref, alt): {...}}.
    """
    variants = {}
    with open(gvcf_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt = fields[4]
            qual = fields[5]
            fltr = fields[6]
            info = fields[7]

            # Skip if in coding region
            if is_in_coding_region(chrom, pos, coding_intervals):
                continue

            # Parse AF
            af_value = None
            info_parts = info.split(';')
            info_dict = {}
            for kv in info_parts:
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    info_dict[k] = v

            if 'AF' in info_dict:
                try:
                    af_value = float(info_dict['AF'])
                except ValueError:
                    af_value = None
            else:
                # fallback if AC and AN
                if 'AC' in info_dict and 'AN' in info_dict:
                    try:
                        ac = float(info_dict['AC'])
                        an = float(info_dict['AN'])
                        if an > 0:
                            af_value = ac / an
                        else:
                            af_value = 0.0
                    except ValueError:
                        af_value = None

            if af_value is None:
                continue
            if af_value < af_threshold:
                continue

            variants[(chrom, pos, ref, alt)] = {
                'AF': af_value,
                'INFO': info,
                'QUAL': qual,
                'FILTER': fltr
            }
    return variants


def parse_gwas(gwas_file):
    """
    Parse the GWAS file (contains positive eQTLs).
    Columns required: Chr, ChrPos, Reference, Alternate, Pheno
    Return a list of dicts: { chrom, pos, ref, alt, gene }
    """
    eqtls = []
    with open(gwas_file, 'r') as f:
        header_line = f.readline().strip()
        if not header_line:
            raise ValueError("GWAS file is empty or missing a header line.")
        header = header_line.split()
        col_map = {h: i for i, h in enumerate(header)}

        required_cols = ["Chr", "ChrPos", "Reference", "Alternate", "Pheno"]
        for rc in required_cols:
            if rc not in col_map:
                raise ValueError(f"Missing required column '{rc}' in header: {header}")

        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.strip().split('\t')
            if len(fields) < len(header):
                # skip malformed
                continue

            chrom = fields[col_map['Chr']]
            pos_str = fields[col_map['ChrPos']]
            ref = fields[col_map['Reference']]
            alt = fields[col_map['Alternate']]
            gene = fields[col_map['Pheno']]

            if not pos_str.isdigit():
                continue
            pos = int(pos_str)

            eqtls.append({
                'chrom': chrom,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                'gene': gene
            })
    return eqtls


def compute_distance_to_phenotype_tss(eqtl, tss_data):
    """
    Given eqtl['gene'], find that gene's TSS in tss_data.
    Return abs distance or None if gene not found.
    """
    gene_id = eqtl['gene']
    if gene_id not in tss_data:
        return None
    (gene_chrom, gene_tss, strand) = tss_data[gene_id]
    return abs(eqtl['pos'] - gene_tss)


def pick_random_gene_within_distance(variant_chrom, variant_pos,
                                     desired_distance, tolerance,
                                     tss_data, max_tries=50000):
    """
    Randomly select a gene from tss_data that is on the same chromosome
    as 'variant_chrom' AND whose TSS is within desired_distance ± tolerance
    from variant_pos.

    Returns: (gene_chrom, gene_id, actual_dist) or None.
    """
    # Filter genes to include only those on the same chromosome
    mapped_variant_chrom = map_chromosome_to_roman(variant_chrom)
    same_chr_genes = [gene_id for gene_id, (chr_g, _, _) in tss_data.items()
                      if chr_g == mapped_variant_chrom]
    # print(f"\tlen(same_chr_genes) = {len(same_chr_genes)}")
    if not same_chr_genes:
        return None

    for idx in range(max_tries):
        gene_id = random.choice(same_chr_genes)
        _, tss, _ = tss_data[gene_id]
        
        actual_dist = abs(variant_pos - tss)
        if abs(actual_dist - desired_distance) <= tolerance:
            return (mapped_variant_chrom, gene_id, actual_dist)

    return None


def main(gwas_file, gtf_file):
    """
    1) Parse GTF -> TSS + coding intervals
    2) Parse gVCF (non-coding, AF>0.05)
    3) Parse GWAS -> positive eQTLs
    4) For each positive eQTL:
       - compute distance to TSS
       - find up to 'negatives_per_positive' negative variants:
            * same REF, ALT
            * not in positive set
            * pick random gene with TSS ~ pos_dist
       - store them
    5) Write negative eQTL file and summary
    """

    print("Parsing GTF...")
    tss_data, coding_intervals = parse_gtf(gtf_file)
    print(f"Found {len(tss_data)} TSS entries.")

    print("Parsing GWAS...")
    positive_eqtls = parse_gwas(gwas_file)
    print(f"Loaded {len(positive_eqtls)} positive eQTLs.")

    # Compute distance to TSS for positives
    for eq in positive_eqtls:
        eq['pos_dist_to_tss'] = compute_distance_to_phenotype_tss(eq, tss_data)

    print("Positive eQTLs with TSS distances:")
    total_positives = len(positive_eqtls)
    eqtl_dist_16384 = 0
    for eqtl in positive_eqtls:
        p_chrom = eqtl['chrom']
        p_pos   = eqtl['pos']
        p_ref   = eqtl['ref']
        p_alt   = eqtl['alt']
        p_gene  = eqtl['gene']
        p_dist  = eqtl['pos_dist_to_tss']
        # print(f"\t{p_chrom} {p_pos} {p_ref} {p_alt} {p_gene} {p_dist}")
        if p_dist is not None and p_dist >= 16384:
            eqtl_dist_16384 += 1
    print(f"Total positives: {total_positives}")
    print(f"Positives with distance >= 16384: {eqtl_dist_16384}")   


if __name__ == "__main__":
    """
    Example usage:
      python find_negative_eqtls.py positive_gwas.tsv variants.g.vcf annotation.gff output_negatives.tsv

    Optionally adjust the distance tolerance or negative count directly in the code
    or pass them via command-line args if desired.
    """
    # if len(sys.argv) < 5:
    #     print("Usage: {} <gwas_file> <gvcf_file> <gff_file> <output_file>".format(sys.argv[0]))
    #     sys.exit(1)

    # gwas_file = sys.argv[1]
    # gvcf_file = sys.argv[2]
    # gff_file = sys.argv[3]
    # output_file = sys.argv[4]

    gwas_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/neg_eQTLS/intersected_data_CIS.tsv"
    gtf_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/neg_eQTLS/GCA_000146045_2.59.gtf"
    negative_eqtl_output = "results/negative_eqtls.tsv"
    summary_output = "results/negative_summary.tsv"

    # Hard-coded or configurable
    DISTANCE_TOLERANCE = 250
    NEGATIVES_PER_POSITIVE = 60

    main(gwas_file, gtf_file)
