import sys
import os
import re
import random
import argparse
import pandas as pd

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
    """Return True if (chrom, pos) is inside any coding interval."""
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
                # fallback if AC and AN are present
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

def parse_gwas(gwas_file, dataset="caudal"):
    """
    Parse the GWAS/eQTL file (contains positive eQTLs).
    Return a list of dicts: { chrom, pos, ref, alt, gene }
    """
    eqtls = []
    
    if dataset == "renganaath":
        df = pd.read_csv(gwas_file)
        print(f"Loaded {len(df)} variants.")
        for _, row in df.iterrows():
            chrom = str(row['chr'])
            if chrom.startswith('chr'):
                chrom = chrom[3:]
            chrom = map_chromosome_to_roman(chrom)
            try:
                pos = int(row['pos'])
            except ValueError:
                continue
            ref = str(row['ref']).strip()
            alt = str(row['alt']).strip()
            gene = str(row['gene']).strip()
            eqtls.append({'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt, 'gene': gene})
        return eqtls
    else:
        # caudal or kita
        if dataset == "caudal":
            required_cols = ["Chr", "ChrPos", "Reference", "Alternate", "Pheno"]
            gene_col = "Pheno"
        elif dataset == "kita":
            required_cols = ["Chr", "ChrPos", "Reference", "Alternate", "#Gene"]
            gene_col = "#Gene"
            
        with open(gwas_file, 'r') as f:
            header_line = f.readline().strip()
            if not header_line:
                raise ValueError("GWAS file is empty or missing a header line.")
            header = header_line.split()
            col_map = {h: i for i, h in enumerate(header)}

            for rc in required_cols:
                if rc not in col_map:
                    raise ValueError(f"Missing required column '{rc}' in header: {header}")

            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < len(header):
                    continue

                chrom = fields[col_map['Chr']]
                pos_str = fields[col_map['ChrPos']]
                ref = fields[col_map['Reference']]
                alt = fields[col_map['Alternate']]
                gene = fields[col_map[gene_col]]

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
    Return absolute distance or None if gene not found.
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
    mapped_variant_chrom = map_chromosome_to_roman(variant_chrom)
    same_chr_genes = [gene_id for gene_id, (chr_g, _, _) in tss_data.items()
                      if chr_g == mapped_variant_chrom]
    if not same_chr_genes:
        return None

    for idx in range(max_tries):
        gene_id = random.choice(same_chr_genes)
        _, tss, _ = tss_data[gene_id]
        actual_dist = abs(variant_pos - tss)
        if abs(actual_dist - desired_distance) <= tolerance:
            return (mapped_variant_chrom, gene_id, actual_dist)

    return None

def main(gwas_file, gvcf_file, gtf_file,
         output_prefix,
         dataset="caudal",
         distance_tolerance=10,
         negatives_per_positive=1,
         iterations=10):
    """
    Process to generate negative eQTL sets.
    1) Parse GTF -> TSS and coding intervals.
    2) Parse gVCF (non-coding, AF>0.05).
    3) Parse GWAS -> positive eQTLs.
    4) For each iteration (negative set):
         For each positive eQTL:
           - Compute distance to TSS.
           - Find 1 negative variant (using the same REF/ALT from gVCF) that is not a known positive,
             and select a gene whose TSS is within 'distance_tolerance' of the positive's distance.
           - If the candidate search (with tolerance=distance_tolerance) fails (i.e. max_tries reached),
             try again with tolerance=200.
           - Ensure that a negative candidate is not repeated within the iteration.
         - Write the negative set details and a summary.
    """
    print("Parsing GTF...")
    tss_data, coding_intervals = parse_gtf(gtf_file)
    print(f"Found {len(tss_data)} TSS entries.")

    print("Parsing gVCF...")
    variant_dict = parse_gvcf(gvcf_file, coding_intervals, af_threshold=0.05)
    print(f"Retained {len(variant_dict)} variants (MAF>0.05, non-coding).")

    print("Parsing GWAS...")
    positive_eqtls = parse_gwas(gwas_file, dataset=dataset)
    print(f"Loaded {len(positive_eqtls)} positive eQTLs.")

    # Compute TSS distance for each positive eQTL
    for eq in positive_eqtls:
        eq['pos_dist_to_tss'] = compute_distance_to_phenotype_tss(eq, tss_data)

    # Build sets for known positive variants
    positive_variant_keys = set((eq['chrom'], eq['pos'], eq['ref'], eq['alt']) for eq in positive_eqtls)
    positive_triplets = set((eq['chrom'], eq['pos'], eq['gene']) for eq in positive_eqtls)

    # Group gVCF variants by (ref, alt)
    grouped_by_refalt = {}
    for (chrom, pos, ref, alt), data in variant_dict.items():
        key = (ref, alt)
        if key not in grouped_by_refalt:
            grouped_by_refalt[key] = []
        grouped_by_refalt[key].append((chrom, pos, ref, alt))

    # Repeat the negative selection process for the specified number of iterations
    for iteration in range(1, iterations + 1):
        print(f"\n--- Generating negative set iteration {iteration} ---")
        negative_results = []
        summary_data = []
        # Set to track negatives that have already been selected in this iteration
        used_negatives = set()

        for eqtl in positive_eqtls:
            p_chrom = eqtl['chrom']
            p_pos   = eqtl['pos']
            p_ref   = eqtl['ref']
            p_alt   = eqtl['alt']
            p_gene  = eqtl['gene']
            p_dist  = eqtl['pos_dist_to_tss']

            # If TSS distance is unavailable, add summary entry with 0 negatives.
            if p_dist is None:
                summary_data.append({
                    'chrom': p_chrom, 'pos': p_pos, 'gene': p_gene,
                    'pos_dist': "NA", 'neg_count': 0, 'avg_neg_dist': "NA"
                })
                continue

            refalt_key = (p_ref, p_alt)
            if refalt_key not in grouped_by_refalt:
                summary_data.append({
                    'chrom': p_chrom, 'pos': p_pos, 'gene': p_gene,
                    'pos_dist': p_dist, 'neg_count': 0, 'avg_neg_dist': "NA"
                })
                continue

            # Build candidate list, excluding known positives
            candidate_list = []
            for (c_chrom, c_pos, c_ref, c_alt) in grouped_by_refalt[refalt_key]:
                if (c_chrom, c_pos, c_ref, c_alt) in positive_variant_keys:
                    continue
                candidate_list.append((c_chrom, c_pos, c_ref, c_alt))

            if not candidate_list:
                summary_data.append({
                    'chrom': p_chrom, 'pos': p_pos, 'gene': p_gene,
                    'pos_dist': p_dist, 'neg_count': 0, 'avg_neg_dist': "NA"
                })
                continue

            random.shuffle(candidate_list)
            negative_found = None
            for (neg_chrom, neg_pos, neg_ref, neg_alt) in candidate_list:
                # First attempt using the provided tolerance (e.g. 100)
                gene_pick = pick_random_gene_within_distance(
                    variant_chrom=neg_chrom,
                    variant_pos=neg_pos,
                    desired_distance=p_dist,
                    tolerance=distance_tolerance,
                    tss_data=tss_data
                )
                # If no candidate found, retry with fallback tolerance=200
                if gene_pick is None:
                    gene_pick = pick_random_gene_within_distance(
                        variant_chrom=neg_chrom,
                        variant_pos=neg_pos,
                        desired_distance=p_dist,
                        tolerance=200,
                        tss_data=tss_data
                    )
                if gene_pick is None:
                    continue
                (chosen_chrom, chosen_gene, chosen_dist) = gene_pick
                if (neg_chrom, neg_pos, chosen_gene) in positive_triplets:
                    continue
                # Check if this negative candidate has already been used in this iteration
                if (neg_chrom, neg_pos, chosen_gene) in used_negatives:
                    continue

                negative_found = {
                    'pos_chrom': p_chrom,
                    'pos_pos': p_pos,
                    'pos_ref': p_ref,
                    'pos_alt': p_alt,
                    'pos_gene': p_gene,
                    'pos_distance_to_tss': p_dist,
                    'neg_chrom': neg_chrom,
                    'neg_pos': neg_pos,
                    'neg_ref': neg_ref,
                    'neg_alt': neg_alt,
                    'neg_gene': chosen_gene,
                    'neg_distance_to_tss': chosen_dist
                }
                used_negatives.add((neg_chrom, neg_pos, chosen_gene))
                break  # Only one negative per positive is selected

            if negative_found:
                negative_results.append(negative_found)
                summary_data.append({
                    'chrom': p_chrom,
                    'pos': p_pos,
                    'gene': p_gene,
                    'pos_dist': p_dist,
                    'neg_count': 1,
                    'avg_neg_dist': round(negative_found['neg_distance_to_tss'], 2)
                })
            else:
                summary_data.append({
                    'chrom': p_chrom,
                    'pos': p_pos,
                    'gene': p_gene,
                    'pos_dist': p_dist,
                    'neg_count': 0,
                    'avg_neg_dist': "NA"
                })

        neg_output_file = f"{output_prefix}_set{iteration}.tsv"
        summ_output_file = f"{output_prefix}_summary_set{iteration}.tsv"

        print(f"Writing negative eQTL details to {neg_output_file}...")
        with open(neg_output_file, 'w') as out:
            header_cols = [
                'pos_chrom','pos_pos','pos_ref','pos_alt','pos_gene','pos_distance_to_tss',
                'neg_chrom','neg_pos','neg_ref','neg_alt','neg_gene','neg_distance_to_tss'
            ]
            out.write("\t".join(header_cols) + "\n")
            for row in negative_results:
                out_vals = [str(row[col]) for col in header_cols]
                out.write("\t".join(out_vals) + "\n")

        print(f"Writing summary to {summ_output_file}...")
        with open(summ_output_file, 'w') as out:
            summ_header = [
                'pos_chrom','pos_pos','pos_gene','pos_distance_to_tss',
                'neg_count','avg_neg_dist'
            ]
            out.write("\t".join(summ_header) + "\n")
            for sd in summary_data:
                row_vals = [
                    str(sd['chrom']), str(sd['pos']), str(sd['gene']),
                    str(sd['pos_dist']), str(sd['neg_count']), str(sd['avg_neg_dist'])
                ]
                out.write("\t".join(row_vals) + "\n")

        print(f"Iteration {iteration} complete. Found {len(negative_results)} negative eQTLs, with {len(summary_data)} summary rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distance-matched negative eQTLs.")
    parser.add_argument("--gwas_file", required=True, help="Input positive eQTLs file")
    parser.add_argument("--gvcf_file", required=True, help="Input gVCF file for non-coding variants")
    parser.add_argument("--gtf_file", required=True, help="Input GTF file for TSS mapping")
    parser.add_argument("--output_prefix", required=True, help="Output file prefix (e.g. results/negative_eqtls)")
    parser.add_argument("--dataset", choices=["caudal", "kita", "renganaath"], default="caudal", help="Dataset format parsing")
    parser.add_argument("--distance_tolerance", type=int, default=100, help="Tolerance for matching TSS distance")
    parser.add_argument("--negatives_per_positive", type=int, default=1, help="Number of negative variants per positive variant")
    parser.add_argument("--iterations", type=int, default=4, help="Number of negative sets to generate")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    main(args.gwas_file, args.gvcf_file, args.gtf_file,
         args.output_prefix,
         dataset=args.dataset,
         distance_tolerance=args.distance_tolerance,
         negatives_per_positive=args.negatives_per_positive,
         iterations=args.iterations)
