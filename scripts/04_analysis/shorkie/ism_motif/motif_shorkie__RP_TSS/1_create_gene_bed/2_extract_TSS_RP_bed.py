#!/usr/bin/env python3

import sys

def main(bed_in, bed_out=None, upstream=450, downstream=50):
    """
    bed_in:  Input BED file handle or path (if reading from stdin, use sys.stdin).
    bed_out: Output BED file handle or path (if writing to stdout, use sys.stdout).
    upstream, downstream: promoter definition around TSS.
    """
    
    # Handle input
    if bed_in == sys.stdin:
        lines = bed_in
    else:
        lines = open(bed_in, 'r')
    
    # Handle output
    out = sys.stdout if bed_out is None else open(bed_out, 'w')
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            # Skip header or empty lines
            continue
        
        # Parse BED fields:
        #   chrom, start, end, name, score, strand
        fields = line.split('\t')
        chrom  = fields[0]
        start  = int(fields[1])
        end    = int(fields[2])
        name   = fields[3] if len(fields) > 3 else '.'
        score  = fields[4] if len(fields) > 4 else '.'
        strand = fields[5] if len(fields) > 5 else '+'
        tss = (start + end) // 2
        if strand == '+':
            # TSS is 'start'
            region_start = tss - upstream
            region_end   = tss + downstream
        else:
            # TSS is 'end'
            region_start = tss - downstream
            region_end   = tss + upstream
        
        # Clip at zero if needed
        if region_start < 0:
            region_start = 0
        
        # Create new BED line
        new_fields = [
            chrom,
            str(region_start),
            str(region_end),
            name,
            score,
            strand
        ]
        out.write('\t'.join(new_fields) + '\n')
    
    # Close file handles if not stdin/stdout
    if bed_in != sys.stdin:
        lines.close()
    if bed_out is not None:
        out.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="""
    Read a 6-column BED file of gene TSS regions and output a new BED with
    450 bp upstream and 50 bp downstream of TSS for the + strand, and
    50 bp upstream and 450 bp downstream for the - strand.
    """)
    
    parser.add_argument(
        "-i", "--input",
        help="Input BED file (default: stdin).",
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help="Output BED file (default: stdout).",
        default=None
    )
    parser.add_argument(
        "--upstream", type=int, default=450,
        help="Number of bp upstream relative to TSS (default: 450)."
    )
    parser.add_argument(
        "--downstream", type=int, default=50,
        help="Number of bp downstream relative to TSS (default: 50)."
    )
    
    args = parser.parse_args()
    
    # Decide on stdin vs file
    if args.input is None:
        bed_in = sys.stdin
    else:
        bed_in = args.input
    
    # Decide on stdout vs file
    if args.output is None:
        bed_out = None
    else:
        bed_out = args.output
    
    main(bed_in, bed_out, args.upstream, args.downstream)
