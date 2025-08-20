#!/bin/bash

# Check if a file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <fasta_file>"
    exit 1
fi

# Count lowercase nucleotides
lowercase_count=$(grep -v '^>' "$1" | tr -cd 'acgt' | wc -c)

echo "Number of lowercase nucleotides: $lowercase_count"