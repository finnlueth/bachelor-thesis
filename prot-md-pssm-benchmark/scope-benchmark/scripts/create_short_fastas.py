#!/usr/bin/env python3

import sys
import os
from Bio import SeqIO

def extract_ids_from_pssm(pssm_file):
    """Extract sequence IDs from PSSM profile TSV file."""
    ids = set()
    with open(pssm_file, 'r') as f:
        for line in f:
            if line.startswith('Query profile of sequence'):
                # Extract ID from line like "Query profile of sequence d12asa_"
                id = line.strip().split()[-1]
                ids.add(id)
    return ids

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_short_fastas.py <pssm_profile.tsv> <sequences.fasta>")
        sys.exit(1)

    pssm_file = sys.argv[1]
    fasta_file = sys.argv[2]

    # Extract IDs from PSSM profile
    target_ids = extract_ids_from_pssm(pssm_file)
    
    # Create output filename
    output_dir = os.path.dirname(fasta_file)
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_short.fasta")

    # Read and filter sequences
    matching_records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id in target_ids:
            matching_records.append(record)

    # Write filtered sequences without line breaks
    with open(output_file, 'w') as f:
        for record in matching_records:
            f.write(f">{record.id}\n{str(record.seq)}\n")
            
    print(f"Created {output_file} with {len(matching_records)} sequences")

if __name__ == "__main__":
    main()
