#!/usr/bin/env python3

def read_fasta_ids(fasta_file):
    """Read domain IDs from FASTA file."""
    domain_ids = set()
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Remove '>' and any whitespace, take first part
                domain_id = line[1:].strip().split()[0]
                domain_ids.add(domain_id)
    return domain_ids

def filter_lookup(lookup_file, output_file, domain_ids):
    """Filter lookup file to only include entries in domain_ids."""
    with open(lookup_file, 'r') as f:
        lines = [line.strip() for line in f if line.split('\t')[0] in domain_ids]
    print(*lines, sep='\n')
    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    

def main():
    fasta_file = "./data/scope40_sequences_short.fasta"
    lookup_file = "./data/scop_lookup.fix.tsv"
    output_file = "./data/scop_lookup_short.fix.tsv"
    
    # Read domain IDs from FASTA file
    domain_ids = read_fasta_ids(fasta_file)
    
    # Filter lookup file
    filter_lookup(lookup_file, output_file, domain_ids)

if __name__ == "__main__":
    main()
