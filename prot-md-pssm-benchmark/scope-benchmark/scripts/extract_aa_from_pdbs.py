#!/usr/bin/env python3

import os
import json
from Bio import PDB
import Bio
import argparse


AA2ONE = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def extract_sequence_from_pdb(pdb_file):
    """Extract amino acid sequence from a PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]

        sequence = ""
        for chain in model:
            for residue in chain:
                aa = AA2ONE.get(residue.get_resname(), 'X')
                sequence += aa

        return sequence
    except Exception as e:
        print(f"Error processing {pdb_file}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract amino acid sequences from PDB files')
    parser.add_argument('--pdb_dir', type=str, default='../../foldseek-analysis/scopbenchmark/data/scop-pdb/',
                        help='Directory containing PDB files')
    parser.add_argument('--output', type=str, default='sequences.json',
                        help='Output JSON file')

    args = parser.parse_args()

    if not os.path.exists(args.pdb_dir):
        print(f"Error: Directory {args.pdb_dir} does not exist!")
        return

    sequences = {}
    for filename in os.listdir(args.pdb_dir):
        pdb_path = os.path.join(args.pdb_dir, filename)
        sequence = extract_sequence_from_pdb(pdb_path)

        if sequence:
            sequences[filename] = sequence
            print(f"Processed: {filename}")

    with open(args.output, 'w') as json_out:
        json.dump(sequences, json_out, indent=2)

if __name__ == "__main__":
    main()
