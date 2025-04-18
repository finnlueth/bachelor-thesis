#!/usr/bin/env python3

import os
import sys
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBIO import Select

class BackboneSelect(Select):
    """Select only backbone atoms (N, CA, C, O)"""
    def accept_atom(self, atom):
        return atom.get_name() in ['N', 'CA', 'C', 'O']

def process_pdb_files(input_dir, output_dir):
    """
    Process all PDB files in input_dir, keeping only backbone atoms,
    and save them to output_dir.
    
    Args:
        input_dir (str): Directory containing input PDB files
        output_dir (str): Directory to save processed PDB files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize PDB parser
    parser = PDBParser()
    
    # Process each PDB file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdb'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Parse the structure
                structure = parser.get_structure('protein', input_path)
                
                # Save only backbone atoms
                io = PDBIO()
                io.set_structure(structure)
                io.save(output_path, BackboneSelect())
                
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_pdb_backbones.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_pdb_files(input_dir, output_dir)
