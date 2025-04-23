import numpy as np
from typing import Dict
import re

def parse_pssm_tsv(file_path: str) -> Dict[str, np.ndarray]:
    """
    Parse a PSSM TSV file and return a dictionary with sequence names as keys and matrices as values.
    
    Args:
        file_path (str): Path to the PSSM TSV file
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with sequence names as keys and numpy arrays as values
    """
    result = {}
    current_matrix = []
    current_sequence = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('Query profile of sequence'):
                if current_sequence and current_matrix:
                    result[current_sequence] = np.array(current_matrix)
                    current_matrix = []
                
                current_sequence = re.search(r'Query profile of sequence (\w+)', line).group(1)
                i += 2
            else:
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 20:
                        current_matrix.append(values)
                except ValueError:
                    print(f"Error parsing line {i}: {line}")
            i += 1
        
        if current_sequence and current_matrix:
            result[current_sequence] = np.array(current_matrix)
    
    return result 