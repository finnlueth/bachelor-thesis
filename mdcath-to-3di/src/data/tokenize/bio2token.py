import torch
import os
import shlex
import argparse
import tempfile
import typing as T
import subprocess as sp
from Bio import SeqIO, SeqRecord, Seq
from src.data.tokenize.tokenize_base import BaseTokenizer, save_to_h5


class Bio2TokenTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

    def load_model(self):
        # Load the Bio2Token model here
        pass

    def tokenize(self, pdb_files: T.List[str]) -> T.List[str]:
        # Implement the tokenization logic for Bio2Token
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_paths = []
            for i, content in enumerate(pdb_files):
                pdb_path = os.path.join(tmpdir, f"file_{i}.pdb")
                with open(pdb_path, 'w') as file:
                    file.write(content)
                pdb_paths.append(pdb_path)

            # Implement specific logic for Bio2Token using pdb_paths
            # Convert results to a format suitable for HDF5
            return []  # Replace with actual data