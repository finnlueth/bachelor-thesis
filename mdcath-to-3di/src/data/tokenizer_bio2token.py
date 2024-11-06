import torch
import os
import shlex
import argparse
import tempfile
import typing as T
import subprocess as sp
from Bio import SeqIO, SeqRecord, Seq


def get_bio2token_sequences_from_memory(pdb_files: T.List[str], bio2token_path="bio2token/scripts"):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_paths = []
        for i, content in enumerate(pdb_files):
            pdb_path = os.path.join(tmpdir, f"file_{i}.pdb")
            with open(pdb_path, 'w') as file:
                file.write(content)
            pdb_paths.append(pdb_path)

        pdb_file_string = " ".join(pdb_paths)
        pdb_dir_name = hash(pdb_file_string)
        
        for _ in pdb_str:
        
        BASE_COMMAND = f"python {bio2token_path}/run_bio2token.py --pdb {pdb_file_string}"
