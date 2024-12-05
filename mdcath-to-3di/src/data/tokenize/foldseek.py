# # Adapted from https://github.com/samsledje/D-SCRIPT/blob/main/dscript/foldseek.py

import os
import shlex
import tempfile
import subprocess as sp
from Bio import SeqRecord, Seq

import logging
import typing as T
from .trajectory_tokenizer import TrajectoryTokenizer


def get_3di_sequences_from_memory(pdb_files: T.List[str], foldseek_path="foldseek"):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_paths = []
        for i, content in enumerate(pdb_files):
            pdb_path = os.path.join(tmpdir, f"file_{i:05d}.pdb")
            with open(pdb_path, "w") as file:
                file.write(content)
            pdb_paths.append(pdb_path)

        pdb_file_string = " ".join(pdb_paths)
        pdb_dir_name = hash(pdb_file_string)
        db_name = f"{tmpdir}/{pdb_dir_name}"

        FSEEK_BASE_CMD = f"{foldseek_path} createdb {pdb_file_string} {db_name}"
        proc = sp.Popen(shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        out, err = proc.communicate()

        seq_file_path = f"{db_name}_ss"
        lookup_file_path = f"{db_name}.lookup"

        if os.path.exists(seq_file_path):
            with open(seq_file_path, "r") as seq_file:
                seqs = [line.strip().strip("\x00") for line in seq_file]
                seqs.remove("")
        else:
            raise FileNotFoundError(f"No sequence file found at {seq_file_path}")

        if os.path.exists(lookup_file_path):
            with open(lookup_file_path, "r") as name_file:
                names = [line.strip().split()[1].split(".")[0] for line in name_file]
        else:
            raise FileNotFoundError(f"No lookup file found at {lookup_file_path}")
        return names, seqs


class FoldSeekTokenizer(TrajectoryTokenizer):
    def __init__(self, tokenizer_name: str):
        super().__init__(tokenizer_name)
        self.model = self.load_model()

    def load_model(self):
        return get_3di_sequences_from_memory

    def tokenize(self, pdb_files: T.List[str]) -> T.List[str]:
        return self.model(pdb_files)

    def detokenize(self, tokens: T.List[str]) -> T.List[str]:
        raise NotImplementedError("FoldSeekTokenizer does not support detokenization.")

