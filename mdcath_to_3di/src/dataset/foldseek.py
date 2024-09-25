# Adapted from https://github.com/samsledje/D-SCRIPT/blob/main/dscript/foldseek.py

import torch
import os
import shlex
import argparse
import tempfile
import typing as T
import subprocess as sp
from Bio import SeqIO, SeqRecord, Seq

from src.utils import logging

fold_vocab = {
    "D": 0,
    "P": 1,
    "V": 2,
    "Q": 3,
    "A": 4,
    "W": 5,
    "K": 6,
    "E": 7,
    "I": 8,
    "T": 9,
    "L": 10,
    "F": 11,
    "G": 12,
    "S": 13,
    "M": 14,
    "H": 15,
    "C": 16,
    "R": 17,
    "Y": 18,
    "N": 19,
    "X": 20,
}


def get_foldseek_onehot(n0, size_n0, fold_record, fold_vocab):
    """
    fold_record is just a dictionary {ensembl_gene_name => foldseek_sequence}
    """
    if n0 in fold_record:
        fold_seq = fold_record[n0]
        assert size_n0 == len(fold_seq)
        foldseek_enc = torch.zeros(
            size_n0, len(fold_vocab), dtype=torch.float32
        )
        for i, a in enumerate(fold_seq):
            assert a in fold_vocab
            foldseek_enc[i, fold_vocab[a]] = 1
        return foldseek_enc
    else:
        return torch.zeros(size_n0, len(fold_vocab), dtype=torch.float32)


def get_3di_sequences_from_file(pdb_files: T.List[str], foldseek_path="foldseek"):
    pdb_file_string = " ".join([str(p) for p in pdb_files])
    pdb_dir_name = hash(pdb_file_string)

    with tempfile.TemporaryDirectory() as tmpdir:
        FSEEK_BASE_CMD = f"{foldseek_path} createdb {pdb_file_string} {tmpdir}/{pdb_dir_name}"
        # log(FSEEK_BASE_CMD)
        proc = sp.Popen(
            shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE
        )
        out, err = proc.communicate()

        with open(f"{tmpdir}/{pdb_dir_name}_ss", "r") as seq_file:
            seqs = [i.strip().strip("\x00") for i in seq_file]

        with open(f"{tmpdir}/{pdb_dir_name}.lookup", "r") as name_file:
            names = [i.strip().split()[1].split(".")[0] for i in name_file]

        seq_records = {
            n: SeqRecord.SeqRecord(Seq.Seq(s), id=n, description=n)
            for (n, s) in zip(names, seqs)
        }

        return seq_records


def get_3di_sequences_from_memory(pdb_files: T.List[str], foldseek_path="foldseek"):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_paths = []
        for i, content in enumerate(pdb_files):
            pdb_path = os.path.join(tmpdir, f"file_{i}.pdb")
            with open(pdb_path, 'w') as file:
                file.write(content)
            pdb_paths.append(pdb_path)

        pdb_file_string = " ".join(pdb_paths)
        pdb_dir_name = hash(pdb_file_string)
        db_name = f"{tmpdir}/{pdb_dir_name}"
        
        FSEEK_BASE_CMD = f"{foldseek_path} createdb {pdb_file_string} {db_name}"
        # log(FSEEK_BASE_CMD)
        proc = sp.Popen(
            shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE
        )
        out, err = proc.communicate()
        
        seq_file_path = f"{db_name}_ss"
        lookup_file_path = f"{db_name}.lookup"

        if os.path.exists(seq_file_path):
            with open(seq_file_path, "r") as seq_file:
                seqs = [line.strip().strip("\x00") for line in seq_file]
                seqs.remove('')
        else:
            raise FileNotFoundError(f"No sequence file found at {seq_file_path}")

        if os.path.exists(lookup_file_path):
            with open(lookup_file_path, "r") as name_file:
                names = [line.strip().split()[1].split(".")[0] for line in name_file]
        else:
            raise FileNotFoundError(f"No lookup file found at {lookup_file_path}")

        # seq_records = {
        #     n: SeqRecord.SeqRecord(Seq.Seq(s), id=n, description="")
        #     for (n, s) in zip(names, seqs)
        # }
        # return seq_records
        
        return seqs
