# # Adapted from https://github.com/samsledje/D-SCRIPT/blob/main/dscript/foldseek.py

# import torch
# import os
# import shlex
# import argparse
# import tempfile
# import subprocess as sp
# from Bio import SeqIO, SeqRecord, Seq

# # from src.utils import logging
import typing as T
from .tokenize_base import BaseTokenizer


class FoldSeekTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        # self.model = self.load_model()

    def load_model(self):
        pass

    def tokenize(self, pdb_files: T.List[str]) -> T.List[str]:
        return ['a', 'b']

    def detokenize(self, tokens: T.List[str]) -> T.List[str]:
        pass

        # # Implement the tokenization logic for FoldSeek
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     pdb_paths = []


#             for i, content in enumerate(pdb_files):
#                 pdb_path = os.path.join(tmpdir, f"file_{i}.pdb")
#                 with open(pdb_path, 'w') as file:
#                     file.write(content)
#                 pdb_paths.append(pdb_path)

#             seq_records = get_3di_sequences_from_file(pdb_paths)
#             # Convert seq_records to a format suitable for HDF5
#             return [str(record.seq) for record in seq_records.values()]


# def get_3di_sequences_from_file(pdb_files: T.List[str], foldseek_path="foldseek"):
#     pdb_file_string = " ".join([str(p) for p in pdb_files])
#     pdb_dir_name = hash(pdb_file_string)

#     with tempfile.TemporaryDirectory() as tmpdir:
#         FSEEK_BASE_CMD = f"{foldseek_path} createdb {pdb_file_string} {tmpdir}/{pdb_dir_name}"
#         # log(FSEEK_BASE_CMD)
#         proc = sp.Popen(
#             shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE
#         )
#         out, err = proc.communicate()

#         with open(f"{tmpdir}/{pdb_dir_name}_ss", "r") as seq_file:
#             seqs = [i.strip().strip("\x00") for i in seq_file]

#         with open(f"{tmpdir}/{pdb_dir_name}.lookup", "r") as name_file:
#             names = [i.strip().split()[1].split(".")[0] for i in name_file]

#         seq_records = {
#             n: SeqRecord.SeqRecord(Seq.Seq(s), id=n, description=n)
#             for (n, s) in zip(names, seqs)
#         }

#         return seq_records


# def get_3di_sequences_from_memory(pdb_files: T.List[str], foldseek_path="foldseek"):
#     with tempfile.TemporaryDirectory() as tmpdir:
#         pdb_paths = []
#         for i, content in enumerate(pdb_files):
#             pdb_path = os.path.join(tmpdir, f"file_{i}.pdb")
#             with open(pdb_path, 'w') as file:
#                 file.write(content)
#             pdb_paths.append(pdb_path)

#         pdb_file_string = " ".join(pdb_paths)
#         pdb_dir_name = hash(pdb_file_string)
#         db_name = f"{tmpdir}/{pdb_dir_name}"

#         FSEEK_BASE_CMD = f"{foldseek_path} createdb {pdb_file_string} {db_name}"
#         # log(FSEEK_BASE_CMD)
#         proc = sp.Popen(
#             shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE
#         )
#         out, err = proc.communicate()

#         seq_file_path = f"{db_name}_ss"
#         lookup_file_path = f"{db_name}.lookup"

#         if os.path.exists(seq_file_path):
#             with open(seq_file_path, "r") as seq_file:
#                 seqs = [line.strip().strip("\x00") for line in seq_file]
#                 seqs.remove('')
#         else:
#             raise FileNotFoundError(f"No sequence file found at {seq_file_path}")

#         if os.path.exists(lookup_file_path):
#             with open(lookup_file_path, "r") as name_file:
#                 names = [line.strip().split()[1].split(".")[0] for line in name_file]
#         else:
#             raise FileNotFoundError(f"No lookup file found at {lookup_file_path}")

#         # seq_records = {
#         #     n: SeqRecord.SeqRecord(Seq.Seq(s), id=n, description="")
#         #     for (n, s) in zip(names, seqs)
#         # }
#         # return seq_records

#         return seqs


# def translate_pdb_to_3di(pbds: dict) -> dict:
#     """
#     Translates a dictionary of PDB files into 3Di sequences.

#     Args:
#         pbds (dict): A dictionary where keys are identifiers and values are lists of PDB file paths.

#     Returns:
#         dict: A dictionary where keys are the same identifiers and values are the corresponding 3DI sequences.
#     """
#     items = {}
#     for key, values in pbds.items():
#         items[key] = get_3di_sequences_from_memory(pdb_files=values)
#     return items


# def generate_fasta(extraced_traj: dict, processed_3Di: dict) -> str:
#     """
#     Generates a FASTA file from the extracted trajectory and the processed 3Di sequences.

#     Args:
#         extraced_traj (dict): A dictionary containing the extracted trajectory data.
#                               It should have the following structure:
#                               {
#                                   "name": trajectory_name,
#                                   "seq": amino_acid_sequence,
#                                 }
#         processed_3Di (dict): A dictionary containing the processed 3Di sequences.
#                                 It should have the following structure:
#                                 {
#                                     "temp|replica": [sequence1, sequence2, ...],
#                                 }
#     Returns:
#         str: A string containing the FASTA formatted data.
#     """
#     items = []
#     for name, sequences in processed_3Di.items():
#         items.append(f">{name}|{extraced_traj['seq']}")
#         items.extend(sequences)
#     return "\n".join(items)


# def read_3Di_fasta(fasta: str) -> dict:
#     """
#     Read a FASTA file containing 3Di sequences.
#     Args:
#         fasta (str): The FASTA formatted string.
#     Returns:
#         dict: A dictionary containing the 3Di sequences.
#     """
#     items = {}
#     for line in fasta.split("\n"):
#         if line.startswith(">"):
#             name = line[1:]
#             items[name] = []
#         else:
#             items[name].append(line)
#     return items


# def fastas_to_hf_dataset(fasta: str, output_path: str) -> None:
#     """
#     Convert a FASTA file to an HDF5 dataset.
#     Args:
#         fasta (str): The FASTA formatted string.
#         output_path (str): The path to the output HDF5 file.
#     """
#     pass
