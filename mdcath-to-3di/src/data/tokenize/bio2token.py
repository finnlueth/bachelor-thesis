import torch
import os
import shlex
import argparse
import tempfile
import typing as T
import subprocess as sp
from Bio import SeqIO, SeqRecord, Seq


def get_bio2token_sequences_from_memory(pdb_files: T.List[str], bio2token_path="bio2token/scripts"):
    pass