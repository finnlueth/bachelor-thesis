import multiprocessing
import os
import pickle
from dataclasses import InitVar, dataclass, field, fields
from functools import partial
from io import BytesIO

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import ClassLabel, Dataset, Image, Value, load_dataset
from datasets.utils.file_utils import xopen

from src.data import foldseek
from src.data.mdcath_processing import (
    download_open,
    download_process_pipeline,
    extract_mdcath_information,
    generate_fasta,
    generate_mdcath_coordiante_pdbs,
    generate_pssms,
    process_pipeline,
    replace_coordinates_in_pdb,
    replace_coordinates_in_pdbs,
    rmsd_align,
    save_fasta,
    save_PDBs,
    translate_pdb_to_3di,
)

FILE_PATHS = {
    "3Di": "../tmp/data/3Di",
    "trajectories": "../tmp/data/mdCATH/data",
    "mdCATH": "../tmp/data/mdCATH",
    "PDB": "../tmp/data/PDB",
    "pssm": "../tmp/data/pssm",
    "cache": "../tmp/cache",
}

for x in FILE_PATHS.values():
    os.makedirs(x, exist_ok=True)


PROCESSING_CONFIG = {
    "temperatures": ["320", "348"],
    "replicas": ["0", "1"],
}
