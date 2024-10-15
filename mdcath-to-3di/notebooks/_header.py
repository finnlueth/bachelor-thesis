import multiprocessing
import os
import pickle
from dataclasses import InitVar, dataclass, field, fields
from functools import partial
from io import BytesIO

import h5py
import pyarrow as pa
from datasets import ClassLabel, Dataset, Image, Value, load_dataset
from datasets.utils.file_utils import xopen

from src.data import foldseek
from src.data.mdcath_processing import (
    download,
    download_process_pipeline,
    extract_mdcath_information,
    generate_fasta,
    generate_mdcath_coordiante_pdbs,
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
    # "trajectories": "../tmp/data/trajectories/",
    # "trajectories": FILE_PATHS["mdCATH"] + "/data",
    "trajectories": "./tmp/data/trajectories/data",
    "mdCATH": "../tmp/data/mdCATH",
    "PDBs": "../tmp/data/PDBs",
    "pssm": "../tmp/data/pssm",
}

for x in FILE_PATHS.values():
    os.makedirs(x, exist_ok=True)


PROCESSING_CONFIG = {
    "temperatures": ["320"],
    "replicas": ["0"],
}
