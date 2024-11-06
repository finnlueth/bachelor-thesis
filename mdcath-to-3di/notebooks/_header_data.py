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

from src.data import tokenize_foldseek
from src.data.processing_mdcath import (
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



# todo read config