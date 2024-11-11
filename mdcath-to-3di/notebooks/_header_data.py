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
from src.data.process_mdcath import (
    download_open,
    extract_mdcath_information,
    generate_mdcath_coordiante_pdbs,
    generate_pssms,
    replace_coordinates_in_pdb,
    replace_coordinates_in_pdbs,
    rmsd_align,
    save_PDBs,
)
# todo read config