#! /usr/bin/env python3
"""
python3 scripts/tokenize_script.py --input_path ./tmp/data/mdCATH/data/ --output_path ./tmp/output/tokenized/mdcath --tokenizers bio2token foldseek --dataset mdcath
"""

import argparse
import logging
import os
from multiprocessing import Pool, Lock, Manager
import h5py
import numpy as np

from src.data.load import (
    AtlasDataset,
    MDCATHDataset,
    MisatoDataset,
    TrajectoryDataset,
)
from src.data.tokenize import (
    Bio2TokenTokenizer,
    FoldSeekTokenizer,
    FoldToken4Tokenizer,
    TrajectoryTokenizer,
)
from src.utils.logging import setup_logging
from src.utils.utils import CodeTimer
from src.utils.errors import TrajectoryAlreadyProcessedError
import typing as T
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter("ignore", BiopythonDeprecationWarning)



def write_to_h5_file(h5_file_path, tokenizer_name, trajectory_name, data, lock):
    """Write data to H5 file with proper locking"""
    with lock:
        with h5py.File(h5_file_path, 'a') as h5_file:
            if tokenizer_name not in h5_file:
                tokenizer_group = h5_file.create_group(tokenizer_name)
            else:
                tokenizer_group = h5_file[tokenizer_name]
            
            if trajectory_name not in tokenizer_group:
                trajectory_group = tokenizer_group.create_group(trajectory_name)
            else:
                trajectory_group = tokenizer_group[trajectory_name]
            
            for i, item in enumerate(data):
                if isinstance(item, np.ndarray):
                    trajectory_group.create_dataset(f"array_{i}", data=item)
                else:
                    trajectory_group.create_dataset(f"string_{i}", data=np.bytes_(item))

def _process_trajectory(idx: int, dataset_class: TrajectoryDataset, tokenizer_classes: T.Dict[str, TrajectoryTokenizer], h5_file_path: str, lock_h5: Lock, lock_cache: Lock):
    try:
        print(f"Processing trajectory {idx} of {len(dataset_class)}")
        trajectory = dataset_class[idx]
        name = trajectory["name"]
        logging.info("Tokenizing trajectory %s", trajectory["name"])
        
        print(trajectory["name"])
        print(trajectory["trajectory_pdbs"].get("320_0")[5])

        for tokenizer_name, tokenizer_wrapper in tokenizer_classes.items():
            results = tokenizer_wrapper.tokenize(trajectory["structure"])
            print(results)
        #     results = tokenizer_wrapper.tokenize(trajectory["trajectories"])
        # write_to_h5_file(h5_file_path, tokenizer_name, trajectory["name"], results, lock_h5)
        dataset_class.use_trajectory_location(idx, lock_cache)
        logging.info("Finished tokenizing trajectory %s", trajectory["name"])
        del trajectory
        return name
    except TrajectoryAlreadyProcessedError as e:
        logging.error(e)
        return None

def process_dataset_parallel(dataset_class: TrajectoryDataset, tokenizer_classes: dict, output_path: str, num_processes: int = os.cpu_count() - 1):
    h5_file_path = os.path.join(output_path, "tokenized_data.h5")
    
    manager_h5 = Manager()
    lock_h5 = manager_h5.Lock()
    manager_cache = Manager()
    lock_cache = manager_cache.Lock()

    args = [(idx, dataset_class, tokenizer_classes, h5_file_path, lock_h5, lock_cache) for idx in range(len(dataset_class))]

    with Pool(processes=num_processes) as pool:
        with CodeTimer():
            results = pool.starmap(_process_trajectory, args)

    return [r for r in results if r is not None]


def main():
    setup_logging(file_path="tmp/logs/", console=False)

    parser = argparse.ArgumentParser(description="Tokenize input trajectory files")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input file/directory. If directory, all files in\
            the directory will be processed. If file, only the file will be processed.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        choices=["bio2token", "foldseek", "foldtoken4"],
        default=["bio2token", "foldseek", "foldtoken4"],
        help="List of tokenizers to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mdcath", "misato", "atlas"],
        help="Type of dataset structure to process.",
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    tokenizers = args.tokenizers
    dataset = args.dataset

    logging.info("Input Path: %s", input_path)
    logging.info("Output Path: %s", output_path)
    logging.info("Tokenizers: %s", tokenizers)

    tokenizer_classes = {t: {"foldseek": FoldSeekTokenizer, "bio2token": Bio2TokenTokenizer, "foldtoken4": FoldToken4Tokenizer}[t]() for t in tokenizers}

    dataset_class = {"mdcath": MDCATHDataset, "misato": MisatoDataset, "atlas": AtlasDataset}[dataset](data_dir=input_path, save_path=output_path)


    results = process_dataset_parallel(dataset_class=dataset_class, tokenizer_classes=tokenizer_classes, output_path=output_path, num_processes=1)
    logging.info("Processed items: %s", results)


if __name__ == "__main__":
    main()
