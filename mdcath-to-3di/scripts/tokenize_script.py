#! /usr/bin/env python3
"""
python3 scripts/tokenize_script.py --input_path ./tmp/data/mdCATH/data/ --output_path ./tmp/output/tokenized/mdcath --tokenizers bio2token foldseek --dataset mdcath

python scripts/tokenize_script.py --input_path ./tmp/data/mdCATH/data/ --output_path ./tmp/output/tokenized/mdcath --tokenizers foldseek --dataset mdcath

python scripts/tokenize_script.py --input_path /home/finnlueth/mnt/smb/data/datasets/mdCATH/data --output_path ./tmp/output/tokenized/mdcath --tokenizers foldseek --dataset mdcath
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
import psutil
from src.data.load import TrajectoryDataset, TrajectoryWrapper
from pathlib import Path
import typing as T
import json
import time
import random
import MDAnalysis as mda

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", BiopythonDeprecationWarning)


null_handler = logging.NullHandler()
mda_logger = logging.getLogger("MDAnalysis")
for handler in mda_logger.handlers[:]:
    mda_logger.removeHandler(handler)
mda_logger.addHandler(null_handler)
mda_logger.setLevel(logging.CRITICAL)
mda_logger.propagate = False



def write_to_h5_file(
    save_path: str,
    tokenizer_name: str,
    trajectory: TrajectoryWrapper,
    tokenizer_results: T.Dict[str, T.List[str]],
):
    """Write data to H5 file with proper locking"""
    while True:
        try:
            with lock_h5:
                with h5py.File(save_path + "/tokenized_data.h5", "a") as h5_file:
                    if tokenizer_name not in h5_file:
                        tokenizer_group = h5_file.create_group(tokenizer_name)
                    else:
                        tokenizer_group = h5_file[tokenizer_name]

                    if trajectory["name"] not in tokenizer_group:
                        trajectory_group = tokenizer_group.create_group(trajectory["name"])
                    else:
                        trajectory_group = tokenizer_group[trajectory["name"]]

                    trajectory_group.attrs["sequence"] = trajectory["sequence"]
                    trajectory_group.attrs["name"] = trajectory["name"]
                    trajectory_group.attrs["structure"] = trajectory["structure"]

                    for temp_replica, tokens in tokenizer_results.items():
                        tokens_array = np.array(tokens, dtype="S")
                        if temp_replica in trajectory_group:
                            del trajectory_group[temp_replica]
                        trajectory_group.create_dataset(temp_replica, data=tokens_array)
                break
        except Exception as e:
            logging.error("Waiting for h5 file to become available...", e)
            time.sleep(random.uniform(0, 0.2))


def use_trajectory_location_new(idx, dataset):
    """
    Mark a trajectory location as used.
    """
    while True:
        try:
            with lock_cache:
                with open(dataset.save_path + f"/{dataset.cache_name}.json", "r") as f:
                    used_trajectory_locations = json.load(f)
                if dataset.trajectory_locations[idx] not in used_trajectory_locations:
                    used_trajectory_locations.append(dataset.trajectory_locations[idx])
                    with open(dataset.save_path + f"/{dataset.cache_name}.json", "w") as f:
                        json.dump(used_trajectory_locations, f)
                break
        except Exception as e:
            logging.error(f"Waiting for trajectory /{dataset.cache_name}.json to become available...", e)
            time.sleep(random.uniform(0, 0.2))


def _process_trajectory(
    idx: int,
    dataset_class: TrajectoryDataset,
    tokenizers: T.List[TrajectoryTokenizer],
):
    trajectory = None
    tokenized_trajectories = None
    
    try:
        logging.info(f"Processing trajectory {idx} of {len(dataset_class)}")
        trajectory = dataset_class[idx]

        # Calculate size of trajectory_pdbs in GB
        total_bytes = sum(sum(len(pdb.encode()) for pdb in pdbs) for pdbs in trajectory["trajectory_pdbs"].values())
        size_gb = total_bytes / (1024 * 1024 * 1024)
        logging.error(f"Size of trajectory_pdbs for {trajectory['name']}: {size_gb:.2f} GB")

        name = trajectory["name"]
        trajectory["trajectory_pdbs"]["base"] = [trajectory["structure"]]
        logging.info(
            "Starting tokenizing trajectory %s with %s tokenizers",
            trajectory["name"],
            [tokenizer.tokenizer_name for tokenizer in tokenizers],
        )

        for tokenizer in tokenizers:
            tokenized_trajectories = {}
            for temp_replica, trajectory_pdb in trajectory["trajectory_pdbs"].items():
                tokenized_trajectories[temp_replica] = tokenizer.tokenize(trajectory_pdb)
                logging.error("Tokenized trajectory %s for %s", trajectory["name"], temp_replica)
            write_to_h5_file(
                save_path=dataset_class.save_path,
                tokenizer_name=tokenizer.tokenizer_name,
                trajectory=trajectory,
                tokenizer_results=tokenized_trajectories,
            )

        use_trajectory_location_new(idx=idx, dataset=dataset_class)

        logging.error("Finished tokenizing trajectory %s", trajectory["name"])
        return name

    except TrajectoryAlreadyProcessedError as e:
        logging.error(e)
        return None
    except Exception as e:
        logging.error(f"Error processing trajectory {idx}: {str(e)}")
        return None
    finally:
        # Explicit cleanup
        if trajectory is not None:
            del trajectory
        if tokenized_trajectories is not None:
            del tokenized_trajectories


def init_parallel(l_cache, l_h5):
    global lock_cache
    global lock_h5
    lock_cache = l_cache
    lock_h5 = l_h5


def process_dataset_parallel(
    dataset_class: TrajectoryDataset,
    tokenizer_classes: dict,
    num_processes: int = psutil.cpu_count(logical=False) - 1,
):
    args = [(idx, dataset_class, tokenizer_classes) for idx in range(len(dataset_class))]

    lock_cache = Lock()
    lock_h5 = Lock()

    with Pool(processes=num_processes, initializer=init_parallel, initargs=(lock_cache, lock_h5)) as pool:
        try:
            results = pool.starmap(_process_trajectory, args)
        finally:
            pool.close()
            pool.join()

    return [r for r in results if r is not None]


def main():
    setup_logging(file_path="tmp/logs/", console=True)

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

    tokenizers = {
        {"foldseek": FoldSeekTokenizer, "bio2token": Bio2TokenTokenizer, "foldtoken4": FoldToken4Tokenizer}[tokenizer](
            tokenizer_name=tokenizer
        )
        for tokenizer in tokenizers
    }

    dataset = {"mdcath": MDCATHDataset, "misato": MisatoDataset, "atlas": AtlasDataset}[dataset](
        dataset_name=dataset, data_dir=input_path, save_path=output_path
    )

    results = process_dataset_parallel(
        dataset_class=dataset,
        tokenizer_classes=tokenizers,
        num_processes=psutil.cpu_count(logical=False) - 1,
        # num_processes=6,
    )
    logging.info("Processed items: %s", results)


if __name__ == "__main__":
    main()
