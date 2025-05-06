"""
python3 scripts/pssm_script.py --input_path /home/finnlueth/mnt/smb/data/datasets/mdCATH/data --output_path ./tmp/output/pssm/mdcath_nopc --dataset mdcath

python3 scripts/pssm_script.py --input_path /mnt/gondolin/data/datasets/mdCATH/data --output_path ./tmp/output/pssm/mdcath_nopc --dataset mdcath

'A C D E F G H I K L M N P Q R S T V W Y'
"""

import argparse
import json
import logging
import os
import random
import subprocess
import tempfile
import time
import typing as T
import warnings
from multiprocessing import Lock, Manager, Pool
from pathlib import Path

import h5py
import MDAnalysis as mda
import numpy as np
import pandas as pd
import psutil
from Bio import BiopythonDeprecationWarning

from src.data.load import (
    AtlasDataset,
    MDCATHDataset,
    MisatoDataset,
    TrajectoryDataset,
    TrajectoryWrapper,
)
from src.utils.errors import TrajectoryAlreadyProcessedError
from src.utils.logging import setup_logging
from src.utils.utils import CodeTimer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", BiopythonDeprecationWarning)

null_handler = logging.NullHandler()
mda_logger = logging.getLogger("MDAnalysis")
for handler in mda_logger.handlers[:]:
    mda_logger.removeHandler(handler)
mda_logger.addHandler(null_handler)
mda_logger.setLevel(logging.CRITICAL)
mda_logger.propagate = False


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


def write_to_h5_file(
    save_path: str,
    trajectory: TrajectoryWrapper,
    pssms: T.Dict[str, np.ndarray],
):
    """Write data to H5 file with proper locking"""
    while True:
        try:
            with lock_h5:
                with h5py.File(save_path + "/pssm_data.h5", "a") as h5_file:
                    if trajectory["name"] not in h5_file:
                        trajectory_group = h5_file.create_group(trajectory["name"])
                    else:
                        trajectory_group = h5_file[trajectory["name"]]

                    trajectory_group.attrs["sequence"] = trajectory["sequence"]
                    trajectory_group.attrs["name"] = trajectory["name"]
                    trajectory_group.attrs["structure"] = trajectory["structure"]

                    for temp_replica, pssm in pssms.items():
                        if temp_replica in trajectory_group:
                            del trajectory_group[temp_replica]
                        trajectory_group.create_dataset(temp_replica, data=pssm)
                break
        except Exception as e:
            logging.error("Waiting for h5 file to become available...", e)
            time.sleep(random.uniform(0, 0.2))


def get_pssm_from_trajectory(trajectory_pdbs: T.List[str]) -> np.ndarray:
    # print(len(trajectory_pdbs))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        os.makedirs(input_dir)

        for i, pdb in enumerate(trajectory_pdbs):
            pdb_path = os.path.join(input_dir, f"structure_{i+1:05d}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb)

        # input_files = sorted(os.listdir(input_dir))
        # print("Files in input directory:")
        # for filename in input_files:
        #     print(filename)

        foldseek = "foldseek"
        mmseqs = "mmseqs"

        subprocess.run(
            [foldseek, "createdb", input_dir, os.path.join(tmpdir, "inputdb")],
            check=True,
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
        )

        with open(os.path.join(tmpdir, "inputdb.index")) as f:
            index_data = f.readlines()

        with open(os.path.join(tmpdir, "fake_aln.tsv"), "w") as f:
            for line in index_data:
                parts = line.split()
                length = int(parts[2]) - 2
                f.write(f"0\t{parts[0]}\t0\t1.00\t0\t0\t{length-1}\t{length}\t0\t{length-1}\t{length}\t{length}M\n")

        # print("Contents of fake_aln.tsv:")
        # with open(os.path.join(tmpdir, "fake_aln.tsv")) as f:
        #     print(f.read())

        subprocess.run(
            [
                mmseqs,
                "tsv2db",
                os.path.join(tmpdir, "fake_aln.tsv"),
                os.path.join(tmpdir, "fake_aln_db"),
                "--output-dbtype",
                "5",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
        )

        victors_params = [
            "--pca",
            "0",
            "--pcb",
            "1.5",
            "--profile-output-mode",
            "1",
            "--sub-mat",
            "/home/finnlueth/repos/bachelor-thesis/prot-md-pssm-legacy/submodules/foldseek/data/mat3di.out",
            "--mask-profile",
            "0",
            "--comp-bias-corr",
            "0",
            "--e-profile",
            "0.1",
            "-e",
            "0.1",
            "--gap-open",
            "11",
            "--gap-extend",
            "1",
        ]

        subprocess.run(
            [
                mmseqs,
                "result2profile",
                os.path.join(tmpdir, "inputdb_ss"),
                os.path.join(tmpdir, "inputdb_ss"),
                os.path.join(tmpdir, "fake_aln_db"),
                os.path.join(tmpdir, "profile.tsv"),
            ]
            + victors_params,
            check=True,
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
        )

        profile_data = pd.read_csv(os.path.join(tmpdir, "profile.tsv"), sep=" ", header=None, skiprows=2)
        profile_data = profile_data.iloc[:, :-1]
        # print(profile_data)
        # print(profile_data.to_numpy())
        # print(profile_data.to_numpy().shape)
        # print(profile_data.to_numpy())
        return profile_data.to_numpy()


def _process_trajectory(
    idx: int,
    dataset_class: TrajectoryDataset,
):
    trajectory = None

    try:
        logging.info(f"Processing trajectory {idx} of {len(dataset_class)}")
        trajectory = dataset_class[idx]

        total_bytes = sum(sum(len(pdb.encode()) for pdb in pdbs) for pdbs in trajectory["trajectory_pdbs"].values())
        size_gb = total_bytes / (1024**3)
        logging.error(f"Size of trajectory_pdbs for {trajectory['name']}: {size_gb:.2f} GB")

        name = trajectory["name"]
        trajectory["trajectory_pdbs"]["base"] = [trajectory["structure"]]
        logging.info(
            "Starting converting trajectory %s to PSSM",
            trajectory["name"],
        )

        trajectory_pssm = {}
        for temp_replica, trajectory_pdbs in trajectory["trajectory_pdbs"].items():
            if "base" not in temp_replica:
                trajectory_pssm[temp_replica] = get_pssm_from_trajectory(trajectory_pdbs)
                logging.error("Creating PSSM for trajectory %s for %s", trajectory["name"], temp_replica)

                write_to_h5_file(
                    save_path=dataset_class.save_path,
                    trajectory=trajectory,
                    pssms=trajectory_pssm,
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


def init_parallel(l_cache, l_h5):
    global lock_cache
    global lock_h5
    lock_cache = l_cache
    lock_h5 = l_h5


def process_dataset_parallel(
    dataset_class: TrajectoryDataset,
    num_processes: int = 1,
):
    args = [(idx, dataset_class) for idx in range(len(dataset_class))]#[:2]

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

    parser = argparse.ArgumentParser(description="Generate PSSM profiles from input trajectory files")
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
        "--dataset",
        type=str,
        required=True,
        choices=["mdcath", "misato", "atlas"],
        help="Type of dataset structure to process.",
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    dataset = args.dataset

    logging.info("Input Path: %s", input_path)
    logging.info("Output Path: %s", output_path)
    logging.info("Dataset: %s", dataset)

    dataset = {"mdcath": MDCATHDataset, "misato": MisatoDataset, "atlas": AtlasDataset}[dataset](
        dataset_name=dataset, data_dir=input_path, save_path=output_path
    )

    results = process_dataset_parallel(
        dataset_class=dataset,
        # num_processes=2,
        num_processes=psutil.cpu_count(logical=False) - 1,
        # num_processes=6,
    )

    logging.info("Processed items: %s", results)


if __name__ == "__main__":
    main()
