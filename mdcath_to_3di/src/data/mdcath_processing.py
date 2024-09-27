import io
import typing as T
from io import BytesIO

import h5py
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import IUPACData
from datasets.utils.file_utils import xopen

from src.data.foldseek import get_3di_sequences_from_memory
from src.data.utils import CodeTimer


def extract_mdcath_information(
    file_path: T.Union[str], config: T.Dict[str, T.List[str]] = None
) -> T.List[dict]:
    """
    Extracts information from an MDCATH HDF5 file.
    Args:
        file_path (str): Path to the HDF5 file containing MDCATH data.
        traj_temp (str): Temperature trajectory identifier within the HDF5 file.
        traj_sim (str): Simulation trajectory identifier within the HDF5 file.
    Returns:
        dict: A dictionary containing the following keys:
            - 'name' (str): The name of the trajectory.
            - 'pdb' (str): The PDB protein atoms in string format.
            - 'coords' (numpy.ndarray): The coordinates of the trajectory.
            - 'seq' (str): The amino acid sequence derived from the PDB structure.
    """

    if config:
        trajectory_temperatures = config["temperatures"]
        trajectory_replicas = config["replicas"]
    else:
        trajectory_temperatures = ["320", "348", "379", "413", "450"]
        trajectory_replicas = ["0", "1", "2", "3", "4"]

    with h5py.File(file_path, "r") as file:
        trajectory_name = list(file.keys())[0]
        traj_pdb_atoms = file[trajectory_name]["pdbProteinAtoms"][()].decode("utf-8")
        # trajectory_cords = {}
        
        for traj_temp in trajectory_temperatures:
            for traj_repl in trajectory_replicas:
                # trajectory_cords[traj_temp][traj_repl] =
                a = file[trajectory_name][traj_temp][traj_repl]["coords"][()]
                print(a)
                # traj_coords = 
                # print(trajectory_name, trajectory_temperature, trajectory_replica)


    # print(traj_temp, traj_sim)
    # print(traj_pdb_atoms)

    # parser = PDBParser()
    # structure = parser.get_structure("pdb_structure", io.StringIO(traj_pdb_atoms))
    # aa_seq = "".join(
    #     [
    #         IUPACData.protein_letters_3to1.get(x.get_resname().capitalize(), "X")
    #         for x in structure[0].get_residues()
    #     ]
    # )

    # traj_coords = file[traj_name][traj_temp][traj_sim]["coords"][()]
    # return {
    #     "name": traj_name,
    #     "pdb": traj_pdb_atoms,
    #     "coords": traj_coords,
    #     "seq": aa_seq,
    # }


def replace_coordinates_in_pdb(original_pdb: str, new_coordinates: np.array):
    """
    Replace coordinates in PDB content with new ones provided as a numpy array.

    Args:
    original_pdb (str): String containing the original PDB content.
    new_coordinates (np.array): New coordinates as a numpy array with shape (n_atoms, 3).

    Returns:
    str: A string of the modified PDB content.
    """
    lines = original_pdb.split("\n")
    atom_lines = [line for line in lines if line.startswith("ATOM")]

    if len(atom_lines) != len(new_coordinates):
        raise ValueError(
            "The number of new coordinates does not match the number of atoms in the PDB content."
        )

    modified_lines = []
    for line, (x, y, z) in zip(atom_lines, new_coordinates):
        new_line = line[:30] + "{:>8.3f}{:>8.3f}{:>8.3f}".format(x, y, z) + line[54:]
        modified_lines.append(new_line)

    atom_line_index = 0
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            lines[i] = modified_lines[atom_line_index]
            atom_line_index += 1

    return "\n".join(lines)


def mdcath_process(data):
    with CodeTimer():
        items = []
        for x in range(0, len(data["coords"])):
            new = replace_coordinates_in_pdb(
                original_pdb=data["pdb"], new_coordinates=data["coords"][x]
            )
            items.append(new)

    with CodeTimer():
        file_path = f"../tmp/data/3Di/{data['name']}.fasta"
        with open(file_path, "w", encoding="UTF-8") as file:
            file.write(
                f">{data['name']}|{data['seq']}\n"
                + "\n".join(get_3di_sequences_from_memory(pdb_files=items))
            )
        # print(data["name"], file_path)


def download_process(values: dict, config: dict):
    trajectory_url = values["image"]["path"]
    # if trajectory_url.split("_")[-1] not in ["1avyB00.h5"]:
    #     return
    # print(trajectory_url)

    with xopen(trajectory_url, "rb") as file:
        bytes_ = BytesIO(file.read())

    data = extract_mdcath_information(
        bytes_, traj_temp=config["traj_temp"], traj_sim=config["traj_sim"]
    )

    mdcath_process(data)
