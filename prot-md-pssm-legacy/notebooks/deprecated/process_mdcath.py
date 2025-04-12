"""
Code for processing MDCATH dataset.
"""

import gc
import io
import os
import typing as T
import warnings
from io import BytesIO

import h5py
import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser
from Bio.SeqUtils import IUPACData
from datasets.utils.file_utils import xopen
from MDAnalysis.analysis.align import AlignTraj

from src.data.tokenize.foldseek import get_3di_sequences_from_memory

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter("ignore", BiopythonDeprecationWarning)


def config_check(config):
    """
    Checks and retrieves trajectory temperatures and replicas from the given configuration.
    Args:
        config (dict): A dictionary containing configuration settings. Expected keys are:
            - "temperatures" (list of str): List of temperature values.
            - "replicas" (list of str): List of replica values.
    Returns:
        tuple: A tuple containing two lists:
            - trajectory_temperatures (list of str): List of temperature values.
            - trajectory_replicas (list of str): List of replica values.
    If the config is None or empty, default values are returned:
        - trajectory_temperatures: ["320", "348", "379", "413", "450"]
        - trajectory_replicas: ["0", "1", "2", "3", "4"]
    """

    if config:
        trajectory_temperatures = config["temperatures"]
        trajectory_replicas = config["replicas"]
    else:
        trajectory_temperatures = ["320", "348", "379", "413", "450"]
        trajectory_replicas = ["0", "1", "2", "3", "4"]
    return trajectory_temperatures, trajectory_replicas


def extract_mdcath_information(file_path: T.Union[str, BytesIO], config: T.Dict[str, T.List[str]] = None) -> T.List[dict]:
    """
    Extracts information from an MDCATH HDF5 file.
    Args:
        file_path (Union[str, BytesIO]): Path to the HDF5 file or BytesIO object containing HDF5 MDCATH data.
        config (dict): Configuration dictionary containing temperature and replica information.
    Returns:
        dict: A dictionary containing the following keys:
            - 'name' (str): The name of the trajectory.
            - 'pdb' (str): The PDB protein atoms in string format.
            - 'coords' (numpy.ndarray): The coordinates of the trajectory.
            - 'seq' (str): The amino acid sequence derived from the PDB structure.
    """
    trajectory_temperatures, trajectory_replicas = config_check(config)

    with h5py.File(file_path, "r") as file:
        trajectory_name = list(file.keys())[0]
        trajectory_pdb_atoms = file[trajectory_name]["pdbProteinAtoms"][()].decode("utf-8")
        trajectory_cords = {temp: {repl: None for repl in trajectory_replicas} for temp in trajectory_temperatures}

        for traj_temp in trajectory_temperatures:
            for traj_repl in trajectory_replicas:
                trajectory_cords[traj_temp][traj_repl] = file[trajectory_name][traj_temp][traj_repl]["coords"][()]

        structure = PDBParser().get_structure("pdb_structure", io.StringIO(trajectory_pdb_atoms))
        aa_seq = "".join(
            [IUPACData.protein_letters_3to1.get(x.get_resname().capitalize(), "X") for x in structure[0].get_residues()]
        )

    return {
        "name": trajectory_name,
        "pdb": trajectory_pdb_atoms,
        "coords": trajectory_cords,
        "seq": aa_seq,
    }


def replace_coordinates_in_pdb(pdb: str, new_coordinates: npt.NDArray) -> str:
    """
    Replace coordinates in PDB content with new ones provided as a numpy array.
    Args:
        original_pdb (str): String containing the original PDB content.
        new_coordinates (np.array): New coordinates as a numpy array with shape (n_atoms, 3).
    Returns:
        tr: A string of the modified PDB content.
    """
    lines = pdb.split("\n")
    total_atoms = sum(1 for line in lines if line.startswith("ATOM"))

    if total_atoms != new_coordinates.shape[0]:
        raise ValueError(f"The number of new coordinates does not match the number of atoms\
                         in the PDB content {total_atoms}, {new_coordinates.shape[0]}.")

    coordinate_index = 0
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            x, y, z = new_coordinates[coordinate_index]
            new_line = f"{line[:30]}{x:>8.3f}{y:>8.3f}{z:>8.3f}{line[54:]}"
            lines[i] = new_line
            coordinate_index += 1

    return "\n".join(lines)


def replace_coordinates_in_pdbs(pdb: str, coordinates: npt.NDArray) -> T.List[str]:
    """
    Replace coordinates in a PDB file with multiple sets of coordinates.
    Args:
        pdb (str): The PDB file content as a string.
        coordinates (np.array): The new coordinates as a numpy array of shape (n_frames, n_atoms, 3).
    Returns:
        List[str]: A list of PDB strings with the new coordinates.
    """
    pdbs_with_coords = []
    for coords in coordinates:
        pdbs_with_coords.append(
            replace_coordinates_in_pdb(
                pdb=pdb,
                new_coordinates=coords,
            )
        )
    return pdbs_with_coords


def extract_coordinates_from_pdb(pdb_str) -> npt.NDArray:
    """
    Extracts the xyz atom coordinates from a PDB file string.
    Args:
        pdb_str (str): The PDB file content as a string.
    Returns:
        np.ndarray: A numpy array of shape (n, 3) where n is the number of atoms.
    """
    coordinates = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM"):
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coordinates.append([x, y, z])

    return np.array(coordinates)


def rmsd_align(coordiantes: npt.NDArray, reference_pdb: str, frame: int = None) -> npt.NDArray:
    """
    Aligns a set of coordinates to a reference PDB structure or a reference frame using RMSD alignment.
    Parameters:
        coordinates : npt.NDArray
            The coordinates to be aligned. This should be a NumPy array of shape (n_frames, n_atoms, 3).
        reference_pdb : str
            The PDB formatted string of the reference structure.
        frame : int, optional
            The frame number to which the trajectory should be aligned (default is 0). If left blank, the trjectory will be aligned to the reference PDB.
    Returns:
        npt.NDArray
            The aligned coordinates as a NumPy array of shape (n_frames, n_atoms, 3).
    """

    if reference_pdb:
        ref = mda.Universe(io.StringIO(reference_pdb), format="PDB")
        trj = mda.Universe(io.StringIO(reference_pdb), format="PDB")
        trj.load_new(coordiantes)
        AlignTraj(trj, ref, select="protein", in_memory=True).run()
    else:
        trj = mda.Universe(io.StringIO(reference_pdb), format="PDB")
        trj.load_new(coordiantes)
        ref = trj.select_atoms("protein")
        trj.trajectory[frame]
        AlignTraj(trj, ref, select="protein", in_memory=True).run()

    return trj.trajectory.timeseries(asel=trj.atoms, order="fac")


def rmsd_align_all(extracted_traj: dict) -> dict:
    """
    Aligns all coordinates in the extracted trajectory to a reference PDB structure using RMSD alignment.
    Args:
        extracted_traj (dict): A dictionary containing trajectory data with the following structure:
            {
                'coords': {
                    temperature1: {
                        replica1: coordinates_array,
                        replica2: coordinates_array,
                        ...
                    },
                    temperature2: {
                        replica1: coordinates_array,
                        replica2: coordinates_array,
                        ...
                    },
                    ...
                },
                'pdb': reference_pdb_structure
            }
    Returns:
        dict: The input dictionary with all coordinates aligned to the reference PDB structure.
    """
    for temp in extracted_traj["coords"]:
        for replica in extracted_traj["coords"][temp]:
            extracted_traj["coords"][temp][replica] = rmsd_align(
                coordiantes=extracted_traj["coords"][temp][replica], reference_pdb=extracted_traj["pdb"]
            )
    return extracted_traj


def save_PDBs(output_dir: str, PDBs: T.List[str] = None, template_PDB: str = None, name: str = None) -> str:
    """
    Save a list of PDB strings to a directory.
    Args:
        PDBs (List[str]): List of PDB strings.
        template_PDB (str): Template PDB string to be saved as the first file.
        output_dir (str): Directory where the PDB files will be saved.
    Returns:
        str: The path to the output directory.
    """
    if not name:
        name = "PDB"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = f"{output_dir}/{name}"

    if template_PDB:
        tmp_path = f"{output_dir}_template.pdb"
        # print(tmp_path)
        with open(tmp_path, "w", encoding="UTF-8") as f:
            f.write(template_PDB)

    if PDBs:
        for i, pdb in enumerate(PDBs):
            tmp_path = f"{output_dir}_{i}.pdb"
            # print(tmp_path)
            with open(tmp_path, "w", encoding="UTF-8") as f:
                f.write(pdb)
    return output_dir


def save_pdbs_from_traj_dict(extraced_traj: dict, output_dir: str) -> str:
    """
    Save PDB files from the extracted trajectory dictionary.
    Args:
        extraced_traj (dict): A dictionary containing trajectory information.
        output_dir (str): Directory where the PDB files will be saved.
    Returns:
        str: The path to the output directory.
    """
    pass
    # pdbs = []
    # for temp in extraced_traj["coords"]:
    #     for replica in extraced_traj["coords"][temp]:
    #         print(temp, replica, output_dir)
    # return save_PDBs(pdbs, output_dir, extraced_traj["pdb"], extraced_traj["name"])
    #         pdbs.append(extraced_traj["coords"][temp][replica])


def generate_mdcath_coordiante_pdbs(extraced_trajectroy: dict) -> dict:
    """
    Generates updated PDB files with new coordinates from the extracted trajectory for all replaicas.

    Args:
        extraced_traj (dict): A dictionary containing the extracted trajectory data.
                              It should have the following structure:
                              {
                                  "coords": {
                                      temp1: {
                                          replica1: coordinates1,
                                          replica2: coordinates2,
                                          ...
                                      },
                                      temp2: {
                                          replica1: coordinates3,
                                          replica2: coordinates4,
                                          ...
                                      },
                                      ...
                                  },
                                  "pdb": original_pdb_data
                              }

    Returns:
        dict: A dictionary where keys are strings in the format "temp|replica" and values are the updated PDB files.
    """
    items = {}
    for temp in extraced_trajectroy["coords"]:
        for replica in extraced_trajectroy["coords"][temp]:
            # print(temp, replica)
            updated_pdbs = replace_coordinates_in_pdbs(
                pdb=extraced_trajectroy["pdb"],
                coordinates=extraced_trajectroy["coords"][temp][replica],
            )
            items[f"{extraced_trajectroy['name']}|{temp}|{replica}|{extraced_trajectroy['seq']}"] = updated_pdbs
    return items


def download_open(url: str = None, path: str = None, config: dict = None) -> dict:
    """
    Download and process MDCATH data from a given URL.
    Args:
        url (str): The URL of the MDCATH data.
        config (dict): Configuration dictionary containing temperature and replica information.
    Returns:
        dict: A dictionary containing the processed data.
    """

    if url:
        with xopen(url, "rb") as file:
            bytes_ = BytesIO(file.read())
            # ["image"]["path"]
    elif path:
        with open(path, "rb") as file:
            bytes_ = BytesIO(file.read())
        return bytes_
    else:
        raise ValueError("Please provide either a URL or a path to the HDF5 file.")


def generate_pssms(processed_traj: dict) -> dict:
    """
    Generate PSSMs from a dictionary of 3Di sequences.
    Args:
        processed_traj (dict): A dictionary containing the processed 3Di sequences.
    Returns:
        dict: A dictionary containing the PSSMs for each 3Di sequence.
    """
    for key, sequences in processed_traj.items():
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        pfm = np.zeros((len(alphabet), len(sequences[0])))

        aa_to_index = {aa: idx for idx, aa in enumerate(alphabet)}

        for seq in sequences:
            for pos, aa in enumerate(seq):
                if aa in aa_to_index:
                    pfm[aa_to_index[aa], pos] += 1

        pwm = pfm / len(alphabet)
        pssm = np.log2(pwm)
        processed_traj[key] = pssm
    return processed_traj