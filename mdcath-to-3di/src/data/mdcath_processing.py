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

from src.data.foldseek import get_3di_sequences_from_memory

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
    for temp in extracted_traj['coords']:
        for replica in extracted_traj['coords'][temp]:
            extracted_traj['coords'][temp][replica] = rmsd_align(
                coordiantes=extracted_traj['coords'][temp][replica],
                reference_pdb=extracted_traj['pdb']
            )
    return extracted_traj


def save_PDBs(output_dir: str, PDBs: T.List[str]= None, template_PDB: str = None, name: str = None) -> str:

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
            items[f"{extraced_trajectroy["name"]}|{temp}|{replica}|{extraced_trajectroy["seq"]}"] = updated_pdbs
    return items


def translate_pdb_to_3di(pbds: dict) -> dict:
    """
    Translates a dictionary of PDB files into 3Di sequences.

    Args:
        pbds (dict): A dictionary where keys are identifiers and values are lists of PDB file paths.

    Returns:
        dict: A dictionary where keys are the same identifiers and values are the corresponding 3DI sequences.
    """
    items = {}
    for key, values in pbds.items():
        items[key] = get_3di_sequences_from_memory(pdb_files=values)
    return items


def generate_fasta(extraced_traj: dict, processed_3Di: dict) -> str:
    """
    Generates a FASTA file from the extracted trajectory and the processed 3Di sequences.

    Args:
        extraced_traj (dict): A dictionary containing the extracted trajectory data.
                              It should have the following structure:
                              {
                                  "name": trajectory_name,
                                  "seq": amino_acid_sequence,
                                }
        processed_3Di (dict): A dictionary containing the processed 3Di sequences.
                                It should have the following structure:
                                {
                                    "temp|replica": [sequence1, sequence2, ...],
                                }
    Returns:
        str: A string containing the FASTA formatted data.
    """
    items = []
    for name, sequences in processed_3Di.items():
        items.append(f">{name}|{extraced_traj['seq']}")
        items.extend(sequences)
    return "\n".join(items)


def save_fasta(path: str, fasta: str) -> str:
    """
    Save a FASTA formatted string to a file.
    Args:
        path (str): The directory path where the FASTA file will be saved.
        extraced_traj (dict): A dictionary containing trajectory information,
                              must include a 'name' key to name the file.
        fasta (str): The FASTA formatted string to be written to the file.
    Returns:
        str: The full path to the saved FASTA file.
    """
    with open(path, "w", encoding="UTF-8") as file:
        file.write(fasta)
    return path


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
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
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


def process_pipeline(md_cath_hdf5: bytes, config: dict = None, save_path_pdb: str = None, save_path_fasta: str = None) -> str:
    extraced_traj = extract_mdcath_information(
        file_path=md_cath_hdf5,
        config=config
    )
    processed_traj = rmsd_align_all(extraced_traj)
    processed_traj = generate_mdcath_coordiante_pdbs(extraced_trajectroy=processed_traj)

    if save_path_pdb:
        save_PDBs(
            template_PDB=extraced_traj['pdb'],
            output_dir=f"{save_path_pdb}/{extraced_traj['name']}",
            name=f"{extraced_traj['name']}"
        )

        for cathid_temp_repl, pdb_frames in processed_traj.items():
            save_PDBs(
                PDBs=pdb_frames,
                output_dir=f"{save_path_pdb}/{extraced_traj['name']}",
                name=cathid_temp_repl
            )

    processed_traj = translate_pdb_to_3di(processed_traj)

    if save_path_fasta:
        processed_fasta = generate_fasta(
            extraced_traj=extraced_traj,
            processed_3Di=processed_traj
        )
        save_fasta(
            fasta=processed_fasta,
            path=f"{save_path_fasta}/{extraced_traj['name']}.fasta",
        )
        del processed_fasta
        
    processed_traj = generate_pssms(processed_traj)

    return processed_traj


def download_process_pipeline(url: str = None, path: str = None, config: dict = None, save_path_pdb: str = None, save_path_fasta: str = None):
    bytes_ = download_open(
        url=url,
        path=path,
        config=config)
    return process_pipeline(
        md_cath_hdf5=bytes_,
        config=config,
        save_path_pdb=save_path_pdb,
        save_path_fasta=save_path_fasta
    )


def read_3Di_fasta(fasta: str) -> dict:
    """
    Read a FASTA file containing 3Di sequences.
    Args:
        fasta (str): The FASTA formatted string.
    Returns:
        dict: A dictionary containing the 3Di sequences.
    """
    items = {}
    for line in fasta.split("\n"):
        if line.startswith(">"):
            name = line[1:]
            items[name] = []
        else:
            items[name].append(line)
    return items


def fastas_to_hf_dataset(fasta: str, output_path: str) -> None:
    """
    Convert a FASTA file to an HDF5 dataset.
    Args:
        fasta (str): The FASTA formatted string.
        output_path (str): The path to the output HDF5 file.
    """
    pass