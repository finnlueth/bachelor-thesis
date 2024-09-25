import h5py
from Bio.SeqUtils import IUPACData
from Bio.PDB import PDBParser
import io
import numpy as np


def h5_tree(val, pre=""):
    output = ""
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            if type(val) == h5py._hl.group.Group:
                output += pre + "└── " + key + "\n"
                output += h5_tree(val, pre + "    ")
            else:
                try:
                    output += pre + "└── " + key + " (%d)\n" % len(val)
                except TypeError:
                    output += pre + "└── " + key + " (scalar)\n"
        else:
            if type(val) == h5py._hl.group.Group:
                output += pre + "├── " + key + "\n"
                output += h5_tree(val, pre + "│   ")
            else:
                try:
                    output += pre + "├── " + key + " (%d)\n" % len(val)
                except TypeError:
                    output += pre + "├── " + key + " (scalar)\n"
    return output


def extract_dataset_information(file_path):
    with h5py.File(file_path, "r") as file:
        traj_name = list(file.keys())[0]
        traj_temp = "320"
        traj_sim = "0"

        traj_pdb_atoms = file[traj_name]["pdbProteinAtoms"][()].decode("utf-8")
        # traj_chain = file[traj_name]["resname"][()].type(str)

        parser = PDBParser()
        structure = parser.get_structure("pdb_structure", io.StringIO(traj_pdb_atoms))
        aa_seq = "".join(
            [
                IUPACData.protein_letters_3to1.get(x.get_resname().capitalize(), "X")
                for x in structure[0].get_residues()
            ]
        )

        traj_coords = file[traj_name][traj_temp][traj_sim]["coords"][()]
        return {
            "name": traj_name,
            "pdb": traj_pdb_atoms,
            "coords": traj_coords,
            "seq": aa_seq,
            # "seq": list(structure[0].get_residues()),
        }


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
