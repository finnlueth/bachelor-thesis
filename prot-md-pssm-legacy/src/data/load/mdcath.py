import os
import h5py
from Bio.SeqUtils import IUPACData
from Bio.PDB import PDBParser
import io
from .trajectory_dataset import TrajectoryDataset, TrajectoryWrapper
import numpy as np
import typing as T
import numpy.typing as npt
import logging
from src.utils.utils import CodeTimer
from src.rust_modules import replace_pdb_coordinates
from src.utils.bio import rmsd_align


class MDCATHDataset(TrajectoryDataset):
    def _load_trajectory_locations(self):
        if os.path.isdir(self.data_dir):
            return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".h5")]
        else:
            return [self.data_dir]

    def _get_item(self, idx):
        temperatures = ["320", "348", "379", "413", "450"]
        replicas = ["0", "1", "2", "3", "4"]

        with h5py.File(self.trajectory_locations[idx], "r") as file:
            name = list(file.keys())[0]
            structure = file[name]["pdbProteinAtoms"][()].decode("utf-8")
            sequence = "".join(
                [
                    IUPACData.protein_letters_3to1.get(x.get_resname().capitalize(), "X")
                    for x in PDBParser(QUIET=True).get_structure("pdb_structure", io.StringIO(structure))[0].get_residues()
                ]
            )

            aligned_trajectories, aligned_trajectory_pdbs = {}, {}
            for temp in temperatures:
                for replica in replicas:
                    logging.error(f"RMSD aligning and replacing coordinates for temperature {temp} and replica {replica} for trajectory {name}.")
                    aligned_trajectories[f"{temp}_{replica}"] = rmsd_align(file[name][temp][replica]["coords"][()], structure)
                    aligned_trajectory_pdbs[f"{temp}_{replica}"] = replace_pdb_coordinates(structure, aligned_trajectories[f"{temp}_{replica}"])

            trajectory = TrajectoryWrapper(
                name=name,
                sequence=sequence,
                structure=structure,
                # trajectories=trajectories,
                trajectory_pdbs=aligned_trajectory_pdbs,
            )
        return trajectory
