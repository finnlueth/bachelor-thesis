import numpy as np
import numpy.typing as npt
import typing as T

import MDAnalysis as mda
from MDAnalysis.analysis.align import AlignTraj
import io


def rmsd_align(
    coordinates: npt.NDArray, reference_pdb: str, frame: int = None
) -> npt.NDArray:
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

    if not frame:
        ref = mda.Universe(io.StringIO(reference_pdb), format="PDB")
        trj = mda.Universe(io.StringIO(reference_pdb), format="PDB")
        trj.load_new(coordinates)
        AlignTraj(trj, ref, select="protein", in_memory=True).run()
    else:
        trj = mda.Universe(io.StringIO(reference_pdb), format="PDB")
        trj.load_new(coordinates)
        ref = trj.select_atoms("protein")
        trj.trajectory[frame]
        AlignTraj(trj, ref, select="protein", in_memory=True).run()

    return trj.trajectory.timeseries(asel=trj.atoms, order="fac")


def replace_pdb_coordinates(pdb: str, coordinates: npt.NDArray) -> T.List[str]:
    """
    Replace coordinates in PDB content with new ones provided as a NumPy array.
    Args:
        pdb (str): String containing the original PDB content.
        coordinates (np.array): New coordinates as a NumPy array.
            - Shape (n_atoms, 3) for a single set.
            - Shape (n_frames, n_atoms, 3) for multiple sets.
    Returns:
        Union[str, List[str]]: Modified PDB content as a string if a single set of coordinates is provided,
                               or a list of strings if multiple sets are provided.
    """
    lines = pdb.split("\n")
    atom_lines = [(i, line) for i, line in enumerate(lines) if line.startswith("ATOM")]
    total_atoms = len(atom_lines)
    if coordinates.ndim == 3:
        if total_atoms != coordinates.shape[1]:
            raise ValueError(
                f"The number of new coordinates ({coordinates.shape[1]}) does not match the number of atoms in the PDB content ({total_atoms})."
            )

        pdbs_with_coords = []
        for frame_coords in coordinates:
            frame_lines = lines.copy()
            for (i, line), coord in zip(atom_lines, frame_coords):
                x, y, z = coord
                frame_lines[i] = f"{line[:30]}{x:>8.3f}{y:>8.3f}{z:>8.3f}{line[54:]}"
            pdbs_with_coords.append("\n".join(frame_lines))

        return pdbs_with_coords
    else:
        raise ValueError(
            "Coordinates array must have shape (n_atoms, 3) or (n_frames, n_atoms, 3)."
        )
