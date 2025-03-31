from Bio import PDB
import py3Dmol
import os
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt


def visualize_structure_alignment(truth_structure, pred_structure, parser):
    """
    Aligns two protein structures, calculates RMSD, and visualizes the alignment.

    Args:
        truth_file (str): Filename of the ground truth structure
        pred_file (str): Filename of the predicted structure
        truth_dir (str): Directory containing truth files
        pred_dir (str): Directory containing prediction files
        parser (Bio.PDB.PDBParser): Parser instance for reading PDB files

    Returns:
        float: RMSD value between the structures
        py3Dmol.view: Visualization object showing the alignment
    """
    truth_model = truth_structure[0]
    pred_model = pred_structure[0]

    truth_ca_atoms = []
    pred_ca_atoms = []

    # Get CA atoms from both structures
    for chain in truth_model:
        for residue in chain:
            if "CA" in residue:
                truth_ca_atoms.append(residue["CA"])

    for chain in pred_model:
        for residue in chain:
            if "CA" in residue:
                pred_ca_atoms.append(residue["CA"])

    # Add debug information
    print(f"Number of CA atoms in truth structure: {len(truth_ca_atoms)}")
    print(f"Number of CA atoms in predicted structure: {len(pred_ca_atoms)}")

    if len(truth_ca_atoms) != len(pred_ca_atoms):
        raise ValueError(
            f"Structures have different numbers of CA atoms: truth={len(truth_ca_atoms)}, pred={len(pred_ca_atoms)}"
        )

    # Superimpose the structures
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(truth_ca_atoms, pred_ca_atoms)
    super_imposer.apply(pred_model.get_atoms())

    # Get RMSD
    rmsd = super_imposer.rms

    # Create visualization
    view = py3Dmol.view(width=800, height=600)

    # Convert structures to PDB strings
    truth_io = StringIO()
    pred_io = StringIO()

    # Create PDBIO instances
    truth_pdbio = PDB.PDBIO()
    pred_pdbio = PDB.PDBIO()

    # Set structures and save to StringIO
    truth_pdbio.set_structure(truth_structure)
    pred_pdbio.set_structure(pred_structure)

    if truth_pdbio.structure is None or pred_pdbio.structure is None:
        raise ValueError("Failed to set structures in PDBIO")

    truth_pdbio.save(truth_io)
    pred_pdbio.save(pred_io)

    # Add truth structure in blue
    view.addModel(truth_io.getvalue(), "pdb")
    view.setStyle({"model": -1}, {"cartoon": {"color": "blue"}})

    # Add predicted structure in red
    view.addModel(pred_io.getvalue(), "pdb")
    view.setStyle({"model": -1}, {"cartoon": {"color": "red"}})

    view.zoomTo()

    return rmsd, tm_score, view


def plot_with_error(values):
    # Calculate mean and standard error
    mean = np.mean(values)
    std_err = np.std(values) / np.sqrt(len(values))

    # Create bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(0, mean, yerr=std_err, capsize=5, color='lightblue', width=0.5)
    
    # Remove x ticks since we only have one bar
    plt.xticks([])
    
    # Add labels and title
    plt.ylabel('Value')
    plt.title('Mean with Standard Error')

    # Adjust layout
    plt.tight_layout()

    return plt