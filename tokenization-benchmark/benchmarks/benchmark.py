
def compute_backbone_rmsd(truth_structure, pred_structure):
    try:
        # Get backbone atoms
        truth_atoms = [
            atom
            for atom in truth_structure.get_atoms()
            if atom.get_name() in ["CA", "C", "N", "O"]
        ]
        pred_atoms = [
            atom
            for atom in pred_structure.get_atoms()
            if atom.get_name() in ["CA", "C", "N", "O"]
        ]

        if len(truth_atoms) != len(pred_atoms):
            print(f"Error: truth_atoms and pred_atoms have different lengths")
            return None

        # Get coordinates
        truth_coords = np.array([atom.get_coord() for atom in truth_atoms])
        pred_coords = np.array([atom.get_coord() for atom in pred_atoms])

        # Center the coordinates
        truth_center = np.mean(truth_coords, axis=0)
        pred_center = np.mean(pred_coords, axis=0)
        truth_coords = truth_coords - truth_center
        pred_coords = pred_coords - pred_center

        # Calculate the covariance matrix
        H = truth_coords.T @ pred_coords

        # Perform SVD
        U, S, Vt = np.linalg.svd(H)

        # Calculate rotation matrix
        R = Vt.T @ U.T

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation to prediction coordinates
        pred_coords_aligned = pred_coords @ R.T

        # Calculate RMSD
        diff = truth_coords - pred_coords_aligned
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        return rmsd

    except Exception as e:
        print(f"Error in RMSD calculation: {e}")
        return None


        # truth_structure = parser.get_structure(
        #     truth_file, os.path.join(truth_dir, truth_file + ".pdb")
        # )
        # pred_structure = parser.get_structure(
        #     pred_file, os.path.join(pred_dir, pred_file + ".pdb")
        # )

        # rmsd = compute_backbone_rmsd(truth_structure, pred_structure)
        # print(f"{model_name} - {pred_file}: {rmsd}")
