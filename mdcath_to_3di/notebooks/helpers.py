import mdtraj as md
import tempfile


def load_pdb_from_string(file_contents):
    """
    Load an MDTraj topology object from a string containing file contents.

    Parameters:
    - file_contents (str): The contents of a topology file as a string.

    Returns:
    - mdtraj.Topology: The loaded topology object.
    """
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.top', delete=False) as tmpfile:
        tmpfile.write(file_contents)
        tmpfile.flush()  # Ensure all data is written

        # Load the topology using the temporary file's path
        topology = md.load_pdb(tmpfile.name)

    return topology