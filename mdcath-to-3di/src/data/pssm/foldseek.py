import os
import shlex
import tempfile
import subprocess as sp
import numpy as np
import typing as T
import shutil


def find_mat3di():
    """Attempt to locate the mat3di.out file in common FoldSeek installation locations."""
    common_paths = [
        os.path.join(os.path.dirname(shutil.which("foldseek")), "..", "lib", "foldseek", "data", "mat3di.out"),
        os.path.join(os.path.dirname(shutil.which("foldseek")), "..", "share", "foldseek", "data", "mat3di.out"),
        "/usr/local/lib/foldseek/data/mat3di.out",
        "/usr/lib/foldseek/data/mat3di.out",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        "Could not find mat3di.out. Please specify the path manually using the matrix_path parameter. "
        "The file should be in your FoldSeek installation directory under data/mat3di.out"
    )

def get_pssm_from_fasta(
    fasta_content: str,
    foldseek_path: str = "foldseek",
    mmseqs_path: str = "mmseqs",
    matrix_path: str = None
) -> np.ndarray:
    """Generate PSSM from a FASTA string using FoldSeek and MMseqs2.
    
    Args:
        fasta_content: String containing FASTA format data
        foldseek_path: Path to Foldseek executable
        mmseqs_path: Path to MMseqs2 executable
        matrix_path: Path to FoldSeek matrix file. If None, will attempt to locate automatically.
    
    Returns:
        numpy.ndarray: PSSM matrix
    """
    if matrix_path is None:
        matrix_path = find_mat3di()
        
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input FASTA to temporary file
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_path = os.path.join(input_dir, "sequence.fasta")
        with open(input_path, "w") as f:
            f.write(fasta_content)

        # Base paths for databases
        input_db = os.path.join(tmpdir, "inputdb")
        fake_aln_tsv = os.path.join(tmpdir, "fake_aln.tsv")
        fake_aln_db = os.path.join(tmpdir, "fake_aln_db")
        profile_tsv = os.path.join(tmpdir, "profile.tsv")

        # Parameters for MMseqs2
        victor_params = [
            "-pca", "1.4",
            "--pcb", "1.5",
            "--sub-mat", matrix_path,
            "--mask-profile", "0",
            "--comp-bias-corr", "0",
            "--e-profile", "0.1",
            "-e", "0.1",
            "--profile-output-mode", "1",
            "--gap-open", "11",
            "--gap-extend", "1"
        ]

        # Run FoldSeek createdb
        cmd = f"{foldseek_path} createdb {input_dir} {input_db}"
        proc = sp.run(shlex.split(cmd), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FoldSeek createdb failed: {proc.stderr}")

        # Generate fake alignment TSV
        with open(f"{input_db}.index", "r") as f:
            index_content = f.readline().strip().split()
            seq_len = int(index_content[2]) - 2
            fake_aln_line = f"0\t{index_content[1]}\t0\t1.00\t0\t0\t{seq_len-1}\t{seq_len}\t0\t{seq_len-1}\t{seq_len}\t{seq_len}M"
        
        with open(fake_aln_tsv, "w") as f:
            f.write(fake_aln_line)

        # Convert TSV to MMseqs2 database
        cmd = f"{mmseqs_path} tsv2db {fake_aln_tsv} {fake_aln_db} --output-dbtype 5"
        proc = sp.run(shlex.split(cmd), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"MMseqs2 tsv2db failed: {proc.stderr}")

        # Generate profile
        cmd = f"{mmseqs_path} result2profile {input_db}_ss {input_db}_ss {fake_aln_db} {profile_tsv} " + " ".join(victor_params)
        proc = sp.run(shlex.split(cmd), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"MMseqs2 result2profile failed: {proc.stderr}")

        # Read and parse the profile TSV into a numpy array
        try:
            pssm = np.loadtxt(profile_tsv, delimiter='\t')
            return pssm
        except Exception as e:
            raise RuntimeError(f"Failed to parse PSSM from profile TSV: {str(e)}")
