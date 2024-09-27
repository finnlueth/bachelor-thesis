import h5py
from Bio.PDB import PDBParser
import io
import time


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


class CodeTimer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
