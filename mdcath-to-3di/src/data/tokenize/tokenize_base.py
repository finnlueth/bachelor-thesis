import h5py
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self):
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the tokenizer model into memory."""
        pass

    @abstractmethod
    def tokenize(self, pdb_files: T.List[str], output_path: str):
        """Tokenize the input data."""
        pass

def save_to_h5(data, output_path, dataset_name="tokenized_data"):
    """Save tokenized data to an HDF5 file."""
    with h5py.File(output_path, 'w') as h5file:
        h5file.create_dataset(dataset_name, data=data)
