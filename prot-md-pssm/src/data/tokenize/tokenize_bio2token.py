import typing as T
from .tokenize_trajectory import TrajectoryTokenizer


class Bio2TokenTokenizer(TrajectoryTokenizer):
    def __init__(self):
        super().__init__()

    def load_model(self):
        # Load the Bio2Token model here
        pass

    def tokenize(self, pdb_files: T.List[str]) -> T.List[str]:
        return ['a', 'b']
        # Implement the tokenization logic for Bio2Token
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     pdb_paths = []
        #     for i, content in enumerate(pdb_files):
        #         pdb_path = os.path.join(tmpdir, f"file_{i}.pdb")
        #         with open(pdb_path, 'w') as file:
        #             file.write(content)
        #         pdb_paths.append(pdb_path)

        #     # Implement specific logic for Bio2Token using pdb_paths
        #     # Convert results to a format suitable for HDF5
        #     return []  # Replace with actual data

    def detokenize(self, tokens: T.List[str]) -> T.List[str]:
        pass
        # Implement the detokenization logic for Bio2Token
