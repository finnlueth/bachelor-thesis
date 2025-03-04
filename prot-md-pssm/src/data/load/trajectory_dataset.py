import json
import os
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import fcntl
import h5py
from torch.utils.data import Dataset
from src.utils.errors import TrajectoryAlreadyProcessedError

class TrajectoryDataset(Dataset, ABC):
    def __init__(self, dataset_name, data_dir, save_path, cache_name="used_trajectory_locations"):
        """
        Args:
            data_dir (str): Directory with all the data files.
            transform (callable, optional): Optional transform to be applied
            on a data sample.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.save_path = save_path
        self.trajectory_locations = self._load_trajectory_locations()
        self.cache_name = cache_name

        os.makedirs(self.save_path, exist_ok=True)
        if not os.path.exists(os.path.join(self.save_path, f"{self.cache_name}.json")):
            with open(os.path.join(self.save_path, f"{self.cache_name}.json"), "w") as f:
                json.dump([], f)

    @abstractmethod
    def _load_trajectory_locations(self):
        """Load and return a list of file paths, subdirectories or other locations from the data directory."""
        pass

    @abstractmethod
    def _get_item(self, idx):
        """Retrieve a data sample for the given index."""
        pass

    def __len__(self):
        return len(self.trajectory_locations)

    def __getitem__(self, idx):
        """Retrieve a data sample for the given index."""
        with open(self.save_path + f"/{self.cache_name}.json", "r") as f:
            used_trajectory_locations = json.load(f)
        if self.trajectory_locations[idx] in used_trajectory_locations:
            raise TrajectoryAlreadyProcessedError(self.trajectory_locations[idx])
        return self._get_item(idx)


@dataclass
class TrajectoryWrapper:
    name: str
    sequence: str
    structure: str
    trajectories: T.Dict[str, T.List[str]] = field(default_factory=dict)
    trajectory_pdbs: T.Dict[str, T.List[str]] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)
