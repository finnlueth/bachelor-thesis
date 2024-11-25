import json
import os
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import h5py
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset, ABC):
    def __init__(self, data_dir, save_path, cache_name="used_trajectory_locations"):
        """
        Args:
            data_dir (str): Directory with all the data files.
            transform (callable, optional): Optional transform to be applied
            on a data sample.
        """
        self.data_dir = data_dir
        self.save_path = save_path
        self.trajectory_locations = self._load_trajectory_locations()
        self.used_trajectory_locations = set()
        self.blocked_trajectory_locations = set()
        self.cache_name = cache_name

        if os.path.exists(self.save_path):
            if os.path.exists(os.path.join(self.save_path, f"{self.cache_name}.json")):
                self._load_used_trajectory_locations()
            else:
                self._save_used_trajectory_locations()
        else:
            os.makedirs(self.save_path, exist_ok=True)
            self._save_used_trajectory_locations()

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
        if self.trajectory_locations[idx] in self.used_trajectory_locations:
            raise ValueError(f"Trajectory {self.trajectory_locations[idx]} has already been accessed and processed.")
        if self.trajectory_locations[idx] in self.blocked_trajectory_locations:
            raise ValueError(f"Trajectory {self.trajectory_locations[idx]} has been blocked and cannot be accessed.")
        self.blocked_trajectory_locations.add(self.trajectory_locations[idx])
        return self._get_item(idx)

    def use_trajectory_location(self, idx):
        """
        Mark a trajectory location as used.
        """
        self.blocked_trajectory_locations.discard(self.trajectory_locations[idx])
        self.used_trajectory_locations.add(self.trajectory_locations[idx])
        self._save_used_trajectory_locations()

    def reset(self):
        """
        Reset the used indices to allow reusing the dataset.
        """
        self.used_trajectory_locations.clear()
        self.blocked_trajectory_locations.clear()
        self._save_used_trajectory_locations()

    def get_unused_indices(self):
        """
        Get a list of indices that have not been accessed yet.
        """
        return [
            i
            for i in range(len(self.trajectory_locations))
            if self.trajectory_locations[i] not in self.used_trajectory_locations
        ]

    def is_used_index(self, idx):
        """
        Check if a specific index has been accessed.
        """
        return self.trajectory_locations[idx] in self.used_trajectory_locations

    def _save_used_trajectory_locations(self):
        """
        Save the used indices to a JSON file.
        """
        with open(self.save_path + "/used_trajectory_locations.json", "w") as f:
            json.dump(list(self.used_trajectory_locations), f)

    def _load_used_trajectory_locations(self):
        """
        Load the used indices from a JSON file.
        """
        with open(self.save_path + "/used_trajectory_locations.json", "r") as f:
            self.used_trajectory_locations = set(json.load(f))

    def load_used_trajectory_locations(self, file_path):
        """
        Load used indices from a specified file and update the dataset.
        :param file_path: Path to the file containing the used indices.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                loaded_indices = set(json.load(f))
                self.used_trajectory_locations.update(loaded_indices)
                self._save_used_trajectory_locations()  # Save the combined result to the default save_path
        else:
            raise FileNotFoundError(f"File {file_path} does not exist.")


@dataclass
class TrajectoryWrapper:
    name: str
    sequence: str
    structure: str
    trajectories: T.Dict[str, T.List[str]] = field(default_factory=dict)
    trajectory_pdbs: T.Dict[str, T.List[str]] = field(default_factory=dict)
    
    def __getitem__(self, item):
        return getattr(self, item)
    
    # def to_h5(self, file_path):
    #     """
    #     Converts the TrajectoryDataWrapper to an HDF5 file.
    #     :param file_path: Path to the HDF5 file to be created.
    #     """
    #     with h5py.File(file_path, "w") as file:
    #         for trajectory_type, trajectories in self.trajectories.items():
    #             group = file.create_group(trajectory_type)
    #             for trajectory_name, trajectory_data in trajectories.items():
    #                 group.create_dataset(trajectory_name, data=trajectory_data)
            
    #         for pdb_type, pdbs in self.trajectroy_pdbs.items():
    #             group = file.create_group(pdb_type)
    #             for pdb_name, pdb_data in pdbs.items():
    #                 group.create_dataset(pdb_name, data=pdb_data)
                    
    # def from_h5(self, file_path):
    #     """
    #     Converts an HDF5 file to a TrajectoryDataWrapper.
    #     """
    #     with h5py.File(file_path, "r") as file:
    #         for trajectory_type, trajectories in self.trajectories.items():
    #             group = file.get(trajectory_type)
    #             for trajectory_name, trajectory_data in trajectories.items():
    #                 group.get(trajectory_name)
