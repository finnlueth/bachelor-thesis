import os
import h5py
import torch
from .trajectory_dataset import TrajectoryDataset, TrajectoryDataWrapper

class MDCATHDataset(TrajectoryDataset):
    def _load_trajectory_locations(self):
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".h5")]

    def _get_item(self, idx):
        with h5py.File(self.trajectory_locations[idx], "r") as file:
            trajectory = list(file.keys())[0]
        return trajectory
