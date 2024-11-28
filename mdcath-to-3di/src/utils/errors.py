class TrajectoryAlreadyProcessedError(Exception):
    """Exception raised when attempting to process a trajectory that has already been processed."""
    def __init__(self, trajectory_name: str):
        self.trajectory_name = trajectory_name
        self.message = f"Trajectory {trajectory_name} has already been accessed and processed."
        super().__init__(self.message)