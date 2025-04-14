import torch

def determine_device() -> str:
    """Determine the device to use for the model.

    Returns:
        The device to use for the model
    """
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
