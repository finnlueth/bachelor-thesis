from transformers import PretrainedConfig

from ..utils import determine_device


class PLMConfig(PretrainedConfig):
    """Configuration for a protein language model.

    Args:
        name_or_path: Name or path of the pretrained model
        device: Device to load the model on
        trim_value: Value to replace unmasked embeddings with
        output_loading_info: Whether to output loading information
    """

    def __init__(
        self,
        name_or_path: str = None,
        device=determine_device(),
        trim_value: int = 0,
        output_loading_info: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name_or_path = name_or_path
        self.trim_value = trim_value
        self.device = device
        self.output_loading_info = output_loading_info
