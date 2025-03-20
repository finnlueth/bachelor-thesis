from transformers import PretrainedConfig

from ..utils import determine_device


class PLMConfig(PretrainedConfig):
    def __init__(
        self,
        name_or_path: str = None,
        device=determine_device(),
        trim_value: int = 0,
        mean_pooling: bool = False,
        output_loading_info: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name_or_path = name_or_path
        self.trim_value = trim_value
        self.mean_pooling = mean_pooling
        self.device = device
        self.output_loading_info = output_loading_info

