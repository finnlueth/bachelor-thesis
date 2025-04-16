from transformers import PretrainedConfig
from dataclasses import dataclass

@dataclass
class PLMConfigForPSSM(PretrainedConfig):
    def __init__(
        self,
        encoder_name_or_path: str,
        num_labels: int,
        hidden_size: int,
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_name_or_path = encoder_name_or_path
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout = dropout
