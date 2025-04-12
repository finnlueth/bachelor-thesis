from dataclasses import dataclass
from transformers import PretrainedConfig

@dataclass
class MDPSSMConfig(PretrainedConfig):
    model_type = "md-pssm-t5"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    