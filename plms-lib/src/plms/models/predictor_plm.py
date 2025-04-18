from abc import ABC, abstractmethod
from typing import List

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import TORCH_INIT_FUNCTIONS

from plms.models import auto_model


class ProteinLanguageModelPredictor(PreTrainedModel, ABC):
    """Abstract base class for protein language model predictors."""

    def __init__(self, config: PretrainedConfig = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.protein_encoder = auto_model(config.encoder_name_or_path)

        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)
        self.post_init()

    @abstractmethod
    def get_modules_to_save(self) -> List[str]:
        """Get the modules to save."""
        pass

    def post_init(self):
        """Post-initialization hook."""
        super().post_init()
