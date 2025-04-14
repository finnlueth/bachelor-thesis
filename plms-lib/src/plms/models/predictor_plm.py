from abc import ABC
from typing import List

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import TORCH_INIT_FUNCTIONS

from plms.models import auto_model


class ProteinLanguageModelPredictor(PreTrainedModel, ABC):
    """Abstract base class for protein language model predictors."""

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.protein_encoder = auto_model(config.encoder_name_or_path)

        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)
        self.post_init()

    def get_modules_to_save(self) -> List[str]:
        """Get the modules to save."""
        modules = []
        for name, _ in self.named_modules():
            if (
                "protein_encoder" not in name
                and hasattr(self, name)
                and hasattr(getattr(self, name.split(".")[0]), "weight")
                and isinstance(getattr(self, name), torch.nn.Module)
            ):
                modules.append(name)
        return modules

    def post_init(self):
        """Post-initialization hook."""
        super().post_init()
