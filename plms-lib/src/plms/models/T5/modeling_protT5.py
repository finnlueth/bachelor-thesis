from typing import List, Optional, Union

import torch
from transformers import T5EncoderModel

from ...configurations import PLMConfig
from ..base_plm import ProteinLanguageModel
from ...utils import modeling_utils
from ...modeling_outputs import ProteinLanguageModelOutput

class ProtT5(ProteinLanguageModel):
    """Wrapper for ProtT5 models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_config(self, name_or_path: str) -> PLMConfig:
        return PLMConfig(name_or_path=name_or_path)

    def _load_model(self) -> T5EncoderModel:
        model = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=self.config.name_or_path,
            device_map=self.config.device,
            output_loading_info=self.config.output_loading_info,
            torch_dtype="auto",
        )
        return model

    def update_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        attention_mask[batch_indices, seq_lengths] = 0
        return attention_mask

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        attention_mask = self.update_attention_mask(attention_mask)

        model_outputs["last_hidden_state"] = modeling_utils.trim_hidden_states(
            model_outputs["last_hidden_state"],
            attention_mask,
            self.config.trim_value,
        )

        if self.config.mean_pooling:
            model_outputs["last_hidden_state"] = modeling_utils.mean_pool(model_outputs["last_hidden_state"], attention_mask)

        return ProteinLanguageModelOutput(**model_outputs, masks=attention_mask)
