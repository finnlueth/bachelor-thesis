from typing import List, Optional, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, T5EncoderModel, T5Tokenizer

from ..base_plm import ProteinLanguageModel
from ...configurations import PLMConfig


class ProtT5(ProteinLanguageModel):
    """Wrapper for ProtT5 models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_config(self, name_or_path: str) -> PretrainedConfig:
        return PLMConfig(name_or_path=name_or_path)

    def _load_model(self) -> T5EncoderModel:
        model = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=self.config.name_or_path,
            device_map=self.config.device,
            output_loading_info=self.config.output_loading_info,
            torch_dtype="auto",
        )
        return model

    def update_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        attention_mask[batch_indices, seq_lengths] = 0
        return attention_mask

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]

        attention_mask = self.update_attention_mask(attention_mask)

        hidden_states = self.trim_hidden_states(
            hidden_states,
            attention_mask,
            self.trim_value,
        )

        if self.mean_pooling:
            hidden_states = self.mean_pooling(hidden_states, attention_mask)

        return hidden_states
