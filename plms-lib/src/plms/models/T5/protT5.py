from ..plm import ProteinLanguageModel
from transformers import T5EncoderModel, T5Tokenizer, PreTrainedTokenizer, PreTrainedModel, PretrainedConfig
import torch
from typing import Union, Optional, List


class ProtT5(ProteinLanguageModel):
    """Wrapper for ProtT5 models."""

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        super().__init__(model_name_or_path, *args, **kwargs)

    def _load_config(self) -> PretrainedConfig:
        return PretrainedConfig(name_or_path=self.model_name_or_path)

    def get_tokenizer(self) -> T5Tokenizer:
        tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            do_lower_case=False,
            use_fast=True,
            legacy=False,
        )
        return tokenizer

    def _load_model(self) -> T5EncoderModel:
        model = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            device_map=self._device,
            output_loading_info=self.output_loading_info,
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
