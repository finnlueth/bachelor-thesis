from typing import List, Optional, Union

import torch
from transformers import PretrainedConfig, T5EncoderModel

from ...configurations import PLMConfig
from ..base_plm import ProteinLanguageModel
from ...utils import modeling_utils
from ...modeling_outputs import ProteinLanguageModelOutput


class ProstT5(ProteinLanguageModel):
    """Wrapper for ProstT5 models."""

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

    def update_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        attention_mask[batch_indices, seq_lengths] = 0
        attention_mask[:, 0] = 0
        return attention_mask[:, 1:]

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

        attention_mask = attention_mask.clone()

        # for ids, mask in zip(input_ids, attention_mask):
        #     print(*[f"{x:<4d}" for x in ids], sep="")
        #     print(*[f"{x:<4d}" for x in mask], sep="")
        # print()
        
        attention_mask = self.update_attention_mask(attention_mask)
        input_ids = input_ids[:, 1:]
        model_outputs["last_hidden_state"] = model_outputs["last_hidden_state"][:, 1:, :]
        

        model_outputs["last_hidden_state"] = modeling_utils.trim_hidden_states(
            model_outputs["last_hidden_state"],
            attention_mask,
            self.config.trim_value,
        )
        
        # print(model_outputs["last_hidden_state"].mean(dim=2).shape)
        
        # for _ids, _mask, _hidden_state_mean in zip(input_ids, attention_mask, model_outputs["last_hidden_state"].abs().sum(dim=2).tolist()):
        #     print(*[f"{x:<6d}" for x in _ids], sep="")
        #     print(*[f"{x:<6d}" for x in _mask], sep="")
        #     print(_mask.sum())
        #     print(*[f"{x:<6.1f}" for x in _hidden_state_mean], sep="")
        # print()

        if self.config.mean_pooling:
            model_outputs["last_hidden_state"] = modeling_utils.mean_pool(model_outputs["last_hidden_state"], attention_mask)

        return ProteinLanguageModelOutput(**model_outputs, mask=attention_mask)
