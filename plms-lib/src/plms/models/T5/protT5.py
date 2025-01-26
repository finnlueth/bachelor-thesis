from ..plm import ProteinLanguageModel
from transformers import T5EncoderModel, T5Tokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Union, Optional, List


class ProtT5(ProteinLanguageModel):
    """Wrapper for ProtT5 model."""

    def __init__(self, model_name: str, *args, **kwargs):
        config = T5EncoderModel.from_pretrained(model_name).config
        super().__init__(config, *args, **kwargs)
        
        self.model_name = model_name
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self) -> T5EncoderModel:
        model_plm, loading_info_plm = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            device_map="auto",
            output_loading_info=True,
            torch_dtype="auto",
        )
        return model_plm

    def _load_tokenizer(self) -> T5Tokenizer:
        tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            do_lower_case=False,
            use_fast=True,
            legacy=False,
        )
        return tokenizer

    def trim_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return embeddings

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]] = None,
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
