from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ..configurations.configuration_base_plm import PLMConfig


class ProteinLanguageModel(PreTrainedModel, ABC):
    """Abstract base class for protein language model wrappers."""

    def __init__(self, name_or_path: str = None, config: Union[PLMConfig, str] = None, *args, **kwargs):
        """Initialize the protein language model wrapper.

        Args:
            config: Configuration of the pretrained model

        Keyword Args:
            output_loading_info: Whether to output loading information (default: False)
            _device: Device to load the model on (default: "cuda:0", options: "cuda:0", "cuda:1-n", "cpu", distributed: "auto")
            mean_pooling: Whether to use mean pooling (default: False)
        """
        if name_or_path is not None and config is not None:
            raise ValueError("Provide either a name_or_path or a config, not both.")

        if config is None:
            config: PLMConfig = self.get_default_config(name_or_path=name_or_path)

        super().__init__(config, *args, **kwargs)
        self.model: PreTrainedModel = self._load_model()

    @abstractmethod
    def get_default_config(name_or_path: str) -> PretrainedConfig:
        """Load the underlying transformer model config.

        Returns:
            The model config
        """
        pass

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Load the underlying transformer model.

        Returns:
            The loaded model
        """
        pass

    @abstractmethod
    def update_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Update the attention mask to ignore special tokens (pad, cls, sep, eos, etc.).
        """
        pass

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]] = None,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, BaseModelOutput]:
        """Forward pass of the model.

        Args:
            input_ids: Tensor of input IDs
            attention_mask: Tensor of attention masks

        Returns:
            Dictionary of model outputs
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.model.forward(input_ids, attention_mask, *args, **kwargs)

        if self.mean_pooling:
            outputs['last_hidden_state'] = self.mean_pooling(outputs['last_hidden_state'], attention_mask)

        return outputs

    def _get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get the embeddings from the model."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            hidden_states = outputs.last_hidden_state
        return hidden_states
