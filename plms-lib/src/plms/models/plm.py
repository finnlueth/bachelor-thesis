from abc import ABC, abstractmethod
import torch
from typing import List, Union, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput


class ProteinLanguageModel(PreTrainedModel, ABC):
    """Abstract base class for protein language model wrappers."""

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        """Initialize the protein language model wrapper.

        Args:
            model_name_or_path: Name or path of the pretrained model

        Keyword Args:
            output_loading_info: Whether to output loading information (default: False)
            _device: Device to load the model on (default: "cuda:0", options: "cuda:0", "cuda:1-n", "cpu", distributed: "auto")
            mean_pooling: Whether to use mean pooling (default: False)
        """
        self.model_name_or_path = model_name_or_path
        super().__init__(self._load_config(), *args, **kwargs)

        self.output_loading_info = kwargs.get("output_loading_info", False)
        self._device = kwargs.get("device", "cuda:0")
        self.mean_pooling = kwargs.get("mean_pooling", False)
        self.trim_value = kwargs.get("trim_value", 0)

        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer()
        self.model: PreTrainedModel = self._load_model()

    @abstractmethod
    def _load_config(self) -> PretrainedConfig:
        """Load the underlying transformer model config.

        Returns:
            The loaded model
        """
        pass

    @abstractmethod
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the underlying tokenizer.

        Returns:
            The loaded tokenizer
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

    @abstractmethod
    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]] = None,
    ) -> Union[torch.Tensor, BaseModelOutput]:
        """Forward pass of the model.

        Args:
            input_ids: Tensor of input IDs
            attention_mask: Tensor of attention masks

        Returns:
            Tensor of last model hidden states (embeddings)
        """
        pass

    @staticmethod
    def trim_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor, trim_value: int = 0) -> torch.Tensor:
        """
        Remove special tokens (pad, cls, sep, eos, etc.) from embeddings.
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_embeddings = torch.where(mask_expanded == 1, hidden_states, trim_value)
        return masked_embeddings

    @staticmethod
    def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling only over tokens where attention mask is 1.
        Args:
            hidden_states: tensor of shape (batch_size, seq_length, hidden_dim)
            attention_mask: tensor of shape (batch_size, seq_length)
        Returns:
            Mean pooled representation of shape (batch_size, hidden_dim)
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_embeddings = hidden_states * mask_expanded
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled
