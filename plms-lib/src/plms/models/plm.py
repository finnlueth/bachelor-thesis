from abc import ABC, abstractmethod
import torch
from typing import List, Union, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput


class ProteinLanguageModel(PreTrainedModel, ABC):
    """Abstract base class for protein language model wrappers."""

    def __init__(self, model_name: str, *args, **kwargs):
        """Initialize the protein language model wrapper.

        Args:
            model_name: Name or path of the pretrained model
            device: Device to load the model on ('cpu', 'cuda', etc.)
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.model: PreTrainedModel = self._load_model()
        self.tokenizer = self._load_tokenizer()

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Load the underlying transformer model.

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
    def trim_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Remove special tokens (pad, cls, sep, eos, etc.) from embeddings.
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids: Tensor of input IDs
            attention_mask: Tensor of attention masks

        Returns:
            Tensor of last model hidden states (embeddings)
        """
        pass
