from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class ProteinLanguageModelOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    masks: Optional[Tuple[torch.FloatTensor, ...]] = None
