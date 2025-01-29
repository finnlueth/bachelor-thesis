# import src.utils.logging as logging
# import src.models.utils.modules as modules

import copy
import inspect
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display
from torch import nn
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    KLDivLoss,
    MSELoss,
)
from transformers import (
    PreTrainedModel,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
    modeling_outputs,
    modeling_utils,
)
from transformers.modeling_outputs import TokenClassifierOutput

from src.model.configuration_md_pssm import MDPSSMConfig

from plms.models.utils import trim_attention_mask


class PSSMHead(nn.Module):
    """Head for PSSM generation from T5 embeddings."""

    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.final = nn.Linear(128, 20)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Embeddings from T5 [batch_size, seq_len, hidden_dim]

        Returns:
            torch.Tensor: PSSM predictions [batch_size, seq_len, 20]
        """
        # Transpose to [batch_size, hidden_dim, seq_len]
        # Conv1D needs channel dimension (hidden_dim) to be before the sequence length dimension
        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        # Transpose to [batch_size, seq_len, channels]
        # Back to the original shape
        x = x.transpose(1, 2)

        # [batch_size, seq_len, 20]
        pssm = self.final(x)

        return pssm


class T5EncoderModelForPssmGeneration(PreTrainedModel):
    def __init__(self, config: MDPSSMConfig):
        super().__init__(config=config)

        self.protein_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path="Rostlab/prot_t5_xl_uniref50",
            device_map="auto",
            output_loading_info=False,
            torch_dtype="auto",
        )

        self.pssm_head = PSSMHead(config)
        self.pssm_head.to(self.device)

        for name, init_func in modeling_utils.TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        encoder_outputs = self.protein_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        # [batch_size, seq_len, hidden_dim]
        hidden_states = encoder_outputs["last_hidden_state"]

        # [batch_size, seq_len, 20]
        logits = self.pssm_head(hidden_states)

        logits = torch.softmax(logits, dim=2)

        loss = None
        if labels is not None:
            # [batch_size * seq_len, 20]
            tensor_truth = labels.flatten(end_dim=1) + 1e-10
            tensor_pred = logits.flatten(end_dim=1) + 1e-10

            mask = ~torch.any(tensor_truth == -100, dim=1)

            loss_fct = KLDivLoss(reduction="batchmean")

            tensor_pred = tensor_pred[mask]
            tensor_truth = tensor_truth[mask]

            loss_1 = loss_fct(torch.log(tensor_pred), tensor_truth)
            loss_2 = loss_fct(torch.log(tensor_truth), tensor_pred)

            loss = loss_1 + loss_2

        if not return_dict:
            output = (logits, encoder_outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
