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


class T5PSSMHead(nn.Module):
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
        x = x.transpose(1, 2)

        # Apply CNN layers with ReLU activation and mask
        x = F.relu(self.conv1(x))
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        # Transpose to [batch_size, seq_len, channels]
        x = x.transpose(1, 2)

        # Project to 20 dimensions for amino acid probabilities
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

        self.pssm_head = T5PSSMHead(config)
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

        hidden_states = encoder_outputs["last_hidden_state"]
        attention_mask = trim_attention_mask(attention_mask, trim_end=1)

        # print("input_ids", input_ids.shape)
        # print("input_ids", input_ids)
        # print("attention_mask", attention_mask.shape)
        # print("labels", labels.shape)
        # print("hidden_states", hidden_states.shape)

        logits = self.pssm_head(hidden_states)
        logits = torch.softmax(logits, dim=-1)

        # print("logits", logits.shape)
        # for x, y in zip(logits.tolist()[:1], labels.tolist()[:1]):
        #     display(pd.DataFrame(x))
        #     print(x)
        #     display(pd.DataFrame(y))
        #     print(y)
        #     print("Sum of first row:", sum(x[0]))
        #     print("Sum of first row:", sum(y[0]))
        #     print("-" * 100)

        loss = None
        if labels is not None:
            mask = None
            if mask is None:
                mask = ~torch.any(labels == -100, dim=2)
            loss_fct = KLDivLoss(reduction="batchmean")

            masked_logits = logits[mask]
            masked_labels = labels[mask]
            masked_logits = torch.log(masked_logits)

            loss = loss_fct(masked_logits, masked_labels)

        print("loss", loss)

        if not return_dict:
            output = (logits, encoder_outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
