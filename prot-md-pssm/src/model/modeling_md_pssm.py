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
from src.model.modules_md_pssm import PSSMHead1 as PSSMHead


class T5EncoderModelForPssmGeneration(PreTrainedModel):
    def __init__(self, config: MDPSSMConfig):
        super().__init__(config=config)
        device_map = config.device if hasattr(config, "device") else "auto"
        self.protein_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path="Rostlab/prot_t5_xl_uniref50",
            device_map=device_map,
            output_loading_info=False,
            torch_dtype="auto",
        )

        self.pssm_head = PSSMHead()

        self.loss_fct = KLDivLoss(reduction="batchmean")

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

        # Attention mask ignores EOS token
        attention_mask = attention_mask.clone()
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        attention_mask[batch_indices, seq_lengths] = 0

        hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        # [batch_size, seq_len, 20]
        pssm = self.pssm_head(hidden_states)

        loss = None
        if labels is not None:
            # [batch_size * seq_len, 20]
            target = labels.flatten(end_dim=1)
            pred = pssm.flatten(end_dim=1)

            mask = ~torch.any(target == -100, dim=1)

            pred = pred[mask]
            target = target[mask]

            loss = self.loss_fct(torch.log(pred), target)

        if not return_dict:
            output = (pssm, encoder_outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=pssm,
            hidden_states=encoder_outputs["last_hidden_state"],
        )
