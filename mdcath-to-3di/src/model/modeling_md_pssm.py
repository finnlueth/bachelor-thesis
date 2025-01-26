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

from src.model.modules_md_pssm import T5PSSMHead4 as T5PSSMHead
from src.model.configuration_md_pssm import MDPSSMConfig


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

        # print("input_ids", input_ids.shape)
        # print("input_ids", type(input_ids))
        # print("input_ids", input_ids)
        # print(*input_ids.tolist(), sep="\n")
        # print("attention_mask", attention_mask.shape)
        # print(*attention_mask.tolist(), sep="\n")
        # print("labels", labels.shape)

        # for x in labels.tolist():
        #     display(pd.DataFrame(x))

        encoder_outputs = self.protein_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        print("encoder_outputs", encoder_outputs.shape)

        hidden_states = encoder_outputs["last_hidden_state"][:, :-1]
        # print("hidden_states", hidden_states.shape)

        logits = self.pssm_head(hidden_states)

        # print("logits", logits.shape)
        # for x in logits.tolist():
        #     display(pd.DataFrame(x))

        loss = None
        if labels is not None:
            loss_fct = KLDivLoss(reduction="batchmean")
            logits = torch.log_softmax(logits, dim=-1)
            labels = torch.softmax(labels, dim=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1)))

        if not return_dict:
            output = (logits, encoder_outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return None
        # return TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )