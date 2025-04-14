from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput

from plms import (
    PLMConfig,
    ProteinLanguageModelPredictor,
)

class PLMForTokenClassification(ProteinLanguageModelPredictor):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        print("input_ids")
        print(input_ids.shape)
        print(*[str(x)[:4].ljust(3) for x in input_ids[0].tolist()], sep=" ")
        print("attention_mask")
        print(attention_mask.shape)
        print(*[str(x)[:4].ljust(3) for x in attention_mask[0].tolist()], sep=" ")
        print("labels")
        print(labels.shape)
        print(*[str(x)[:4].ljust(3) for x in labels[0].tolist()], sep=" ")

        outputs = self.protein_encoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs["last_hidden_state"]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        print("hidden_states")
        print(hidden_states.shape)
        print(*[str(x)[:4].ljust(3) for x in hidden_states[0].tolist()], sep=" ")
        print("outputs['mask']")
        print(outputs["mask"].shape)
        print(*[str(x)[:4].ljust(3) for x in outputs["mask"][0].tolist()], sep=" ")
        print("logits")
        print(logits.shape)
        print(*[str(x)[:4].ljust(3) for x in logits[0].argmax(dim=-1).tolist()], sep=" ")
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PLMConfigForTokenClassification(PLMConfig):
    def __init__(
        self,
        encoder_name_or_path: str = None,
        num_labels: int = None,
        hidden_size: int = None,
        classifier_dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        if num_labels is None:
            raise ValueError("num_labels must be provided")
        if hidden_size is None:
            raise ValueError("hidden_size must be provided")

        super().__init__(*args, **kwargs)
        self.encoder_name_or_path = encoder_name_or_path
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.hidden_size = hidden_size
