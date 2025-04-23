import torch
import torch.nn as nn
from torch.nn import KLDivLoss
from typing import Optional, Tuple, Union, List

from plms import ProteinLanguageModelPredictor
from . import PLMConfigForPSSM, PSSMOutput

class PSSMHeadLinear(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_classes: int = 20):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class PSSMHead(nn.Module):
    """Head for PSSM generation from T5 embeddings. based on https://github.com/hefeda/PGP/blob/master/prott5_batch_predictor.py#L144"""

    def __init__(
        self,
        num_classes: int = 20,
        dropout: float = 0.25,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(hidden_size, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, num_classes, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.classifier(x)
        x = x.transpose(1, 2)
        pssm = torch.softmax(x, dim=2)
        return pssm


class PLMForPssmGeneration(ProteinLanguageModelPredictor):
    def __init__(self, config: PLMConfigForPSSM, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
        self.pssm_head = PSSMHead(
            num_classes=config.num_labels,
            dropout=config.dropout,
            hidden_size=config.hidden_size,
        )
        self.loss_fct = KLDivLoss(reduction="batchmean")

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
    ) -> Union[Tuple[torch.Tensor], PSSMOutput]:
        encoder_outputs = self.protein_encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs["last_hidden_state"]
        attention_mask = encoder_outputs["mask"]

        # [batch_size, seq_len, 20]
        pssm = self.pssm_head(hidden_states)

        loss = None
        if labels is not None:
            # [batch_size * seq_len, 20]
            pred = pssm.flatten(end_dim=1)
            target = labels.flatten(end_dim=1)

            pred_mask = attention_mask.flatten(end_dim=1)
            target_mask = ~torch.any(target == -100, dim=1)

            pred = pred[pred_mask.bool()]
            target = target[target_mask.bool()]

            # print(pred.shape, target.shape)

            loss = self.loss_fct(torch.log(pred), target)

        if not return_dict:
            output = (pssm, encoder_outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return PSSMOutput(
            loss=loss,
            pssms=pssm,
            hidden_states=encoder_outputs["last_hidden_state"] if output_hidden_states else None,
            masks=attention_mask,
        )

    def get_modules_to_save(self) -> List[str]:
        return ["pssm_head"]
