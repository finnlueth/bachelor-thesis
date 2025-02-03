import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    T5Config,
    T5EncoderModel,
    T5PreTrainedModel,
    T5Tokenizer,
    modeling_outputs,
    modeling_utils,
)


class PSSMHead1(nn.Module):
    """Head for PSSM generation from T5 embeddings. based on https://github.com/hefeda/PGP/blob/master/prott5_batch_predictor.py#L144"""
    def __init__(self):
        """
        Args:
            config (MDPSSMConfig): Configuration object for the model
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(1024, 32, kernel_size=7, padding=3),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 20, kernel_size=7, padding=3)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.classifier(x).squeeze(dim=-1)
        x = x.transpose(1, 2)
        pssm = torch.softmax(x, dim=2)
        return pssm


class PSSMHead2(nn.Module):
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

        pssm = torch.softmax(pssm, dim=2)

        return pssm
