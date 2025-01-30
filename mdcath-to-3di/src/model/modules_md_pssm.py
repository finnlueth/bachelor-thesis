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
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1024, 32, kernel_size=7, padding=3),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.dssp3_classifier = torch.nn.Sequential(
            nn.Conv1d(32, 3, kernel_size=7, padding=3)  # 7
        )

        self.classifier = torch.nn.Sequential(nn.Conv1d(32, 20, kernel_size=7, padding=3))

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.feature_extractor(x)  # OUT: (B x 32 x L x 1)
        pssm = self.classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 20)
        pssm = torch.softmax(pssm, dim=2)
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
