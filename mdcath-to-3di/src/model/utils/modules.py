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


class T5PSSMHead1(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()

        config.d_model = 1024

        self.conv1 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.bn1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.bn2(hidden_states)
        hidden_states = torch.relu(hidden_states)
        # hidden_states = self.global_pool(hidden_states)
        # hidden_states = hidden_states.view(hidden_states.size(0), -1)
        return hidden_states


class T5PSSMHead2(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        embedding_size = 1024
        num_3Di_classes = 27
        num_filters = 256
        kernel_size = 5

        self.conv1 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same",
        )

        self.batch_norm = nn.BatchNorm1d(num_filters)

        self.fc = nn.Linear(num_filters, num_3Di_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.fc(hidden_states)