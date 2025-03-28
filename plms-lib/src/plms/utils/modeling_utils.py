from typing import List, Optional, Union

import torch


def mean_pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs mean pooling only over tokens where attention mask is 1.
    Args:
        hidden_states: tensor of shape (batch_size, seq_length, hidden_dim)
        attention_mask: tensor of shape (batch_size, seq_length)
    Returns:
        Mean pooled representation of shape (batch_size, hidden_dim)
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"Expected hidden_states to have 3 dimensions (batch_size, seq_length, hidden_dim), got {hidden_states.dim()}")

    if attention_mask is None:
        return torch.mean(hidden_states, dim=1)

    if attention_mask.dim() != 2:
        raise ValueError(f"Expected attention_mask to have 2 dimensions (batch_size, seq_length), got {attention_mask.dim()}")

    if hidden_states.size(0) != attention_mask.size(0) or hidden_states.size(1) != attention_mask.size(1):
        raise ValueError(
            f"Shape mismatch: hidden_states {hidden_states.shape} and attention_mask {attention_mask.shape} "
            "must have matching batch_size and seq_length dimensions"
        )

    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    masked_embeddings = hidden_states * mask_expanded

    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    sum_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)

    mean_pooled = sum_embeddings / sum_mask
    return mean_pooled


def trim_attention_mask(
    attention_mask: Union[torch.Tensor, List[int]], trim_beginning: int = 0, trim_end: int = 0
) -> torch.Tensor:
    """
    Finds indices of first n and last m 1s in attention mask and sets them to 0.

    Args:
        attention_mask: tensor of shape (batch_size, seq_length)
        trim_beginning: number of 1s to trim from beginning
        trim_end: number of 1s to trim from end
    Returns:
        Modified attention mask with first n and last m 1s set to 0
    """
    if trim_beginning == 0 and trim_end == 0:
        return attention_mask

    if trim_beginning > 0:
        cumsum_forward = torch.cumsum(attention_mask, dim=1)
        beginning_mask = cumsum_forward > trim_beginning
        attention_mask = attention_mask * beginning_mask

    if trim_end > 0:
        cumsum_backward = torch.cumsum(attention_mask.flip(dims=[1]), dim=1).flip(dims=[1])
        end_mask = cumsum_backward > trim_end
        attention_mask = attention_mask * end_mask

    return attention_mask


def rotate_padding_side(hidden_states: torch.Tensor, attention_mask: Union[torch.Tensor, List[List[int]]]) -> torch.Tensor:
    """
    Rotates padding side of embeddings. This is necessary for models that use padding to the left.
    Args:
        hidden_states: tensor of shape (batch_size, seq_length, hidden_dim)
        attention_mask: tensor of shape (batch_size, seq_length)
    Returns:
        Adjusted hidden states with same shape but meaningful tokens at start
    """
    pad_lengths = (~attention_mask.bool()).sum(dim=1)
    seq_length = hidden_states.size(1)

    indices = torch.arange(seq_length, device=hidden_states.device)
    indices = indices.expand(hidden_states.size(0), -1)
    rolled_indices = (indices + pad_lengths.unsqueeze(1)) % seq_length

    adjusted_hidden_states = torch.gather(hidden_states, 1, rolled_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
    adjusted_attention_mask = torch.gather(attention_mask, 1, rolled_indices)

    return adjusted_hidden_states, adjusted_attention_mask


def trim_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor, trim_value: int = 0) -> torch.Tensor:
    """Remove special tokens from embeddings.

    Args:
        hidden_states: tensor of shape (batch_size, seq_length, hidden_dim)
        attention_mask: tensor of shape (batch_size, seq_length)
        trim_value:
    Returns:
        Trimmed embeddings
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    masked_embeddings = torch.where(mask_expanded == 1, hidden_states, trim_value)
    return masked_embeddings
