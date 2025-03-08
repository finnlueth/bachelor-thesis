"""
Tests for the base ProteinLanguageModel class.
"""
import pytest
import torch
from plms.models.plm import ProteinLanguageModel


def test_mean_pooling(dummy_embeddings, dummy_attention_mask):
    """Test the mean pooling functionality."""
    result = ProteinLanguageModel.mean_pooling(dummy_embeddings, dummy_attention_mask)
    
    # Check output shape
    assert result.shape == (dummy_embeddings.size(0), dummy_embeddings.size(2))
    
    # Check if mean pooling is correct for a simple case
    test_embeddings = torch.ones(2, 3, 4)  # batch_size=2, seq_len=3, hidden_dim=4
    test_mask = torch.ones(2, 3)
    test_mask[:, -1] = 0  # mask out last token
    
    result = ProteinLanguageModel.mean_pooling(test_embeddings, test_mask)
    expected = torch.ones(2, 4)  # should still be all ones due to proper averaging
    torch.testing.assert_close(result, expected)


def test_trim_hidden_states(dummy_embeddings, dummy_attention_mask):
    """Test the trimming of hidden states."""
    # Create a mask that ignores the last token
    mask = dummy_attention_mask.clone()
    mask[:, -1] = 0
    
    result = ProteinLanguageModel.trim_hidden_states(dummy_embeddings, mask)
    
    # Check that masked positions are zeroed out
    assert torch.all(result[:, -1] == 0)
    
    # Original embeddings should be unchanged where mask is 1
    torch.testing.assert_close(
        result[:, :-1],
        dummy_embeddings[:, :-1]
    ) 