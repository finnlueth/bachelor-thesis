"""
Pytest configuration and shared fixtures.
"""
import pytest
import torch


@pytest.fixture
def device():
    """Return the default device for testing."""
    return "cpu"


@pytest.fixture
def sample_sequence():
    """Return a sample protein sequence for testing."""
    return "MDCATH"


@pytest.fixture
def sample_batch():
    """Return a sample batch of protein sequences for testing."""
    return ["MDCATH", "PROTEIN"]


@pytest.fixture
def dummy_embeddings():
    """Return dummy embeddings tensor for testing."""
    return torch.randn(2, 6, 1024)  # batch_size=2, seq_len=6, hidden_dim=1024


@pytest.fixture
def dummy_attention_mask():
    """Return dummy attention mask for testing."""
    return torch.ones(2, 6)  # batch_size=2, seq_len=6 