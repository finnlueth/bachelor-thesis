"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch

import random


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return ["MDCATH", "PROTEIN", "ROSTLAB"]


@pytest.fixture
def dummy_embeddings():
    """Return dummy embeddings tensor for testing."""
    return torch.randn(2, 6, 1024)  # batch_size=2, seq_len=6, hidden_dim=1024


@pytest.fixture
def dummy_attention_mask():
    """Return dummy attention mask for testing."""
    return torch.ones(2, 6)  # batch_size=2, seq_len=6
