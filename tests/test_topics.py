# Tests in faux_lingo/tests/test_topics.py
"""Tests for topic vector space functionality."""

import pytest
import torch

from faux_lingo.core.topics import TopicVectorSpace


def test_init_validation():
    """Test input validation during initialization."""
    with pytest.raises(ValueError):
        # n_topics > vocab_size not allowed
        TopicVectorSpace(n_topics=5, vocab_size=3)


def test_vector_properties():
    """Test that topic vectors have required mathematical properties."""
    space = TopicVectorSpace(n_topics=3, vocab_size=5)
    vectors = space.vectors

    # Test unit length
    norms = torch.linalg.norm(vectors, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))

    # Test orthogonality
    gram = vectors @ vectors.T
    identity = torch.eye(3, device=vectors.device)
    assert torch.allclose(gram, identity, atol=1e-6)


def test_distribution_shape():
    """Test output shape of get_distribution."""
    space = TopicVectorSpace(n_topics=3, vocab_size=5)

    # Single mixture
    mixture = torch.ones(3) / 3  # Uniform mixture
    dist = space.get_distribution(mixture)
    assert dist.shape == (5,)

    # Batch of mixtures
    mixtures = torch.ones(4, 3) / 3  # Batch of uniform mixtures
    dists = space.get_distribution(mixtures)
    assert dists.shape == (4, 5)


def test_save_load(tmp_path):
    """Test serialization of topic vectors."""
    original = TopicVectorSpace(n_topics=3, vocab_size=5)
    path = tmp_path / "topics.pt"

    # Save and load
    original.save(path)
    loaded = TopicVectorSpace.load(path)

    # Verify properties preserved
    assert torch.allclose(original.vectors, loaded.vectors)
    assert original.n_topics == loaded.n_topics
    assert original.vocab_size == loaded.vocab_size
