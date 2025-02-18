# faux_lingo/tests/test_dataset.py
"""Tests for dataset generation functionality."""

import pytest
import torch

from faux_lingo.core.generator import SequenceGenerator
from faux_lingo.data.dataset import DatasetConfig, SequenceDataset


@pytest.fixture
def simple_generator():
    """Create a simple generator for testing."""
    return SequenceGenerator.create_uniform(
        vocab_size=9,
        n_topics=2,
        color_fractions=[1, 1, 1],  # Three equal color classes
    )


@pytest.fixture
def simple_config():
    """Create a basic dataset configuration."""
    return DatasetConfig(
        batch_size=4,
        seq_length=10,
        n_batches=3,
        seed=42,
    )


def test_dataset_iteration(simple_generator, simple_config):
    """Test basic dataset iteration."""
    dataset = SequenceDataset(simple_generator, simple_config)

    # Check total length
    assert len(dataset) == simple_config.n_batches

    # Check batch shapes
    batches = list(dataset)
    assert len(batches) == simple_config.n_batches
    for batch in batches:
        assert batch.tokens.shape == (simple_config.batch_size, simple_config.seq_length)
        assert batch.topic_mixtures.shape == (simple_config.batch_size, dataset.n_topics)


def test_reproducibility(simple_generator, simple_config):
    """Test that sequence generation is reproducible with same seed."""
    dataset1 = SequenceDataset(simple_generator, simple_config)
    batch1 = next(iter(dataset1))

    dataset2 = SequenceDataset(simple_generator, simple_config)
    batch2 = next(iter(dataset2))

    assert torch.all(batch1.tokens == batch2.tokens)
    assert torch.allclose(batch1.log_probs, batch2.log_probs)
    assert torch.allclose(batch1.topic_mixtures, batch2.topic_mixtures)


def test_color_sequence_conversion(simple_generator, simple_config):
    """Test conversion of tokens to color sequences."""
    dataset = SequenceDataset(simple_generator, simple_config)
    batch = next(iter(dataset))

    color_seqs = dataset.get_color_sequences(batch.tokens)
    assert color_seqs.shape == batch.tokens.shape

    # Check color indices are valid
    assert torch.all(color_seqs >= 0)
    assert torch.all(color_seqs < dataset.n_colors)


def test_batch_stats(simple_generator, simple_config):
    """Test batch statistics computation."""
    dataset = SequenceDataset(simple_generator, simple_config)
    batch = next(iter(dataset))
    stats = dataset.get_batch_stats(batch)

    # Check required statistics are present
    assert "mean_log_prob" in stats
    assert "topic_weights" in stats
    assert "color_counts" in stats

    # Check shapes and values
    assert len(stats["topic_weights"]) == dataset.n_topics
    assert len(stats["color_counts"]) == dataset.n_colors
    assert sum(stats["color_counts"]) == simple_config.batch_size * simple_config.seq_length


def test_color_constrained_generation(simple_generator, simple_config):
    """Test generation with specific start color."""
    dataset = SequenceDataset(simple_generator, simple_config)
    start_color = 1

    batch = dataset.generate_batch(start_color=start_color)
    color_seqs = dataset.get_color_sequences(batch.tokens)

    # Check first token of each sequence is correct color
    assert torch.all(color_seqs[:, 0] == start_color)


def test_topic_constrained_generation(simple_generator, simple_config):
    """Test generation with specific topic mixtures."""
    dataset = SequenceDataset(simple_generator, simple_config)
    
    # Create specific topic mixture
    mixtures = torch.tensor([
        [0.8, 0.2],  # Strong bias to first topic
        [0.2, 0.8],  # Strong bias to second topic
        [0.5, 0.5],  # Equal mixture
        [1.0, 0.0],  # Pure first topic
    ])

    batch = dataset.generate_batch(topic_mixtures=mixtures)
    assert torch.allclose(batch.topic_mixtures, mixtures)


def test_device_handling(simple_generator, simple_config):
    """Test device placement and consistency."""
    dataset = SequenceDataset(simple_generator, simple_config)
    batch = next(iter(dataset))

    # Check all tensors are on same device
    assert batch.tokens.device == dataset.device
    assert batch.topic_mixtures.device == dataset.device
    assert batch.log_probs.device == dataset.device

    # Test color sequence conversion maintains device
    color_seqs = dataset.get_color_sequences(batch.tokens)
    assert color_seqs.device == dataset.device
