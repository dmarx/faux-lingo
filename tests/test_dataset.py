# faux_lingo/tests/test_dataset.py
"""Tests for dataset generation functionality."""

import pytest
import torch

from faux_lingo.core.generator import SequenceGenerator
from faux_lingo.data.dataset import DatasetConfig, SequenceDataset
from faux_lingo.core.vocabulary import Vocabulary


@pytest.fixture
def simple_vocab():
    """Create simple vocabulary for testing."""
    return Vocabulary.create_simple(
        base_vocab_size=9,
        pad=True
    )


@pytest.fixture
def simple_generator(simple_vocab):
    """Create a simple generator for testing."""
    return SequenceGenerator.create_uniform(
        vocabulary=simple_vocab,
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

    color_seqs = dataset.get_color_sequences(batch.latent_tokens)
    assert color_seqs.shape == batch.latent_tokens.shape

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
    color_seqs = dataset.get_color_sequences(batch.latent_tokens)

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
    assert batch.tokens.device.type == dataset.device
    assert batch.topic_mixtures.device.type == dataset.device
    assert batch.log_probs.device.type == dataset.device

    # Test color sequence conversion maintains device
    color_seqs = dataset.get_color_sequences(batch.latent_tokens)
    assert color_seqs.device.type == dataset.device


def test_hierarchical_dataset():
    """Test dataset with hierarchical vocabulary."""
    vocab = Vocabulary.create_hierarchical(
        base_vocab_size=6,
        level_configs=[
            (12, 2),  # Level 1: 12 tokens, chunks of 2
            (24, 2),  # Level 2: 24 tokens, chunks of 2
        ],
        pad=True
    )

    generator = SequenceGenerator.create_uniform(
        vocabulary=vocab,
        n_topics=2,
        color_fractions=[1, 1],
    )

    config = DatasetConfig(
        batch_size=4,
        seq_length=16,  # Multiple of total expansion ratio
        n_batches=2,
    )

    dataset = SequenceDataset(generator, config)
    batch = next(iter(dataset))

    # Check sequence lengths
    assert batch.tokens.shape == (config.batch_size, config.seq_length)
    expected_latent_length = config.seq_length // vocab.hierarchy.expansion_ratio
    assert batch.latent_tokens.shape == (config.batch_size, expected_latent_length)


def test_special_token_handling():
    """Test dataset handling of special tokens."""
    # Create vocabulary with special tokens
    vocab = Vocabulary.create_simple(
        base_vocab_size=9,
        pad=True,
        bos=True,
        eos=True,
    )

    generator = SequenceGenerator.create_uniform(
        vocabulary=vocab,
        n_topics=2,
        color_fractions=[1, 1, 1],
    )

    # Create a config that ensures we need padding (longer than base tokens)
    expansion_ratio = generator.vocabulary.hierarchy.expansion_ratio if generator.vocabulary.has_hierarchy else 1
    base_length = 5  # Short enough to need padding
    config = DatasetConfig(
        batch_size=4,
        seq_length=base_length * expansion_ratio,
        n_batches=2,
    )

    dataset = SequenceDataset(generator, config)
    
    # Generate with both BOS and EOS to ensure special tokens appear
    sequences = generator.generate(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        return_latent=True,
    )

    # Verify token ranges
    assert sequences.latent_tokens is not None
    max_latent = torch.max(sequences.latent_tokens).item()
    assert max_latent < vocab.base_vocab_size, "Latent tokens should be in base vocabulary range"

    # Color sequences should only reference base vocabulary
    color_seqs = dataset.get_color_sequences(sequences.latent_tokens)
    max_color = torch.max(color_seqs).item()
    assert max_color < generator.transition_model.color_space.n_colors, "Color indices should be valid"

    # Verify special tokens are accessible but don't have to appear
    assert vocab.special_tokens is not None
    if vocab.special_tokens.pad_token is not None:
        assert vocab.special_tokens.pad_token >= vocab.base_vocab_size
    if vocab.special_tokens.bos_token is not None:
        assert vocab.special_tokens.bos_token >= vocab.base_vocab_size
    if vocab.special_tokens.eos_token is not None:
        assert vocab.special_tokens.eos_token >= vocab.base_vocab_size
