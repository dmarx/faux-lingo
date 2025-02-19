# faux_lingo/tests/test_transitions.py
"""Tests for transition matrix generation."""

import pytest
import torch

from faux_lingo.core.colors import ColorSpace
from faux_lingo.core.topics import TopicVectorSpace
from faux_lingo.core.transitions import TransitionMatrix
from faux_lingo.core.vocabulary import Vocabulary


@pytest.fixture
def simple_vocab():
    """Create simple vocabulary for testing."""
    return Vocabulary.create_simple(base_vocab_size=9)


@pytest.fixture
def simple_matrix(simple_vocab):
    """Create a simple transition matrix for testing."""
    return TransitionMatrix.create_uniform(
        vocabulary=simple_vocab,
        n_topics=2,
        color_fractions=[1, 1, 1],  # Three equal-sized color classes
    )


def test_initialization(simple_vocab):
    """Test constructor validation."""
    # Create spaces with mismatched vocab sizes
    topic_space = TopicVectorSpace(n_topics=2, vocab_size=10)
    color_space = ColorSpace(color_fractions=[1, 1], vocab_size=12)

    # Should raise error
    with pytest.raises(ValueError, match="Topic space vocab size"):
        TransitionMatrix(simple_vocab, topic_space, color_space)


def test_uniform_creation(simple_vocab):
    """Test creation of uniform transition matrix."""
    matrix = TransitionMatrix.create_uniform(
        vocabulary=simple_vocab,
        n_topics=2,
        color_fractions=[1, 1],  # Two equal color classes
    )

    assert matrix.vocabulary.base_vocab_size == 9
    assert matrix.topic_space.n_topics == 2
    assert matrix.color_space.n_colors == 2


def test_probability_properties(simple_matrix):
    """Test that generated matrices have valid probability properties."""
    # Generate matrix with uniform mixture
    mixture = torch.ones(1, 2) / 2  # Equal mixture of two topics
    transitions = simple_matrix.generate(mixture)

    # Check shape
    assert transitions.shape == (1, 9, 9)

    # Check row sums
    row_sums = transitions.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))

    # Check non-negativity
    assert torch.all(transitions >= 0)


def test_color_constraints(simple_matrix):
    """Test that color transition constraints are respected."""
    # Set up transition weights that forbid some transitions
    weights = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Color 0 can only transition to itself
            [0.0, 1.0, 0.0],  # Color 1 can only transition to itself
            [0.0, 0.0, 1.0],  # Color 2 can only transition to itself
        ]
    )
    simple_matrix.color_space.transition_weights = weights

    # Generate transitions
    mixture = torch.ones(1, 2) / 2
    transitions = simple_matrix.generate(mixture)

    # Check block structure
    for i in range(3):  # For each color
        for j in range(3):  # For each target color
            if i != j:  # Off-diagonal blocks should be zero
                start_i = i * 3
                end_i = (i + 1) * 3
                start_j = j * 3
                end_j = (j + 1) * 3
                block = transitions[0, start_i:end_i, start_j:end_j]
                assert torch.all(block == 0)


def test_temperature_effect(simple_matrix):
    """Test that temperature affects distribution entropy."""
    mixture = torch.ones(1, 2) / 2

    # Generate with different temperatures
    cold = simple_matrix.generate(mixture, temperature=0.1)
    hot = simple_matrix.generate(mixture, temperature=10.0)

    # Higher temperature should give more uniform distributions
    cold_entropy = -(cold * torch.log(cold + 1e-10)).sum(dim=-1).mean()
    hot_entropy = -(hot * torch.log(hot + 1e-10)).sum(dim=-1).mean()

    assert hot_entropy > cold_entropy


def test_batch_generation(simple_matrix):
    """Test generation of multiple matrices simultaneously."""
    # Create batch of different mixtures
    mixtures = torch.tensor(
        [
            [0.8, 0.2],  # First sequence favors topic 0
            [0.2, 0.8],  # Second sequence favors topic 1
        ]
    )

    transitions = simple_matrix.generate(mixtures)

    # Check batch dimension
    assert transitions.shape == (2, 9, 9)

    # Check each matrix independently sums to 1
    row_sums = transitions.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_min_probability(simple_matrix):
    """Test that minimum probability is respected for valid transitions."""
    mixture = torch.ones(1, 2) / 2
    min_prob = 1e-4

    transitions = simple_matrix.generate(mixture, min_prob=min_prob)

    # Get color mask to identify valid transitions
    color_mask = simple_matrix.color_space.get_transition_mask()
    valid_transitions = transitions[0][color_mask > 0]

    # Check minimum probability is respected
    assert torch.all(valid_transitions >= min_prob)


def test_device_consistency(simple_matrix):
    """Test that all tensors stay on the same device."""
    mixture = torch.ones(1, 2) / 2
    transitions = simple_matrix.generate(mixture)

    # All tensors should be on the same device
    assert transitions.device == simple_matrix.topic_space.vectors.device
    assert transitions.device == simple_matrix.color_space.mapping.boundaries.device


def test_hierarchical_vocab():
    """Test transitions with hierarchical vocabulary."""
    vocab = Vocabulary.create_hierarchical(
        base_vocab_size=6,
        level_configs=[
            (12, 2),  # Level 1: 12 tokens, chunks of 2
            (24, 2),  # Level 2: 24 tokens, chunks of 2
        ]
    )

    matrix = TransitionMatrix.create_uniform(
        vocabulary=vocab,
        n_topics=2,
        color_fractions=[1, 1],  # Two color classes
    )

    # Transitions should operate on base vocabulary
    mixture = torch.ones(1, 2) / 2
    transitions = matrix.generate(mixture)
    assert transitions.shape == (1, 6, 6)


def test_special_tokens():
    """Test vocabulary with special tokens."""
    vocab = Vocabulary.create_simple(
        base_vocab_size=6,
        pad=True,
        bos=True
    )

    matrix = TransitionMatrix.create_uniform(
        vocabulary=vocab,
        n_topics=2,
        color_fractions=[1, 1],
    )

    # Transitions should still use base vocabulary size
    mixture = torch.ones(1, 2) / 2
    transitions = matrix.generate(mixture)
    assert transitions.shape == (1, 6, 6)
