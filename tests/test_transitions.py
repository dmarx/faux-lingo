# faux_lingo/tests/test_transitions.py
"""Tests for transition matrix generation."""

import pytest
import torch

from faux_lingo.core.colors import ColorSpace
from faux_lingo.core.topics import TopicVectorSpace
from faux_lingo.core.transitions import TransitionMatrix


@pytest.fixture
def simple_matrix():
    """Create a simple transition matrix for testing."""
    return TransitionMatrix.create_uniform(
        vocab_size=9,
        n_topics=2,
        color_fractions=[1, 1, 1],  # Three equal-sized color classes
    )


def test_initialization():
    """Test constructor validation."""
    # Create spaces with mismatched vocab sizes
    topic_space = TopicVectorSpace(n_topics=2, vocab_size=10)
    color_space = ColorSpace(color_fractions=[1, 1], vocab_size=12)

    # Should raise error
    with pytest.raises(ValueError, match="Vocab size mismatch"):
        TransitionMatrix(topic_space, color_space)


def test_uniform_creation():
    """Test creation of uniform transition matrix."""
    matrix = TransitionMatrix.create_uniform(
        vocab_size=6,
        n_topics=2,
        color_fractions=[1, 1],  # Two equal color classes
    )

    assert matrix.vocab_size == 6
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


def test_invalid_mixture_shape(simple_matrix):
    """Test validation of topic mixture shape."""
    # Wrong number of topics
    bad_mixture = torch.ones(1, 3) / 3  # 3 topics when space has 2

    with pytest.raises(ValueError):
        simple_matrix.generate(bad_mixture)


def test_reproducibility():
    """Test that results are reproducible with same random seed."""
    torch.manual_seed(42)
    matrix1 = TransitionMatrix.create_uniform(
        vocab_size=6, n_topics=2, color_fractions=[1, 1]
    )
    result1 = matrix1.generate(torch.ones(1, 2) / 2)

    torch.manual_seed(42)
    matrix2 = TransitionMatrix.create_uniform(
        vocab_size=6, n_topics=2, color_fractions=[1, 1]
    )
    result2 = matrix2.generate(torch.ones(1, 2) / 2)

    assert torch.allclose(result1, result2)
