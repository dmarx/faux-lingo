# faux_lingo/tests/test_generator.py
"""Tests for sequence generation."""

import pytest
import torch

from faux_lingo.core.generator import SequenceGenerator


@pytest.fixture
def simple_generator():
    """Create a simple generator for testing."""
    return SequenceGenerator.create_uniform(
        vocab_size=9,
        n_topics=2,
        color_fractions=[1, 1, 1],  # Three equal color classes
    )


def test_sequence_shapes(simple_generator):
    """Test output shapes from generation."""
    batch_size = 4
    seq_length = 10

    sequences = simple_generator.generate(
        batch_size=batch_size,
        seq_length=seq_length,
    )

    assert sequences.tokens.shape == (batch_size, seq_length)
    assert sequences.topic_mixtures.shape == (batch_size, 2)  # 2 topics
    assert sequences.log_probs.shape == (batch_size,)


def test_token_ranges(simple_generator):
    """Test that generated tokens are within vocabulary."""
    sequences = simple_generator.generate(
        batch_size=10,
        seq_length=20,
    )

    assert torch.all(sequences.tokens >= 0)
    assert torch.all(sequences.tokens < simple_generator.vocab_size)


def test_color_start(simple_generator):
    """Test generation with specific start color."""
    batch_size = 5
    color_idx = 1  # Middle color class

    sequences = simple_generator.generate_with_color(
        batch_size=batch_size,
        seq_length=10,
        start_color=color_idx,
    )

    # Get expected token range for color
    start_idx, end_idx = simple_generator.transition_model.color_space.get_color_range(
        color_idx
    )

    # Check first tokens are in correct range
    first_tokens = sequences.tokens[:, 0]
    assert torch.all(first_tokens >= start_idx)
    assert torch.all(first_tokens < end_idx)


def test_temperature_effect(simple_generator):
    """Test that temperature affects sequence diversity."""
    batch_size = 100
    seq_length = 20

    # Generate with different temperatures
    cold_seqs = simple_generator.generate(
        batch_size=batch_size,
        seq_length=seq_length,
        temperature=0.1,
    )
    hot_seqs = simple_generator.generate(
        batch_size=batch_size,
        seq_length=seq_length,
        temperature=10.0,
    )

    # Compare token diversity (higher temperature should give more diverse tokens)
    cold_unique = torch.unique(cold_seqs.tokens).numel()
    hot_unique = torch.unique(hot_seqs.tokens).numel()

    assert hot_unique > cold_unique


def test_topic_mixture_validation(simple_generator):
    """Test validation of topic mixture inputs."""
    # Wrong batch size
    bad_mixtures = torch.ones(3, 2) / 2  # 3 sequences when asking for 2

    with pytest.raises(ValueError):
        simple_generator.generate(
            batch_size=2,
            seq_length=10,
            topic_mixtures=bad_mixtures,
        )


def test_start_token_validation(simple_generator):
    """Test validation of start token inputs."""
    # Wrong shape
    bad_tokens = torch.zeros(3)  # 3 tokens when asking for 2 sequences

    with pytest.raises(ValueError):
        simple_generator.generate(
            batch_size=2,
            seq_length=10,
            start_tokens=bad_tokens,
        )


def test_color_validation(simple_generator):
    """Test validation of color inputs."""
    with pytest.raises(ValueError):
        simple_generator.generate_with_color(
            batch_size=2,
            seq_length=10,
            start_color=99,  # Invalid color index
        )


def test_log_probability_consistency(simple_generator):
    """Test that log probabilities are consistent with transitions."""
    # Generate single sequence for simplicity
    batch_size = 1
    seq_length = 5
    temperature = 1.0

    # Generate with specific topic mixture
    mixture = torch.tensor([[0.7, 0.3]], device=simple_generator.device)
    sequences = simple_generator.generate(
        batch_size=batch_size,
        seq_length=seq_length,
        topic_mixtures=mixture,
        temperature=temperature,
    )

    # Get transition matrix
    transitions = simple_generator.transition_model.generate(
        mixture,
        temperature=temperature,
    )

    # Manually compute log probability
    manual_log_prob = 0.0
    for t in range(1, seq_length):
        prev_token = sequences.tokens[0, t - 1]
        curr_token = sequences.tokens[0, t]
        prob = transitions[0, prev_token, curr_token]
        manual_log_prob += torch.log(prob).item()

    assert torch.allclose(
        sequences.log_probs[0],
        torch.tensor(manual_log_prob, device=simple_generator.device),
        rtol=1e-5,
    )


def test_reproducibility(simple_generator):
    """Test that sequences are reproducible with same seed."""
    torch.manual_seed(42)
    seq1 = simple_generator.generate(
        batch_size=2,
        seq_length=10,
    )

    torch.manual_seed(42)
    seq2 = simple_generator.generate(
        batch_size=2,
        seq_length=10,
    )

    assert torch.all(seq1.tokens == seq2.tokens)
    assert torch.allclose(seq1.log_probs, seq2.log_probs)
