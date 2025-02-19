# faux_lingo/tests/test_generator.py
"""Tests for sequence generation."""

import pytest
import torch

from faux_lingo.core.generator import SequenceGenerator
from faux_lingo.core.vocabulary import Vocabulary


@pytest.fixture
def simple_vocab():
    """Create simple vocabulary for testing."""
    return Vocabulary.create_simple(
        base_vocab_size=9,
        pad=True
    )


@pytest.fixture
def hierarchical_vocab():
    """Create hierarchical vocabulary for testing."""
    return Vocabulary.create_hierarchical(
        base_vocab_size=9,
        level_configs=[
            (18, 2),  # Level 1: 18 tokens, chunks of 2
            (36, 2),  # Level 2: 36 tokens, chunks of 2
        ],
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
def hierarchical_generator(hierarchical_vocab):
    """Create a generator with hierarchical vocabulary."""
    return SequenceGenerator.create_uniform(
        vocabulary=hierarchical_vocab,
        n_topics=2,
        color_fractions=[1, 1, 1],
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
        return_latent=True,
    )

    # Check base sequence
    assert torch.all(sequences.latent_tokens >= 0)
    assert torch.all(sequences.latent_tokens < simple_generator.vocabulary.base_vocab_size)

    # Check output sequence (may include special tokens)
    assert torch.all(sequences.tokens >= 0)
    assert torch.all(sequences.tokens < simple_generator.vocabulary.concrete_vocab_size)


def test_hierarchical_generation(hierarchical_generator):
    """Test sequence generation with hierarchical vocabulary."""
    batch_size = 4
    seq_length = 16  # Must be divisible by total expansion ratio

    sequences = hierarchical_generator.generate(
        batch_size=batch_size,
        seq_length=seq_length,
        return_latent=True,
    )

    # Check sequence lengths
    assert sequences.tokens.shape == (batch_size, seq_length)
    
    # Latent sequences should be shorter by expansion ratio
    expansion_ratio = hierarchical_generator.vocabulary.hierarchy.expansion_ratio
    expected_latent_length = seq_length // expansion_ratio
    assert sequences.latent_tokens.shape == (batch_size, expected_latent_length)


def test_color_start(simple_generator):
    """Test generation with specific start color."""
    batch_size = 5
    color_idx = 1  # Middle color class

    sequences = simple_generator.generate_with_color(
        batch_size=batch_size,
        seq_length=10,
        start_color=color_idx,
        return_latent=True,
    )

    # Get expected token range for color
    start_idx, end_idx = simple_generator.transition_model.color_space.get_color_range(
        color_idx
    )

    # Check first tokens are in correct range
    first_tokens = sequences.latent_tokens[:, 0]
    assert torch.all(first_tokens >= start_idx)
    assert torch.all(first_tokens < end_idx)


def test_temperature_effect(simple_generator):
    """Test that temperature affects transition entropy."""
    batch_size = 100
    seq_length = 20
    n_trials = 3

    entropy_diffs = []  # Store hot - cold entropy differences

    for seed in range(n_trials):
        torch.manual_seed(seed)

        # Generate with different temperatures
        cold_seqs = simple_generator.generate(
            batch_size=batch_size,
            seq_length=seq_length,
            temperature=0.1,
            return_latent=True,
        )
        hot_seqs = simple_generator.generate(
            batch_size=batch_size,
            seq_length=seq_length,
            temperature=10.0,
            return_latent=True,
        )

        # Compare transition statistics using latent sequences
        def get_transition_counts(tokens: torch.Tensor) -> torch.Tensor:
            """Get counts of token-to-token transitions."""
            counts = torch.zeros(
                (simple_generator.vocabulary.base_vocab_size,
                 simple_generator.vocabulary.base_vocab_size),
                device=tokens.device,
            )
            for i in range(tokens.shape[0]):  # For each sequence
                for t in range(tokens.shape[1] - 1):  # For each transition
                    curr, next = tokens[i, t], tokens[i, t + 1]
                    counts[curr, next] += 1
            return counts

        # Get transition counts and convert to probabilities
        cold_counts = get_transition_counts(cold_seqs.latent_tokens)
        hot_counts = get_transition_counts(hot_seqs.latent_tokens)

        cold_probs = cold_counts / (cold_counts.sum(-1, keepdim=True) + 1e-10)
        hot_probs = hot_counts / (hot_counts.sum(-1, keepdim=True) + 1e-10)

        # Calculate entropies
        def get_entropy(probs: torch.Tensor) -> float:
            """Calculate average entropy of transition distributions."""
            return -(probs * torch.log(probs + 1e-10)).sum(-1).mean().item()

        cold_entropy = get_entropy(cold_probs)
        hot_entropy = get_entropy(hot_probs)

        entropy_diffs.append(hot_entropy - cold_entropy)

    # Check if the effect is consistent
    assert all(diff > 0 for diff in entropy_diffs), "Higher temperature should increase entropy"


def test_padding(simple_generator):
    """Test sequence padding behavior."""
    vocab = Vocabulary.create_simple(base_vocab_size=9, pad=True)
    generator = SequenceGenerator.create_uniform(
        vocabulary=vocab,
        n_topics=2,
        color_fractions=[1, 1, 1],
    )

    # Generate sequence shorter than requested
    sequences = generator.generate(batch_size=2, seq_length=10)
    
    # Check padding token is used correctly
    padding_token = generator.vocabulary.special_tokens.pad_token
    assert padding_token is not None
    padded_positions = sequences.tokens == padding_token
    assert torch.any(padded_positions), "Padding token should be used"


def test_no_padding_token_error(simple_vocab):
    """Test error when padding needed but no padding token defined."""
    # Create vocabulary without padding token
    vocab = Vocabulary.create_simple(base_vocab_size=9)  # No padding token
    generator = SequenceGenerator.create_uniform(
        vocabulary=vocab,
        n_topics=2,
        color_fractions=[1, 1, 1],
    )

    with pytest.raises(ValueError, match="Padding token not defined"):
        generator.generate(batch_size=2, seq_length=10)


def test_reproducibility(simple_generator):
    """Test that sequences are reproducible with same seed."""
    torch.manual_seed(42)
    seq1 = simple_generator.generate(
        batch_size=2,
        seq_length=10,
        return_latent=True,
    )

    torch.manual_seed(42)
    seq2 = simple_generator.generate(
        batch_size=2,
        seq_length=10,
        return_latent=True,
    )

    assert torch.all(seq1.tokens == seq2.tokens)
    assert torch.all(seq1.latent_tokens == seq2.latent_tokens)
    assert torch.allclose(seq1.log_probs, seq2.log_probs)


def test_sequence_length_rounding(hierarchical_generator):
    """Test handling of sequence lengths that don't divide evenly."""
    # Request a sequence length that isn't divisible by expansion ratio
    expansion_ratio = hierarchical_generator.vocabulary.hierarchy.expansion_ratio
    seq_length = expansion_ratio * 5 + 1  # Odd length

    sequences = hierarchical_generator.generate(
        batch_size=2,
        seq_length=seq_length,
    )

    # Final sequence should have exactly the requested length
    assert sequences.tokens.shape[1] == seq_length


def test_bos_eos_tokens():
    """Test generation with beginning/end tokens."""
    vocab = Vocabulary.create_simple(
        base_vocab_size=9,
        bos=True,
        eos=True,
    )
    generator = SequenceGenerator.create_uniform(
        vocabulary=vocab,
        n_topics=2,
        color_fractions=[1, 1, 1],
    )

    sequences = generator.generate(batch_size=2, seq_length=10)
    
    # First token should be BOS
    assert torch.all(sequences.tokens[:, 0] == vocab.special_tokens.bos_token)
    
    # Last token should be EOS
    assert torch.all(sequences.tokens[:, -1] == vocab.special_tokens.eos_token)


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
