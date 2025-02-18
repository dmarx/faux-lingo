# faux_lingo/tests/test_vocab_builder.py
"""Tests for vocabulary building functionality."""

import pytest

from faux_lingo.core.vocab_builder import BuilderConfig, VocabBuilder, create_word_hierarchy


@pytest.fixture
def simple_config():
    """Create simple builder configuration."""
    return BuilderConfig(
        token_vocab_size=4,
        sequence_lengths=[2, 2],  # Two levels, sequences of length 2
        vocab_sizes=[4, 3],  # 4 chars from tokens, 3 words from chars
    )


def test_config_validation():
    """Test builder configuration validation."""
    # Valid configuration
    config = BuilderConfig(
        token_vocab_size=4,
        sequence_lengths=[2, 2],
        vocab_sizes=[4, 3],
    )
    assert config.token_vocab_size == 4

    # Invalid token vocab size
    with pytest.raises(ValueError, match="token_vocab_size must be positive"):
        BuilderConfig(
            token_vocab_size=0,
            sequence_lengths=[2],
            vocab_sizes=[4],
        )

    # Mismatched lengths
    with pytest.raises(ValueError, match="sequence length and vocabulary size"):
        BuilderConfig(
            token_vocab_size=4,
            sequence_lengths=[2],
            vocab_sizes=[4, 3],
        )

    # Invalid sequence length
    with pytest.raises(ValueError, match="sequence lengths must be positive"):
        BuilderConfig(
            token_vocab_size=4,
            sequence_lengths=[0, 2],
            vocab_sizes=[4, 3],
        )

    # Vocabulary too large for combinations
    with pytest.raises(ValueError, match="exceeds maximum possible combinations"):
        BuilderConfig(
            token_vocab_size=2,  # Only 4 possible pairs
            sequence_lengths=[2],
            vocab_sizes=[5],  # Want 5 unique pairs
        )


def test_builder_reproducibility(simple_config):
    """Test that building is reproducible with same seed."""
    config1 = BuilderConfig(
        **{**simple_config.__dict__, "seed": 42}
    )
    hierarchy1 = VocabBuilder(config1).build()

    config2 = BuilderConfig(
        **{**simple_config.__dict__, "seed": 42}
    )
    hierarchy2 = VocabBuilder(config2).build()

    # Check sequences match at each level
    for level1, level2 in zip(hierarchy1, hierarchy2):
        assert level1.sequences == level2.sequences


def test_sequence_uniqueness(simple_config):
    """Test that generated sequences are unique within levels."""
    hierarchy = VocabBuilder(simple_config).build()

    for level in hierarchy:
        sequences = set(level.sequences.values())
        assert len(sequences) == level.vocab_size


def test_sequence_validity(simple_config):
    """Test that sequences use valid tokens from previous level."""
    hierarchy = VocabBuilder(simple_config).build()

    # Check first level uses valid base tokens
    for sequence in hierarchy[0].sequences.values():
        assert all(0 <= token < simple_config.token_vocab_size for token in sequence)

    # Check second level uses valid tokens from first level
    for sequence in hierarchy[1].sequences.values():
        assert all(0 <= token < hierarchy[0].vocab_size for token in sequence)


def test_create_word_hierarchy():
    """Test convenience function for word hierarchy creation."""
    hierarchy = create_word_hierarchy(
        token_vocab_size=5,
        n_chars=10,
        n_words=20,
        chars_per_word=3,
        seed=42,
    )

    assert len(hierarchy) == 2  # Two levels: chars and words
    assert hierarchy[0].vocab_size == 10  # Number of characters
    assert hierarchy[1].vocab_size == 20  # Number of words
    assert all(len(seq) == 3 for seq in hierarchy[1].sequences.values())


def test_default_config():
    """Test default configuration creation."""
    config = VocabBuilder.create_default_config()
    
    # Verify defaults are valid
    hierarchy = VocabBuilder(config).build()
    assert len(hierarchy) == 3  # Three levels
    assert hierarchy[0].vocab_size == 20  # First level vocab size
    assert hierarchy[1].vocab_size == 15  # Second level vocab size
    assert hierarchy[2].vocab_size == 10  # Third level vocab size
