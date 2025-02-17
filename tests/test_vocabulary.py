# tests/test_vocabulary.py

"""Tests for the vocabulary generation system."""

import pytest
from faux_lingo.core.vocabulary import VocabBuilder, VocabConfig

def test_vocab_config_validation():
    """Test that VocabConfig validates parameters correctly."""
    # Valid configuration
    config = VocabConfig(
        token_vocab_size=10,
        rune_vocab_size=20,
        char_vocab_size=15,
        word_vocab_size=100,
        tokens_per_rune=1,
        runes_per_char=2,
        chars_per_word=3
    )
    config.validate()  # Should not raise

    # Invalid: negative vocabulary size
    with pytest.raises(ValueError, match="token_vocab_size must be positive"):
        VocabConfig(
            token_vocab_size=-1,
            rune_vocab_size=20,
            char_vocab_size=15,
            word_vocab_size=100,
            tokens_per_rune=1,
            runes_per_char=2,
            chars_per_word=3
        ).validate()

    # Invalid: rune vocabulary too large for possible combinations
    with pytest.raises(ValueError, match="rune_vocab_size .* exceeds maximum"):
        VocabConfig(
            token_vocab_size=2,  # Only 2 tokens
            rune_vocab_size=5,   # But asking for 5 unique runes with 1 token each
            char_vocab_size=15,
            word_vocab_size=100,
            tokens_per_rune=1,
            runes_per_char=2,
            chars_per_word=3
        ).validate()

def test_vocab_builder_initialization():
    """Test VocabBuilder initialization and state."""
    config = VocabConfig(
        token_vocab_size=10,
        rune_vocab_size=20,
        char_vocab_size=15,
        word_vocab_size=100,
        tokens_per_rune=1,
        runes_per_char=2,
        chars_per_word=3
    )
    builder = VocabBuilder(config, seed=42)

    assert len(builder.token_vocab) == 0
    assert len(builder.rune_vocab) == 0
    assert len(builder.char_vocab) == 0
    assert len(builder.word_vocab) == 0
    assert len(builder._used_runes) == 0
    assert len(builder._used_chars) == 0
    assert len(builder._used_words) == 0

def test_token_vocab_generation():
    """Test generation of base token vocabulary."""
    config = VocabConfig(
        token_vocab_size=5,
        rune_vocab_size=20,
        char_vocab_size=15,
        word_vocab_size=100,
        tokens_per_rune=1,
        runes_per_char=2,
        chars_per_word=3
    )
    builder = VocabBuilder(config, seed=42)
    builder.build_token_vocab()

    assert len(builder.token_vocab) == 5
    assert builder.token_vocab == [0, 1, 2, 3, 4]

def test_rune_vocab_generation():
    """Test generation of rune vocabulary."""
    config = VocabConfig(
        token_vocab_size=3,
        rune_vocab_size=4,
        char_vocab_size=15,
        word_vocab_size=100,
        tokens_per_rune=2,  # Each rune is 2 tokens
        runes_per_char=2,
        chars_per_word=3
    )
    builder = VocabBuilder(config, seed=42)
    builder.build_token_vocab()
    builder.build_rune_vocab()

    assert len(builder.rune_vocab) == 4
    # Check all runes are unique
    assert len(set(builder.rune_vocab)) == 4
    # Check each rune has correct length
    assert all(len(rune) == 2 for rune in builder.rune_vocab)
    # Check all tokens in runes are valid
    assert all(all(0 <= t < 3 for t in rune) for rune in builder.rune_vocab)

def test_full_vocabulary_build():
    """Test complete vocabulary building process."""
    config = VocabConfig(
        token_vocab_size=3,
        rune_vocab_size=4,
        char_vocab_size=3,
        word_vocab_size=2,
        tokens_per_rune=1,
        runes_per_char=2,
        chars_per_word=2
    )
    builder = VocabBuilder(config, seed=42)
    vocab = builder.build()

    assert "token_vocab" in vocab
    assert "rune_vocab" in vocab
    assert "char_vocab" in vocab
    assert "word_vocab" in vocab

    # Check sizes
    assert len(vocab["token_vocab"]) == 3
    assert len(vocab["rune_vocab"]) == 4
    assert len(vocab["char_vocab"]) == 3
    assert len(vocab["word_vocab"]) == 2

    # Check word structure
    for word in vocab["word_vocab"]:
        # Each word should be a flat list of tokens
        # Size = chars_per_word * runes_per_char * tokens_per_rune
        assert len(word) == 2 * 2 * 1
        # All tokens should be valid
        assert all(0 <= t < 3 for t in word)

def test_deterministic_generation():
    """Test that setting a seed produces deterministic results."""
    config = VocabConfig(
        token_vocab_size=3,
        rune_vocab_size=4,
        char_vocab_size=3,
        word_vocab_size=2,
        tokens_per_rune=1,
        runes_per_char=2,
        chars_per_word=2
    )
    
    builder1 = VocabBuilder(config, seed=42)
    vocab1 = builder1.build()
    
    builder2 = VocabBuilder(config, seed=42)
    vocab2 = builder2.build()

    assert vocab1["word_vocab"] == vocab2["word_vocab"]
