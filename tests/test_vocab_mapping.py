# faux_lingo/tests/test_vocab_mapping.py
"""Tests for vocabulary mapping and decoding."""

import pytest
import torch

from faux_lingo.core.vocab_mapping import VocabHierarchy, VocabLevel


@pytest.fixture
def simple_hierarchy():
    """Create a simple 3-level hierarchy for testing.
    
    Structure:
    - Level 0 (tokens): 0-4
    - Level 1 (chars): Each maps to two tokens
    - Level 2 (words): Each maps to two chars
    """
    levels = [
        VocabLevel(  # Base tokens
            vocab_size=5,
            chunk_size=1,
            sequences={i: (i,) for i in range(5)},  # Identity mapping
        ),
        VocabLevel(  # Characters
            vocab_size=3,
            chunk_size=2,
            sequences={
                0: (0, 1),  # char 0 -> tokens [0,1]
                1: (2, 3),  # char 1 -> tokens [2,3]
                2: (3, 4),  # char 2 -> tokens [3,4]
            },
        ),
        VocabLevel(  # Words
            vocab_size=2,
            chunk_size=2,
            sequences={
                0: (0, 1),  # word 0 -> chars [0,1]
                1: (1, 2),  # word 1 -> chars [1,2]
            },
        ),
    ]
    return VocabHierarchy(levels)


def test_vocab_level_validation():
    """Test validation of vocabulary level properties."""
    # Valid level
    level = VocabLevel(vocab_size=2, chunk_size=1, sequences={0: (0,), 1: (1,)})
    assert level.max_sequence_length == 1

    # Invalid vocab size
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        VocabLevel(vocab_size=0, chunk_size=1, sequences={})

    # Invalid chunk size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        VocabLevel(vocab_size=1, chunk_size=0, sequences={0: (0,)})

    # Mismatched sequence count
    with pytest.raises(ValueError, match="Number of sequences"):
        VocabLevel(vocab_size=2, chunk_size=1, sequences={0: (0,)})

    # Invalid sequence type
    with pytest.raises(ValueError, match="must be a tuple"):
        VocabLevel(vocab_size=1, chunk_size=1, sequences={0: [0]})

    # Invalid sequence elements
    with pytest.raises(ValueError, match="must be integers"):
        VocabLevel(vocab_size=1, chunk_size=1, sequences={0: (0.5,)})


def test_single_token_decoding(simple_hierarchy):
    """Test decoding of individual tokens."""
    # Decode from word to chars
    word = torch.tensor([[0]], device=simple_hierarchy.device)
    chars = simple_hierarchy.decode_sequence(word, start_level=2, target_level=1)
    assert torch.equal(chars, torch.tensor([[0, 1]], device=simple_hierarchy.device))

    # Decode from word to tokens
    tokens = simple_hierarchy.decode_sequence(word, start_level=2, target_level=0)
    assert torch.equal(
        tokens, torch.tensor([[0, 1, 2, 3]], device=simple_hierarchy.device)
    )


def test_sequence_decoding(simple_hierarchy):
    """Test decoding of token sequences."""
    # Decode sequence of words
    words = torch.tensor([[0, 1]], device=simple_hierarchy.device)
    chars = simple_hierarchy.decode_sequence(words, start_level=2, target_level=1)
    assert torch.equal(
        chars, torch.tensor([[0, 1, 1, 2]], device=simple_hierarchy.device)
    )

    tokens = simple_hierarchy.decode_sequence(words, start_level=2, target_level=0)
    assert torch.equal(
        tokens,
        torch.tensor([[0, 1, 2, 3, 2, 3, 3, 4]], device=simple_hierarchy.device),
    )


def test_batch_decoding(simple_hierarchy):
    """Test decoding of batched sequences."""
    words = torch.tensor(
        [
            [0, 1],  # First sequence
            [1, 0],  # Second sequence
        ],
        device=simple_hierarchy.device,
    )
    tokens = simple_hierarchy.decode_sequence(words, start_level=2, target_level=0)

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 2, 3, 3, 4],  # Decoded first sequence
            [2, 3, 3, 4, 0, 1, 2, 3],  # Decoded second sequence
        ],
        device=simple_hierarchy.device,
    )
    assert torch.equal(tokens, expected)


def test_invalid_level_decoding(simple_hierarchy):
    """Test validation of decoding levels."""
    words = torch.tensor([[0]], device=simple_hierarchy.device)

    # Invalid start level
    with pytest.raises(ValueError, match="Invalid start_level"):
        simple_hierarchy.decode_sequence(words, start_level=3, target_level=0)

    # Invalid target level
    with pytest.raises(ValueError, match="Invalid target_level"):
        simple_hierarchy.decode_sequence(words, start_level=2, target_level=-1)

    # Can't decode upward
    with pytest.raises(ValueError, match="Can only decode to same or lower levels"):
        simple_hierarchy.decode_sequence(words, start_level=0, target_level=1)


def test_from_sequences():
    """Test creation of hierarchy from sequence mappings."""
    sequences = [
        {0: (0,), 1: (1,)},  # Base tokens
        {0: (0, 1)},  # One character
    ]
    chunk_sizes = [1, 2]

    hierarchy = VocabHierarchy.from_sequences(sequences, chunk_sizes)
    assert len(hierarchy) == 2
    assert hierarchy[0].vocab_size == 2
    assert hierarchy[1].vocab_size == 1

    # Mismatched lengths
    with pytest.raises(ValueError, match="chunk size for each level"):
        VocabHierarchy.from_sequences(sequences, chunk_sizes[:-1])


def test_device_handling():
    """Test device placement and movement of tensors."""
    level = VocabLevel(vocab_size=2, chunk_size=1, sequences={0: (0,), 1: (1,)})
    hierarchy = VocabHierarchy([level], device="cpu")

    # Input on same device
    tokens = torch.tensor([[0]], device="cpu")
    result = hierarchy.decode_sequence(tokens, start_level=0, target_level=0)
    assert result.device.type == "cpu"

    # Input on different device gets moved
    if torch.cuda.is_available():
        hierarchy = VocabHierarchy([level], device="cuda")
        result = hierarchy.decode_sequence(tokens, start_level=0, target_level=0)
        assert result.device.type == "cuda"
