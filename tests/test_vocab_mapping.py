# faux_lingo/tests/test_vocab_mapping.py
"""Tests for vocabulary mapping and decoding."""

import pytest
import torch

from faux_lingo.core.vocab_mapping import VocabHierarchy, VocabLevel

@pytest.fixture
def simple_hierarchy():
    """Create a simple 3-level hierarchy for testing.
    
    Structure:
    - Level 0 (most abstract) tokens map to 2 level 1 tokens
    - Level 1 tokens map to 2 level 2 (most concrete) tokens
    
    Example mappings:
    Level 0 -> Level 1:
    - 0 -> (0, 1)
    - 1 -> (1, 2)
    
    Level 1 -> Level 2:
    - 0 -> (0, 1)
    - 1 -> (2, 3)
    - 2 -> (3, 4)
    """
    levels = [
        VocabLevel(  # Level 0 -> Level 1 mapping
            vocab_size=2,
            chunk_size=2,
            sequences={
                0: (0, 1),  # word 0 -> chars [0,1]
                1: (1, 2),  # word 1 -> chars [1,2]
            },
        ),
        VocabLevel(  # Level 1 -> Level 2 mapping
            vocab_size=3,
            chunk_size=2,
            sequences={
                0: (0, 1),  # char 0 -> tokens [0,1]
                1: (2, 3),  # char 1 -> tokens [2,3]
                2: (3, 4),  # char 2 -> tokens [3,4]
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

def test_hierarchy_respected():
    import torch
    import numpy as np
    from faux_lingo.core.vocab_builder import BuilderConfig, VocabBuilder
    
    config = BuilderConfig(
        token_vocab_size=10,
        sequence_lengths=[2, 3],  # Length at each level
        vocab_sizes=[20, 30]      # Size of each level
    )
    
    builder = VocabBuilder(config)
    hierarchy = builder.build()
    
    tokens = torch.tensor([[0,1,2]])
    decoded = hierarchy.decode_sequence(tokens,start_level=2, target_level=0)
    
    target_length = np.prod( [level.chunk_size for level in hierarchy.levels]) * tokens.shape[1]
    assert target_length == decoded.shape[1]
    
def test_single_token_decoding(simple_hierarchy):
    """Test decoding of individual tokens."""
    # Level 0 token 0 maps to level 1 tokens [0,1]
    # which map to level 2 tokens [0,1,2,3]
    word = torch.tensor([[0]], device=simple_hierarchy.device)
    
    # Default decoding (full expansion)
    tokens = simple_hierarchy.decode_sequence(word)
    assert torch.equal(tokens, torch.tensor([[0, 1, 2, 3]], device=simple_hierarchy.device))
    
    # Decode from level 0 to level 1
    chars = simple_hierarchy.decode_sequence(word, target_level=1)
    assert torch.equal(chars, torch.tensor([[0, 1]], device=simple_hierarchy.device))


def test_sequence_decoding(simple_hierarchy):
    """Test decoding of token sequences."""
    # Level 0 sequence [0,1] maps to:
    # Level 1: [0,1,1,2]
    # Level 2: [0,1,2,3,2,3,3,4]
    words = torch.tensor([[0, 1]], device=simple_hierarchy.device)
    
    # Default decoding (full expansion)
    tokens = simple_hierarchy.decode_sequence(words)
    assert torch.equal(tokens, 
        torch.tensor([[0, 1, 2, 3, 2, 3, 3, 4]], device=simple_hierarchy.device))
    
    # Decode to intermediate level
    chars = simple_hierarchy.decode_sequence(words, target_level=1)
    assert torch.equal(chars, torch.tensor([[0, 1, 1, 2]], device=simple_hierarchy.device))


def test_batch_decoding(simple_hierarchy):
    """Test decoding of batched sequences."""
    words = torch.tensor([
        [0, 1],  # First sequence: word 0 followed by word 1
        [1, 0],  # Second sequence: word 1 followed by word 0
    ], device=simple_hierarchy.device)
    
    # Default decoding (full expansion)
    tokens = simple_hierarchy.decode_sequence(words)
    
    expected = torch.tensor([
        [0, 1, 2, 3, 2, 3, 3, 4],  # Decoded first sequence
        [2, 3, 3, 4, 0, 1, 2, 3],  # Decoded second sequence
    ], device=simple_hierarchy.device)
    
    assert torch.equal(tokens, expected)


def test_invalid_level_decoding(simple_hierarchy):
    """Test validation of decoding levels."""
    words = torch.tensor([[0]], device=simple_hierarchy.device)
    
    # Invalid start level (too high)
    with pytest.raises(ValueError, match="Invalid start_level"):
        simple_hierarchy.decode_sequence(words, start_level=3)
        
    # Invalid target level (negative)
    with pytest.raises(ValueError, match="Invalid target_level"):
        simple_hierarchy.decode_sequence(words, target_level=-1)
    
    # Can't decode upward
    with pytest.raises(ValueError, match="Can only decode to same or higher levels"):
        simple_hierarchy.decode_sequence(words, start_level=1, target_level=0)


def test_default_decoding(simple_hierarchy):
    """Test default decoding behavior."""
    # Single token at most abstract level
    word = torch.tensor([[0]], device=simple_hierarchy.device)
    
    # These should all be equivalent
    full_decode = simple_hierarchy.decode_sequence(word)
    explicit_decode = simple_hierarchy.decode_sequence(word, start_level=0, target_level=2)
    
    assert torch.equal(full_decode, explicit_decode)
    assert torch.equal(full_decode, torch.tensor([[0, 1, 2, 3]], device=simple_hierarchy.device))


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
    # Create two-level hierarchy for testing
    levels = [
        VocabLevel(  # Base tokens
            vocab_size=2,
            chunk_size=1,
            sequences={0: (0,), 1: (1,)},
        ),
        VocabLevel(  # Higher level tokens
            vocab_size=2,
            chunk_size=2,
            sequences={0: (0, 1), 1: (1, 0)},
        ),
    ]
    hierarchy = VocabHierarchy(levels, device="cpu")

    # Test decoding maintains device
    tokens = torch.tensor([[0]], device="cpu")
    result = hierarchy.decode_sequence(tokens, start_level=1, target_level=0)
    assert result.device.type == "cpu"

    # Test with same level returns input unchanged
    tokens = torch.tensor([[0]], device="cpu")
    result = hierarchy.decode_sequence(tokens, start_level=0, target_level=0)
    assert result.device.type == "cpu"

    # Input on different device gets moved
    if torch.cuda.is_available():
        hierarchy = VocabHierarchy(levels, device="cuda")
        result = hierarchy.decode_sequence(tokens, start_level=1, target_level=0)
        assert result.device.type == "cuda"
