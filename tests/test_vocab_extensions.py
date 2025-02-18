# faux_lingo/tests/test_vocab_extensions.py
"""Tests for vocabulary extension functionality."""

import pytest
import torch

from faux_lingo.core.vocab_builder import create_word_hierarchy
from faux_lingo.core.vocab_extensions import (
    AugmentationConfig,
    MultiMappingHierarchy,
    MultiMappingLevel,
    SequenceAugmenter,
    convert_to_multi_mapping,
)


@pytest.fixture
def simple_multi_level():
    """Create simple multi-mapping level for testing."""
    return MultiMappingLevel(
        vocab_size=2,
        chunk_size=2,
        sequences={
            0: [((0, 1), 0.7), ((1, 0), 0.3)],  # Two variants for token 0
            1: [((1, 1), 1.0)],  # Single mapping for token 1
        },
    )


@pytest.fixture
def simple_augmenter():
    """Create sequence augmenter with test configuration."""
    config = AugmentationConfig(
        deletion_prob=0.2,
        insertion_prob=0.2,
        substitution_prob=0.2,
        transposition_prob=0.2,
        seed=42,
    )
    return SequenceAugmenter(vocab_size=4, config=config)


def test_multi_level_validation():
    """Test validation of multi-mapping level properties."""
    # Valid level
    level = MultiMappingLevel(
        vocab_size=1,
        chunk_size=2,
        sequences={0: [((0, 1), 1.0)]},
    )
    assert level.vocab_size == 1

    # Invalid probabilities
    with pytest.raises(ValueError, match="do not sum to 1"):
        MultiMappingLevel(
            vocab_size=1,
            chunk_size=2,
            sequences={0: [((0, 1), 0.5)]},  # Prob < 1
        )

    # No sequences
    with pytest.raises(ValueError, match="No sequences defined"):
        MultiMappingLevel(
            vocab_size=1,
            chunk_size=2,
            sequences={0: []},
        )


def test_multi_hierarchy_decoding(simple_multi_level):
    """Test sequence decoding with multiple mappings."""
    hierarchy = MultiMappingHierarchy([simple_multi_level], device="cpu")
    
    # Create test sequence
    tokens = torch.tensor([[0, 1]], device="cpu")
    
    # Test reproducibility with seed
    torch.manual_seed(42)
    result1 = hierarchy.decode_sequence(tokens, start_level=0, target_level=0)
    
    torch.manual_seed(42)
    result2 = hierarchy.decode_sequence(tokens, start_level=0, target_level=0)
    
    assert torch.equal(result1, result2)


def test_augmenter_operations(simple_augmenter):
    """Test individual augmentation operations."""
    sequence = [0, 1, 2, 3]

    # Test deletion
    deleted = simple_augmenter._delete(sequence.copy())
    assert len(deleted) == len(sequence) - 1

    # Test insertion
    inserted = simple_augmenter._insert(sequence.copy())
    assert len(inserted) == len(sequence) + 1

    # Test substitution
    substituted = simple_augmenter._substitute(sequence.copy())
    assert len(substituted) == len(sequence)
    assert substituted != sequence

    # Test transposition
    transposed = simple_augmenter._transpose(sequence.copy())
    assert len(transposed) == len(sequence)
    assert transposed != sequence


def test_augmenter_sequence_handling(simple_augmenter):
    """Test sequence augmentation edge cases."""
    # Empty sequence
    assert simple_augmenter.augment_sequence(()) == ()

    # Single token
    single = (0,)
    augmented = simple_augmenter.augment_sequence(single)
    assert isinstance(augmented, tuple)
    assert len(augmented) > 0

    # Reproducibility
    torch.manual_seed(42)
    result1 = simple_augmenter.augment_sequence((0, 1, 2))
    
    torch.manual_seed(42)
    result2 = simple_augmenter.augment_sequence((0, 1, 2))
    
    assert result1 == result2


def test_hierarchy_conversion():
    """Test conversion from standard to multi-mapping hierarchy."""
    # Create simple word hierarchy
    hierarchy = create_word_hierarchy(
        token_vocab_size=4,
        n_chars=3,
        n_words=2,
        chars_per_word=2,
        seed=42,
    )

    # Create augmenter for variants
    config = AugmentationConfig(
        deletion_prob=0.1,
        insertion_prob=0.1,
        substitution_prob=0.1,
        transposition_prob=0.1,
        seed=42,
    )
    augmenter = SequenceAugmenter(vocab_size=4, config=config)

    # Convert to multi-mapping
    multi_hierarchy = convert_to_multi_mapping(
        hierarchy,
        augmenter=augmenter,
        n_variants=3,
    )

    # Check structure preserved
    assert len(multi_hierarchy.levels) == len(hierarchy.levels)
    
    # Check variants generated
    for level in multi_hierarchy.levels:
        for token, variants in level.sequences.items():
            # At least original sequence plus some variants
            assert len(variants) > 1
            # Probabilities sum to 1
            probs = sum(prob for _, prob in variants)
            assert torch.isclose(torch.tensor(probs), torch.tensor(1.0))


def test_augmentation_config_validation():
    """Test validation of augmentation configuration."""
    # Valid config
    config = AugmentationConfig(
        deletion_prob=0.1,
        insertion_prob=0.1,
        substitution_prob=0.1,
        transposition_prob=0.1,
    )
    assert config.deletion_prob == 0.1

    # Invalid probability value
    with pytest.raises(ValueError, match="between 0 and 1"):
        AugmentationConfig(deletion_prob=1.5)

    # Sum too large
    with pytest.raises(ValueError, match="must not exceed 1"):
        AugmentationConfig(
            deletion_prob=0.3,
            insertion_prob=0.3,
            substitution_prob=0.3,
            transposition_prob=0.3,
        )


def test_device_handling():
    """Test device placement and consistency."""
    # Create hierarchy on CPU
    level = MultiMappingLevel(
        vocab_size=1,
        chunk_size=2,
        sequences={0: [((0, 1), 1.0)]},
    )
    hierarchy = MultiMappingHierarchy([level], device="cpu")

    # Test decoding maintains device
    tokens = torch.tensor([[0]], device="cpu")
    result = hierarchy.decode_sequence(tokens, start_level=0, target_level=0)
    assert result.device.type == "cpu"

    # Test with same level returns input unchanged
    result = hierarchy.decode_sequence(tokens, start_level=0, target_level=0)
    assert result.device.type == "cpu"
