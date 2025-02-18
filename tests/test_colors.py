# faux_lingo/tests/test_colors.py
"""Tests for color space functionality."""

import pytest
import torch

from faux_lingo.core.colors import ColorSpace


def test_normalization():
    """Test that color fractions are properly normalized."""
    # Test with list input
    space = ColorSpace(color_fractions=[3, 2, 1], vocab_size=60)
    expected = torch.tensor([0.5, 0.333333, 0.166667], dtype=torch.float32)
    assert torch.allclose(space.mapping.fractions, expected, rtol=1e-5)

    # Test with tensor input
    space = ColorSpace(color_fractions=torch.tensor([1.0, 2.0, 3.0]), vocab_size=60)
    expected = torch.tensor([0.166667, 0.333333, 0.5], dtype=torch.float32)
    assert torch.allclose(space.mapping.fractions, expected, rtol=1e-5)


def test_boundaries():
    """Test token boundary calculations."""
    space = ColorSpace(color_fractions=[1, 1, 1], vocab_size=100)

    # Check boundary points
    assert space.get_color_range(0) == (0, 33)  # First third
    assert space.get_color_range(1) == (33, 66)  # Second third
    assert space.get_color_range(2) == (66, 100)  # Last third (gets remainder)

    # Test boundary adjustments
    space = ColorSpace(color_fractions=[1, 1], vocab_size=7)
    assert space.get_color_range(1)[1] == 7  # Last range should end at vocab_size


def test_color_lookup():
    """Test token to color mapping."""
    space = ColorSpace(color_fractions=[1, 2, 1], vocab_size=40)

    # With fractions [1,2,1], boundaries should be at [0, 10, 30, 40]
    # Print actual boundaries for debugging
    print(f"Boundaries: {space.mapping.boundaries}")
    print(f"Color of token 10: {space.get_color(10)}")
    print(f"Fractions: {space.mapping.fractions}")

    # Test boundaries
    assert space.get_color(0) == 0  # First color
    assert space.get_color(10) == 1  # Middle color
    assert space.get_color(39) == 2  # Last color

    # Test invalid indices
    with pytest.raises(ValueError):
        space.get_color(-1)
    with pytest.raises(ValueError):
        space.get_color(40)


def test_transition_weights():
    """Test transition weight validation and mask creation."""
    weights = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])

    space = ColorSpace(
        color_fractions=[1, 1, 1], vocab_size=9, transition_weights=weights
    )

    mask = space.get_transition_mask()

    # Check mask shape
    assert mask.shape == (9, 9)

    # Check block structure
    block_size = 3
    for i in range(3):
        for j in range(3):
            block = mask[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            assert torch.all(block == weights[i, j])

    # Test invalid weights
    with pytest.raises(ValueError):
        ColorSpace(
            color_fractions=[1, 1],
            vocab_size=10,
            transition_weights=weights,  # Wrong shape for n_colors=2
        )

    with pytest.raises(ValueError):
        ColorSpace(
            color_fractions=[1, 1],
            vocab_size=10,
            transition_weights=torch.tensor(
                [[-1.0, 1.0], [1.0, -1.0]]
            ),  # Negative weights
        )


def test_save_load(tmp_path):
    """Test serialization of color space."""
    weights = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    original = ColorSpace(
        color_fractions=[2, 3], vocab_size=100, transition_weights=weights
    )

    path = tmp_path / "color_space.pt"

    # Save and load
    original.save(path)
    loaded = ColorSpace.load(path)

    # Verify properties preserved
    assert original.vocab_size == loaded.vocab_size
    assert torch.allclose(original.mapping.fractions, loaded.mapping.fractions)
    assert torch.allclose(original.mapping.boundaries, loaded.mapping.boundaries)
    assert torch.allclose(original.transition_weights, loaded.transition_weights)


def test_device_handling():
    """Test device placement and movement."""
    # Default to CPU
    space = ColorSpace(color_fractions=[1, 1], vocab_size=10)
    assert space.mapping.boundaries.device.type == "cpu"
    assert space.transition_weights.device.type == "cpu"

    # Test device specification
    space = ColorSpace(color_fractions=[1, 1], vocab_size=10, device="cpu")
    assert space.mapping.boundaries.device.type == "cpu"

    # Test mask device matches space
    mask = space.get_transition_mask()
    assert mask.device.type == "cpu"
