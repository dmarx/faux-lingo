import pytest
import torch
import tempfile
from pathlib import Path
import numpy as np
from prob_color_gen import (
    ProbColorConstrainedGenerator,
    LanguageParams,
    validate_shapes
)

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture
def simple_generator(device):
    """Create a simple generator with 3 colors and 5 topics"""
    color_fractions = [3, 5, 2]  # Will normalize to [0.3, 0.5, 0.2]
    color_transitions = torch.tensor([
        [1.0, 0.5, 0.1],
        [0.4, 1.0, 0.7],
        [0.2, 0.6, 1.0]
    ])
    
    return ProbColorConstrainedGenerator(
        n_topics=5,
        vocab_size=100,
        color_fractions=color_fractions,
        color_transitions=color_transitions,
        device=device
    )

def test_initialization(device):
    """Test basic initialization with valid parameters"""
    gen = ProbColorConstrainedGenerator(
        n_topics=5,
        vocab_size=100,
        color_fractions=[1, 1, 1],
        color_transitions=torch.ones(3, 3),
        device=device
    )
    
    assert gen.n_topics == 5
    assert gen.vocab_size == 100
    assert gen.n_colors == 3
    assert torch.allclose(gen.color_fractions.sum(), torch.tensor(1.0))

def test_color_fraction_normalization(device):
    """Test that color fractions are properly normalized"""
    gen = ProbColorConstrainedGenerator(
        n_topics=5,
        vocab_size=100,
        color_fractions=[3, 5, 2],  # Should normalize to [0.3, 0.5, 0.2]
        color_transitions=torch.ones(3, 3),
        device=device
    )
    
    expected = torch.tensor([0.3, 0.5, 0.2], device=device)
    assert torch.allclose(gen.color_fractions, expected, atol=1e-6)

def test_invalid_color_transitions(device):
    """Test that invalid color transitions raise appropriate errors"""
    with pytest.raises(ValueError):
        # Wrong shape
        ProbColorConstrainedGenerator(
            n_topics=5,
            vocab_size=100,
            color_fractions=[1, 1, 1],
            color_transitions=torch.ones(2, 3),  # Wrong shape
            device=device
        )
    
    with pytest.raises(ValueError):
        # Negative transitions
        ProbColorConstrainedGenerator(
            n_topics=5,
            vocab_size=100,
            color_fractions=[1, 1, 1],
            color_transitions=torch.tensor([
                [1.0, -0.5, 0.1],
                [0.4, 1.0, 0.7],
                [0.2, 0.6, 1.0]
            ]),
            device=device
        )

def test_topic_vector_orthonormality(simple_generator):
    """Test that topic vectors are orthonormal"""
    vectors = simple_generator.topic_vectors
    
    # Test orthogonality
    product = torch.mm(vectors, vectors.T)
    identity = torch.eye(vectors.shape[0], device=vectors.device)
    assert torch.allclose(product, identity, atol=1e-6)
    
    # Test normalization
    norms = torch.norm(vectors, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_transition_matrix_properties(simple_generator):
    """Test properties of generated transition matrices"""
    matrices, mixtures = simple_generator.generate_transitions(batch_size=10)
    
    # Test shape
    assert matrices.shape == (10, simple_generator.vocab_size, simple_generator.vocab_size)
    assert mixtures.shape == (10, simple_generator.n_topics)
    
    # Test stochastic properties
    assert torch.allclose(matrices.sum(dim=-1), torch.ones_like(matrices[:, 0]))
    assert torch.all(matrices >= 0)

def test_sequence_generation(simple_generator):
    """Test basic sequence generation properties"""
    batch_size = 32
    seq_length = 50
    sequences, mixtures = simple_generator.sample_sequences(
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    # Test shapes
    assert sequences.shape == (batch_size, seq_length)
    assert mixtures.shape == (batch_size, simple_generator.n_topics)
    
    # Test value ranges
    assert torch.all(sequences >= 0)
    assert torch.all(sequences < simple_generator.vocab_size)
    
    # Test mixture properties
    assert torch.allclose(mixtures.sum(dim=-1), torch.ones(batch_size))
    assert torch.all(mixtures >= 0)

def test_color_constraints(simple_generator):
    """Test that color transition constraints are respected"""
    batch_size = 1000
    seq_length = 50
    sequences, _ = simple_generator.sample_sequences(
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    # Convert sequences to color sequences
    color_sequences = torch.tensor([
        [simple_generator.get_color(idx.item()) for idx in seq]
        for seq in sequences
    ], device=sequences.device)
    
    # Test transition probabilities
    for i in range(simple_generator.n_colors):
        for j in range(simple_generator.n_colors):
            mask = color_sequences[:, :-1] == i
            if mask.any():
                next_colors = color_sequences[:, 1:][mask]
                if simple_generator.color_transitions[i, j] == 0:
                    # Should never see forbidden transitions
                    assert torch.sum(next_colors == j) == 0

def test_serialization(simple_generator, tmp_path):
    """Test saving and loading language parameters"""
    save_path = tmp_path / "test_language.tensors"
    
    # Save
    simple_generator.save_language(save_path)
    
    # Load
    loaded = ProbColorConstrainedGenerator.load_language(save_path)
    
    # Compare parameters
    assert loaded.n_topics == simple_generator.n_topics
    assert loaded.vocab_size == simple_generator.vocab_size
    assert torch.allclose(loaded.color_fractions, simple_generator.color_fractions)
    assert torch.allclose(loaded.color_transitions, simple_generator.color_transitions)
    assert torch.allclose(loaded.topic_vectors, simple_generator.topic_vectors)

def test_deterministic_generation(simple_generator):
    """Test that same mixtures produce same transition matrices"""
    batch_size = 10
    mixtures = torch.rand(batch_size, simple_generator.n_topics)
    mixtures = mixtures / mixtures.sum(dim=-1, keepdim=True)
    
    matrices1, _ = simple_generator.generate_transitions(
        batch_size=batch_size,
        mixtures=mixtures
    )
    
    matrices2, _ = simple_generator.generate_transitions(
        batch_size=batch_size,
        mixtures=mixtures
    )
    
    assert torch.allclose(matrices1, matrices2)

def test_temperature_effect(simple_generator):
    """Test that temperature affects transition probabilities"""
    matrices_hot, _ = simple_generator.generate_transitions(
        batch_size=1,
        temperature=0.1  # Sharp distributions
    )
    
    matrices_cold, _ = simple_generator.generate_transitions(
        batch_size=1,
        temperature=2.0  # Smooth distributions
    )
    
    # Hot temperature should give more peaked distributions
    hot_entropy = -(matrices_hot * torch.log(matrices_hot + 1e-10)).sum()
    cold_entropy = -(matrices_cold * torch.log(matrices_cold + 1e-10)).sum()
    
    assert hot_entropy < cold_entropy

def test_start_color_constraint(simple_generator):
    """Test that sequences start with specified color when requested"""
    start_color = 1
    batch_size = 100
    
    sequences, _ = simple_generator.sample_sequences(
        batch_size=batch_size,
        start_color=start_color
    )
    
    # Check that all sequences start with tokens from the specified color
    start_colors = torch.tensor([
        simple_generator.get_color(idx.item())
        for idx in sequences[:, 0]
    ])
    
    assert torch.all(start_colors == start_color)

def test_shape_validation():
    """Test the shape validation utility"""
    tensors = {
        'a': torch.randn(2, 3),
        'b': torch.randn(4, 4)
    }
    
    # Correct shapes
    validate_shapes(tensors, {
        'a': (2, 3),
        'b': (4, 4)
    })
    
    # Wrong shapes
    with pytest.raises(ValueError):
        validate_shapes(tensors, {
            'a': (3, 2),
            'b': (4, 4)
        })

def test_get_color_range(simple_generator):
    """Test color range retrieval"""
    for color in range(simple_generator.n_colors):
        start, end = simple_generator.get_color_range(color)
        assert start < end
        assert start >= 0
        assert end <= simple_generator.vocab_size
    
    with pytest.raises(ValueError):
        simple_generator.get_color_range(-1)
    
    with pytest.raises(ValueError):
        simple_generator.get_color_range(simple_generator.n_colors)
