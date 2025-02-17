# tests/test_graph.py

"""Tests for the colored graph transition system."""

import numpy as np
import pytest
from faux_lingo.core.graph import ColoredGraph, GraphConfig

def test_graph_config_validation():
    """Test that GraphConfig validates parameters correctly."""
    # Valid configuration
    config = GraphConfig(
        num_colors=3,
        avg_degree=2,
        vocab_size=10,
        sigma=1.0,
        epsilon=0.1,
        random_color_transitions=False
    )
    config.validate()  # Should not raise

    # Invalid: negative number of colors
    with pytest.raises(ValueError, match="num_colors must be positive"):
        GraphConfig(
            num_colors=-1,
            avg_degree=2,
            vocab_size=10
        ).validate()

    # Invalid: average degree exceeds number of colors
    with pytest.raises(ValueError, match="avg_degree cannot exceed number of colors"):
        GraphConfig(
            num_colors=2,
            avg_degree=3,
            vocab_size=10
        ).validate()

    # Invalid: vocabulary size too small
    with pytest.raises(ValueError, match="vocab_size must be at least num_colors"):
        GraphConfig(
            num_colors=5,
            avg_degree=2,
            vocab_size=3
        ).validate()

def test_color_assignment():
    """Test assignment of colors to words."""
    config = GraphConfig(
        num_colors=3,
        avg_degree=2,
        vocab_size=10
    )
    graph = ColoredGraph(config, seed=42)
    graph.assign_colors()

    # Check all words have been assigned colors
    assert len(graph.word_colors) == 10
    # Check colors are within valid range
    assert all(0 <= c < 3 for c in graph.word_colors.values())
    # Check at least one word of each color exists
    assert len(set(graph.word_colors.values())) == 3

def test_color_transition_matrix():
    """Test generation of color transition matrix."""
    config = GraphConfig(
        num_colors=3,
        avg_degree=2,
        vocab_size=10,
        random_color_transitions=True,
        sigma=1.0,
        epsilon=0.1
    )
    graph = ColoredGraph(config, seed=42)
    graph.assign_colors()
    graph.build_color_transitions()

    assert graph.color_matrix is not None
    assert graph.color_matrix.shape == (3, 3)
    # Check rows sum to 1 (within floating point precision)
    assert np.allclose(graph.color_matrix.sum(axis=1), 1.0)
    # Check all probabilities are non-negative
    assert np.all(graph.color_matrix >= 0)

def test_transition_matrix_properties():
    """Test properties of generated transition matrix."""
    config = GraphConfig(
        num_colors=3,
        avg_degree=2,
        vocab_size=10
    )
    graph = ColoredGraph(config, seed=42)
    graph.build()

    T = graph.transition_matrix
    assert T is not None
    assert T.shape == (10, 10)
    # Check rows sum to 1 or 0
    row_sums = T.sum(axis=1)
    assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0)))
    # Check all probabilities are non-negative
    assert np.all(T >= 0)
    # Check sparsity - each row should have at most avg_degree nonzero entries
    assert all(np.count_nonzero(row) <= config.avg_degree for row in T)

def test_color_constrained_transitions():
    """Test that transitions follow color constraints."""
    config = GraphConfig(
        num_colors=3,
        avg_degree=2,
        vocab_size=10,
        random_color_transitions=False  # Use uniform color transitions
    )
    graph = ColoredGraph(config, seed=42)
    graph.build()

    # For each word, check its transitions are to words of different colors
    T = graph.transition_matrix
    for i in range(config.vocab_size):
        if T[i].sum() > 0:  # If row has any transitions
            # Get colors of target words
            target_indices = np.nonzero(T[i])[0]
            target_colors = [graph.word_colors[j] for j in target_indices]
            # Check targets have different colors
            assert len(set(target_colors)) == len(target_colors)

def test_deterministic_generation():
    """Test that setting a seed produces deterministic results."""
    config = GraphConfig(
        num_colors=3,
        avg_degree=2,
        vocab_size=10,
        random_color_transitions=True
    )
    
    graph1 = ColoredGraph(config, seed=42)
    result1 = graph1.build()
    
    graph2 = ColoredGraph(config, seed=42)
    result2 = graph2.build()

    assert graph1.word_colors == graph2.word_colors
    assert np.array_equal(graph1.transition_matrix, graph2.transition_matrix)
    assert np.array_equal(graph1.color_matrix, graph2.color_matrix)

def test_steady_state_distribution():
    """Test computation of steady state distribution."""
    config = GraphConfig(
        num_colors=3,
        avg_degree=2,
        vocab_size=10
    )
    graph = ColoredGraph(config, seed=42)
    graph.build()

    pi = graph.get_steady_state()
    # Check it's a probability distribution
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(pi >= 0)
    # Check it's actually steady
    pi_next = pi @ graph.transition_matrix
    assert np.allclose(pi, pi_next)
