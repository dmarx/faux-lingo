# tests/test_topics.py

"""Tests for the topic modeling system."""

import numpy as np
import pytest

from faux_lingo.core.topics import TopicConfig, TopicModel


# Test fixtures
@pytest.fixture
def word_colors():
    """Sample word color assignments."""
    return {i: i % 3 for i in range(9)}  # 9 words, 3 colors


@pytest.fixture
def base_matrix():
    """Sample base transition matrix."""
    # 9x9 matrix with 2 transitions per row
    matrix = np.zeros((9, 9))
    for i in range(9):
        targets = [(i + 1) % 9, (i + 2) % 9]
        matrix[i, targets] = 0.5
    return matrix


def test_topic_config_validation():
    """Test that TopicConfig validates parameters correctly."""
    # Valid configuration
    config = TopicConfig(num_topics=3, modes_per_color=2, attachment_bias=0.5)
    config.validate()  # Should not raise

    # Invalid: negative number of topics
    with pytest.raises(ValueError, match="num_topics must be positive"):
        TopicConfig(num_topics=-1, modes_per_color=2, attachment_bias=0.5).validate()

    # Invalid: negative attachment bias
    with pytest.raises(ValueError, match="attachment_bias must be non-negative"):
        TopicConfig(num_topics=3, modes_per_color=2, attachment_bias=-0.1).validate()


def test_topic_model_initialization(word_colors, base_matrix):
    """Test TopicModel initialization and state."""
    config = TopicConfig(num_topics=3, modes_per_color=1, attachment_bias=0.5)
    model = TopicModel(config, word_colors, base_matrix, seed=42)

    assert model.vocab_size == 9
    assert model.num_colors == 3
    assert len(model.topic_modes) == 0
    assert len(model.topic_matrices) == 0
    assert len(model.topic_distributions) == 0


def test_topic_mode_sampling(word_colors, base_matrix):
    """Test sampling of topic mode words."""
    config = TopicConfig(num_topics=2, modes_per_color=1, attachment_bias=0.5)
    model = TopicModel(config, word_colors, base_matrix, seed=42)
    model.sample_topic_modes()

    assert len(model.topic_modes) == 2  # Two topics
    for modes in model.topic_modes:
        assert len(modes) == 3  # Three colors
        for color in range(3):
            assert color in modes  # Each color has modes
            assert len(modes[color]) == 1  # One mode per color
            # Mode words have correct colors
            for word in modes[color]:
                assert word_colors[word] == color


def test_topic_matrix_generation(word_colors, base_matrix):
    """Test generation of topic-specific transition matrices."""
    config = TopicConfig(num_topics=2, modes_per_color=1, attachment_bias=0.5)
    model = TopicModel(config, word_colors, base_matrix, seed=42)
    model.sample_topic_modes()
    model.build_topic_matrices()

    assert len(model.topic_matrices) == 2
    for matrix in model.topic_matrices:
        assert matrix.shape == base_matrix.shape
        # Check row stochasticity
        row_sums = matrix.sum(axis=1)
        assert np.all(np.isclose(row_sums[row_sums > 0], 1.0))
        # Check non-negativity
        assert np.all(matrix >= 0)


def test_mode_word_boost(word_colors, base_matrix):
    """Test that mode words receive higher transition probabilities."""
    config = TopicConfig(
        num_topics=1,
        modes_per_color=1,
        attachment_bias=1.0,  # Double probability to mode words
    )
    model = TopicModel(config, word_colors, base_matrix, seed=42)
    model.sample_topic_modes()
    model.build_topic_matrices()

    topic_matrix = model.topic_matrices[0]
    modes = model.topic_modes[0]

    # For each word with transitions
    for i in range(model.vocab_size):
        if topic_matrix[i].sum() > 0:
            # Get its transition targets
            targets = np.nonzero(topic_matrix[i])[0]
            for j in targets:
                color = word_colors[j]
                if j in modes[color]:
                    # If target is a mode word, its probability should be higher
                    assert topic_matrix[i, j] > base_matrix[i, j]


def test_topic_distribution_computation(word_colors, base_matrix):
    """Test computation of topic-specific stationary distributions."""
    config = TopicConfig(num_topics=2, modes_per_color=1, attachment_bias=0.5)
    model = TopicModel(config, word_colors, base_matrix, seed=42)
    model.build()

    assert len(model.topic_distributions) == 2
    for dist in model.topic_distributions:
        # Check it's a probability distribution
        assert np.isclose(dist.sum(), 1.0)
        assert np.all(dist >= 0)
        # Check mode words have higher probability
        modes = {
            word for mode_set in model.topic_modes[0].values() for word in mode_set
        }
        mode_probs = dist[list(modes)]
        non_mode_probs = dist[[i for i in range(len(dist)) if i not in modes]]
        assert np.mean(mode_probs) > np.mean(non_mode_probs)


def test_topic_entropy_computation(word_colors, base_matrix):
    """Test computation of topic entropy measures."""
    config = TopicConfig(num_topics=1, modes_per_color=1, attachment_bias=0.5)
    model = TopicModel(config, word_colors, base_matrix, seed=42)
    model.build()

    stationary_entropy, conditional_entropy = model.get_topic_entropy(0)

    # Basic sanity checks
    assert stationary_entropy >= 0
    assert conditional_entropy >= 0
    # Stationary entropy should be less than log2(vocab_size)
    assert stationary_entropy <= np.log2(model.vocab_size)
    # With our attachment bias, entropy should be less than base matrix
    base_stationary = -np.sum(
        model.topic_distributions[0] * np.log2(model.topic_distributions[0] + 1e-12)
    )
    assert stationary_entropy <= base_stationary


def test_deterministic_generation(word_colors, base_matrix):
    """Test that setting a seed produces deterministic results."""
    config = TopicConfig(num_topics=2, modes_per_color=1, attachment_bias=0.5)

    model1 = TopicModel(config, word_colors, base_matrix, seed=42)
    result1 = model1.build()

    model2 = TopicModel(config, word_colors, base_matrix, seed=42)
    result2 = model2.build()

    assert model1.topic_modes == model2.topic_modes
    assert all(
        np.array_equal(m1, m2)
        for m1, m2 in zip(model1.topic_matrices, model2.topic_matrices)
    )
    assert all(
        np.array_equal(d1, d2)
        for d1, d2 in zip(model1.topic_distributions, model2.topic_distributions)
    )
