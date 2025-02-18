# faux_lingo/tests/test_entropy.py
"""Tests for entropy analysis functionality."""

import pytest
import torch

from faux_lingo.analysis.entropy import EntropyAnalyzer, EntropyMetrics
from faux_lingo.core.generator import GeneratedSequences, SequenceGenerator


@pytest.fixture
def simple_analyzer():
    """Create analyzer with simple uniform generator."""
    generator = SequenceGenerator.create_uniform(
        vocab_size=9,
        n_topics=2,
        color_fractions=[1, 1, 1],  # Three equal color classes
    )
    return EntropyAnalyzer(generator.transition_model)


@pytest.fixture
def sample_sequences(simple_analyzer):
    """Generate sample sequences for testing."""
    generator = SequenceGenerator(simple_analyzer.transition_model)
    return generator.generate(batch_size=10, seq_length=20)


def test_metrics_zero():
    """Test zero initialization of metrics."""
    metrics = EntropyMetrics.zero()
    assert metrics.color_entropy == 0.0
    assert metrics.topic_entropy == 0.0
    assert metrics.token_entropy == 0.0


def test_color_entropy(simple_analyzer):
    """Test color entropy computation with different transition rules."""
    # Generate sequences with uniform transitions
    generator = SequenceGenerator(simple_analyzer.transition_model)
    uniform_sequences = generator.generate(batch_size=10, seq_length=20)
    uniform_metrics = simple_analyzer.analyze_sequences(uniform_sequences)

    # Generate sequences with deterministic transitions
    det_weights = torch.eye(3)  # Only self-transitions allowed
    simple_analyzer.transition_model.color_space.transition_weights = det_weights
    det_generator = SequenceGenerator(simple_analyzer.transition_model)
    det_sequences = det_generator.generate(batch_size=10, seq_length=20)
    det_metrics = simple_analyzer.analyze_sequences(det_sequences)

    # Deterministic transitions should have lower entropy
    assert det_metrics.color_entropy < uniform_metrics.color_entropy

    # Entropy should be non-negative and bounded
    assert det_metrics.color_entropy >= 0
    max_entropy = torch.log2(torch.tensor(3.0))  # log2(num_colors)
    assert det_metrics.color_entropy <= max_entropy


def test_topic_entropy(simple_analyzer):
    """Test topic entropy computation."""
    # Test with uniform mixture
    uniform_mix = torch.ones(4, 2) / 2
    sequences = GeneratedSequences(
        tokens=torch.zeros(4, 10, dtype=torch.long),  # Dummy tokens
        topic_mixtures=uniform_mix,
        log_probs=torch.zeros(4),
    )

    metrics = simple_analyzer.analyze_sequences(sequences)
    expected = torch.log2(torch.tensor(2.0))  # log2(num_topics)
    assert torch.isclose(
        torch.tensor(metrics.topic_entropy),
        expected,
        atol=1e-6
    )

    # Test with deterministic mixture
    det_mix = torch.zeros(4, 2)
    det_mix[:, 0] = 1.0  # All weight on first topic
    sequences.topic_mixtures = det_mix

    metrics = simple_analyzer.analyze_sequences(sequences)
    assert metrics.topic_entropy == 0.0


def test_token_entropy(simple_analyzer, sample_sequences):
    """Test token entropy computation."""
    metrics = simple_analyzer.analyze_sequences(sample_sequences)

    # Entropy should be non-negative and bounded
    assert metrics.token_entropy >= 0
    max_entropy = torch.log2(torch.tensor(9.0))  # log2(vocab_size)
    assert metrics.token_entropy <= max_entropy

    # Test with repeated tokens
    repeated = torch.zeros_like(sample_sequences.tokens)
    sequences = GeneratedSequences(
        tokens=repeated,
        topic_mixtures=sample_sequences.topic_mixtures,
        log_probs=sample_sequences.log_probs,
    )

    metrics = simple_analyzer.analyze_sequences(sequences)
    assert metrics.token_entropy == 0.0


def test_device_handling(simple_analyzer, sample_sequences):
    """Test device placement and consistency."""
    # All computations should happen on analyzer's device
    metrics = simple_analyzer.analyze_sequences(sample_sequences)

    # Move sequences to different device
    cpu_sequences = GeneratedSequences(
        tokens=sample_sequences.tokens.cpu(),
        topic_mixtures=sample_sequences.topic_mixtures.cpu(),
        log_probs=sample_sequences.log_probs.cpu(),
    )

    # Should still work and give same results
    cpu_metrics = simple_analyzer.analyze_sequences(cpu_sequences)
    assert metrics.color_entropy == cpu_metrics.color_entropy
    assert metrics.topic_entropy == cpu_metrics.topic_entropy
    assert metrics.token_entropy == cpu_metrics.token_entropy
