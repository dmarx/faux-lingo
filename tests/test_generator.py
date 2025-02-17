# tests/test_generator.py

"""Tests for the document generation system."""

from pathlib import Path

import numpy as np
import pytest

from faux_lingo.core.generator import (
    ArtifactGenerator,
    DocumentGenerator,
    GeneratorConfig,
)
from faux_lingo.core.graph import GraphConfig
from faux_lingo.core.topics import TopicConfig
from faux_lingo.core.vocabulary import VocabConfig


@pytest.fixture
def small_config():
    """Create a small configuration for testing."""
    return GeneratorConfig(
        vocab_config=VocabConfig(
            token_vocab_size=4,
            rune_vocab_size=4,
            char_vocab_size=4,
            word_vocab_size=8,
            tokens_per_rune=1,
            runes_per_char=2,
            chars_per_word=2,
        ),
        graph_config=GraphConfig(
            num_colors=3,
            avg_degree=2,
            vocab_size=8,
            sigma=1.0,
            epsilon=0.1,
            random_color_transitions=False,
        ),
        topic_config=TopicConfig(
            num_topics=2,
            modes_per_color=1,
            attachment_bias=0.5,
        ),
    )


def test_generator_config_validation(small_config):
    """Test configuration validation."""
    small_config.validate()  # Should not raise

    # Test vocab size mismatch
    bad_config = GeneratorConfig(
        vocab_config=VocabConfig(
            token_vocab_size=4,
            rune_vocab_size=4,
            char_vocab_size=4,
            word_vocab_size=8,
            tokens_per_rune=1,
            runes_per_char=2,
            chars_per_word=2,
        ),
        graph_config=GraphConfig(
            num_colors=3,
            avg_degree=2,
            vocab_size=20,  # Mismatched with word_vocab_size
            sigma=1.0,
            epsilon=0.1,
            random_color_transitions=False,
        ),
        topic_config=small_config.topic_config,
    )
    with pytest.raises(ValueError, match="Vocabulary size mismatch"):
        bad_config.validate()


def test_artifact_generation(small_config):
    """Test generation of all artifacts."""
    generator = ArtifactGenerator(small_config, seed=42)
    artifacts = generator.build()

    # Check all required artifacts are present
    required_artifacts = {
        "token_vocab",
        "rune_vocab",
        "char_vocab",
        "word_vocab",
        "word_colors",
        "transition_matrix",
        "color_matrix",
        "topic_modes",
        "topic_matrices",
        "topic_distributions",
    }
    assert all(key in artifacts for key in required_artifacts)

    # Check vocabulary sizes
    assert len(artifacts["token_vocab"]) == small_config.vocab_config.token_vocab_size
    assert len(artifacts["word_vocab"]) == small_config.vocab_config.word_vocab_size

    # Check transition matrices
    assert len(artifacts["topic_matrices"]) == small_config.topic_config.num_topics
    assert len(artifacts["topic_distributions"]) == small_config.topic_config.num_topics


def test_artifact_serialization(small_config, tmp_path):
    """Test saving and loading of artifacts."""
    generator = ArtifactGenerator(small_config, seed=42)
    artifacts = generator.build()

    # Save artifacts
    generator.save(tmp_path)

    # Load artifacts
    loaded_generator = ArtifactGenerator.load(tmp_path, small_config)

    # Compare artifacts
    for key in artifacts:
        if isinstance(artifacts[key], np.ndarray):
            assert np.array_equal(artifacts[key], loaded_generator.artifacts[key])
        elif (
            isinstance(artifacts[key], (list, tuple))
            and len(artifacts[key]) > 0
            and isinstance(artifacts[key][0], np.ndarray)
        ):
            # Handle lists/tuples of numpy arrays
            assert len(artifacts[key]) == len(loaded_generator.artifacts[key])
            for a1, a2 in zip(artifacts[key], loaded_generator.artifacts[key]):
                assert np.array_equal(a1, a2)
        else:
            assert artifacts[key] == loaded_generator.artifacts[key]


def test_document_generation(small_config):
    """Test generation of individual documents."""
    # Generate artifacts
    artifact_gen = ArtifactGenerator(small_config, seed=42)
    artifacts = artifact_gen.build()

    # Create document generator
    doc_gen = DocumentGenerator(
        artifacts=artifacts,
        doc_topic_alpha=0.5,
        include_whitespace=True,
        include_markers=True,
        seed=42,
    )

    # Generate a document
    doc = doc_gen.generate(doc_length=5)

    # Check it's a valid numpy array
    assert isinstance(doc, np.ndarray)
    assert doc.dtype == np.int64

    # Check special tokens
    if doc_gen.include_markers:
        assert doc[0] == doc_gen.BOD_TOKEN
        assert doc[-1] == doc_gen.EOD_TOKEN


def test_document_entropy_computation(small_config):
    """Test entropy computation during document generation."""
    # Generate artifacts
    artifact_gen = ArtifactGenerator(small_config, seed=42)
    artifacts = artifact_gen.build()

    # Create document generator
    doc_gen = DocumentGenerator(
        artifacts=artifacts,
        doc_topic_alpha=0.5,
        include_whitespace=True,
        include_markers=True,
        seed=42,
    )

    # Generate a document with entropy measures
    doc, entropy, perplexity = doc_gen.generate(doc_length=5, return_entropy=True)

    # Check entropy measures
    assert entropy >= 0
    assert perplexity >= 1
    assert np.isclose(perplexity, 2**entropy)


def test_deterministic_document_generation(small_config):
    """Test that document generation is deterministic with fixed seed."""
    # Generate artifacts
    artifact_gen = ArtifactGenerator(small_config, seed=42)
    artifacts = artifact_gen.build()

    # Create two document generators with same seed
    doc_gen1 = DocumentGenerator(artifacts=artifacts, seed=42)
    doc_gen2 = DocumentGenerator(artifacts=artifacts, seed=42)

    # Generate documents
    doc1 = doc_gen1.generate(doc_length=5)
    doc2 = doc_gen2.generate(doc_length=5)

    # Check they're identical
    assert np.array_equal(doc1, doc2)


def test_document_token_validity(small_config):
    """Test that generated documents contain valid tokens."""
    # Generate artifacts
    artifact_gen = ArtifactGenerator(small_config, seed=42)
    artifacts = artifact_gen.build()

    # Create document generator
    doc_gen = DocumentGenerator(artifacts=artifacts, seed=42)

    # Generate a document
    doc = doc_gen.generate(doc_length=5)

    # Get valid token range
    max_token = small_config.vocab_config.token_vocab_size - 1
    special_tokens = {doc_gen.BOD_TOKEN, doc_gen.EOD_TOKEN, doc_gen.WS_TOKEN}

    # Check all tokens are either valid vocabulary tokens or special tokens
    for token in doc:
        assert (0 <= token <= max_token) or (token in special_tokens)
