# faux_lingo/core/generator.py

"""
Core generator system that combines vocabulary, graph, and topic components
to produce synthetic language data.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from .graph import ColoredGraph, GraphConfig
from .topics import TopicConfig, TopicModel
from .vocabulary import VocabBuilder, VocabConfig


@dataclass
class GeneratorConfig:
    """Complete configuration for the synthetic language generator."""

    vocab_config: VocabConfig
    graph_config: GraphConfig
    topic_config: TopicConfig

    def validate(self) -> None:
        """Validate all configuration components."""
        self.vocab_config.validate()
        self.graph_config.validate()
        self.topic_config.validate()

        # Cross-component validation
        if self.graph_config.vocab_size != self.vocab_config.word_vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: graph={self.graph_config.vocab_size}, "
                f"vocab={self.vocab_config.word_vocab_size}"
            )


class ArtifactGenerator:
    """
    Generates and manages all fixed artifacts needed for synthetic language
    generation.
    """

    def __init__(self, config: GeneratorConfig, seed: int | None = None):
        """Initialize the artifact generator system."""
        self.config = config
        self.config.validate()
        self._seed = seed

        # Initialize components
        self.vocab_builder = VocabBuilder(config.vocab_config, seed=seed)
        self.graph = ColoredGraph(config.graph_config, seed=seed)
        self.topic_model: TopicModel | None = None

        # Initialize artifact containers
        self.artifacts: Dict = {}

        logger.info("Initialized ArtifactGenerator with config: {}", config)

    def build(self) -> Dict:
        """Build all artifacts needed for document generation."""
        # Build hierarchical vocabularies
        vocab_artifacts = self.vocab_builder.build()
        self.artifacts.update(vocab_artifacts)

        # Build colored transition graph
        graph_artifacts = self.graph.build()
        self.artifacts.update(graph_artifacts)

        # Build topic model using graph outputs
        self.topic_model = TopicModel(
            config=self.config.topic_config,
            word_colors=graph_artifacts["word_colors"],
            base_matrix=graph_artifacts["transition_matrix"],
            seed=self._seed,
        )
        topic_artifacts = self.topic_model.build()
        self.artifacts.update(topic_artifacts)

        logger.info("Successfully built all artifacts")
        return self.artifacts
        
    def save(self, directory: str | Path) -> None:
        """Save artifacts to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    
        # Save numpy arrays and dictionaries separately
        for name, artifact in self.artifacts.items():
            if isinstance(artifact, np.ndarray):
                np.save(directory / f"{name}.npy", artifact)
            elif isinstance(artifact, dict):
                # Save dictionaries as JSON to preserve structure
                with open(directory / f"{name}.json", "w") as f:
                    json.dump(artifact, f)
            elif isinstance(artifact, (list, tuple)):
                np.save(directory / f"{name}.npy", np.array(artifact, dtype=object))
    
        logger.info("Saved artifacts to {}", directory)
    
    @classmethod
    def load(cls, directory: str | Path, config: GeneratorConfig) -> "ArtifactGenerator":
        """Load artifacts from a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
    
        generator = cls(config)
    
        # Load artifacts based on file extension
        for path in directory.iterdir():
            name = path.stem
            if path.suffix == ".npy":
                generator.artifacts[name] = np.load(path, allow_pickle=True)
            elif path.suffix == ".json":
                with open(path) as f:
                    generator.artifacts[name] = json.load(f)
    
        logger.info("Loaded artifacts from {}", directory)
        return generator


class DocumentGenerator:
    """
    Generates synthetic documents using the fixed artifacts.
    """

    def __init__(
        self,
        artifacts: Dict,
        doc_topic_alpha: float = 0.5,
        include_whitespace: bool = True,
        include_markers: bool = True,
        seed: int | None = None,
    ):
        """Initialize the document generator."""
        self.artifacts = artifacts
        self.doc_topic_alpha = doc_topic_alpha
        self.include_whitespace = include_whitespace
        self.include_markers = include_markers
        self._np_rng = np.random.RandomState(seed)

        # Special tokens
        self.BOD_TOKEN = -1  # Beginning of document
        self.EOD_TOKEN = -2  # End of document
        self.WS_TOKEN = -3  # Whitespace

        logger.info("Initialized DocumentGenerator")

    def generate(
        self, doc_length: int, return_entropy: bool = False
    ) -> Tuple[np.ndarray, float, float] | np.ndarray:
        """
        Generate a single document by:
        1. Sampling a topic mixture
        2. Selecting one topic
        3. Using that topic's transition matrix to generate a sequence
        4. Converting word indices to token sequences

        Args:
            doc_length: Number of words in the document
            return_entropy: If True, also return entropy measures

        Returns:
            If return_entropy is False:
                numpy array of tokens
            If return_entropy is True:
                tuple of (tokens, avg_entropy, perplexity)
        """
        # 1. Sample document-level topic mixture
        topic_mixture = self._np_rng.dirichlet(
            [self.doc_topic_alpha] * len(self.artifacts["topic_matrices"])
        )

        # 2. Select one topic
        topic_idx = self._np_rng.choice(len(topic_mixture), p=topic_mixture)
        T_topic = self.artifacts["topic_matrices"][topic_idx]
        stationary = self.artifacts["topic_distributions"][topic_idx]

        # 3. Generate sequence of word indices
        word_indices = []
        total_neg_log_prob = 0.0

        # Sample first word from stationary distribution
        current_word = self._np_rng.choice(len(stationary), p=stationary)
        word_indices.append(current_word)
        total_neg_log_prob += -np.log2(stationary[current_word] + 1e-12)

        # Generate remaining words using transition matrix
        for _ in range(doc_length - 1):
            row = T_topic[current_word]
            if row.sum() == 0:
                # Fallback to stationary if no transitions
                p_dist = stationary
            else:
                p_dist = row

            next_word = self._np_rng.choice(len(p_dist), p=p_dist)
            word_indices.append(next_word)
            total_neg_log_prob += -np.log2(p_dist[next_word] + 1e-12)
            current_word = next_word

        # 4. Convert word indices to token sequences
        tokens = []
        word_vocab = self.artifacts["word_vocab"]

        # Add beginning-of-document token
        if self.include_markers:
            tokens.append(self.BOD_TOKEN)

        # Add word tokens with optional whitespace
        for i, word_idx in enumerate(word_indices):
            tokens.extend(word_vocab[word_idx])
            if self.include_whitespace and i < len(word_indices) - 1:
                tokens.append(self.WS_TOKEN)

        # Add end-of-document token
        if self.include_markers:
            tokens.append(self.EOD_TOKEN)

        # Convert to numpy array
        doc_tensor = np.array(tokens, dtype=np.int64)

        if return_entropy:
            # Compute average entropy and perplexity
            avg_entropy = total_neg_log_prob / doc_length
            perplexity = 2**avg_entropy
            return doc_tensor, avg_entropy, perplexity

        return doc_tensor
