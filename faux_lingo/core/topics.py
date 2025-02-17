# faux_lingo/core/topics.py

"""
Topic modeling system for synthetic language generation using preferential attachment.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from loguru import logger


@dataclass
class TopicConfig:
    """Configuration for topic model generation."""

    num_topics: int
    modes_per_color: int
    attachment_bias: float

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_topics < 1:
            raise ValueError("num_topics must be positive")
        if self.modes_per_color < 1:
            raise ValueError("modes_per_color must be positive")
        if self.attachment_bias < 0:
            raise ValueError("attachment_bias must be non-negative")


class TopicModel:
    """
    Manages topic-specific transition matrices using preferential attachment
    to topic mode words.
    """

    def __init__(
        self,
        config: TopicConfig,
        word_colors: Dict[int, int],
        base_matrix: np.ndarray,
        seed: int | None = None,
    ):
        """Initialize the topic model system."""
        self.config = config
        self.config.validate()
        self.word_colors = word_colors
        self.base_matrix = base_matrix
        self._np_rng = np.random.RandomState(seed)

        # Get vocabulary size and number of colors from inputs
        self.vocab_size = base_matrix.shape[0]
        self.num_colors = max(word_colors.values()) + 1

        # Initialize containers
        self.topic_modes: List[Dict[int, Set[int]]] = []
        self.topic_matrices: List[np.ndarray] = []
        self.topic_distributions: List[np.ndarray] = []

        # Build color -> words mapping for efficiency
        self._color_to_words: Dict[int, List[int]] = {}
        for word, color in word_colors.items():
            self._color_to_words.setdefault(color, []).append(word)

        logger.info("Initialized TopicModel with config: {}", config)

    def build(self) -> dict:
        """Build the complete topic model and return relevant artifacts."""
        self.sample_topic_modes()
        self.build_topic_matrices()
        self.compute_topic_distributions()

        return {
            "topic_modes": self.topic_modes,
            "topic_matrices": self.topic_matrices,
            "topic_distributions": self.topic_distributions,
        }

    def sample_topic_modes(self) -> None:
        """
        For each topic and color, sample a set of mode words that will have
        enhanced transition probabilities.
        """
        self.topic_modes = []

        for _ in range(self.config.num_topics):
            mode_dict: Dict[int, Set[int]] = {}

            for color in range(self.num_colors):
                words_in_color = self._color_to_words.get(color, [])

                if len(words_in_color) < self.config.modes_per_color:
                    raise ValueError(
                        f"Not enough words with color {color} to sample "
                        f"{self.config.modes_per_color} modes"
                    )

                # Sample mode words for this topic and color
                mode_words = set(
                    self._np_rng.choice(
                        words_in_color, size=self.config.modes_per_color, replace=False
                    )
                )
                mode_dict[color] = mode_words

            self.topic_modes.append(mode_dict)

        logger.debug("Sampled mode words for {} topics", len(self.topic_modes))
        
    def build_topic_matrices(self) -> None:
        """
        Generate topic-specific transition matrices by applying small random
        perturbations to the base matrix while preserving sparsity patterns.
        """
        if not self.topic_modes:
            raise RuntimeError("Topic modes must be sampled first")
    
        self.topic_matrices = []
        epsilon = self.config.attachment_bias  # Use attachment_bias as perturbation scale
    
        for topic_idx, mode_dict in enumerate(self.topic_modes):
            # Get sparsity pattern from base matrix
            nonzero_mask = self.base_matrix > 0
    
            # Generate random perturbation matrix with same sparsity
            perturbation = self._np_rng.rand(*self.base_matrix.shape)
            perturbation = perturbation * nonzero_mask  # Apply sparsity mask
            perturbation = perturbation / perturbation.sum(axis=1, keepdims=True)
            perturbation = np.nan_to_num(perturbation)  # Handle zero rows
    
            # Combine base matrix with perturbation
            topic_matrix = (1 - epsilon) * self.base_matrix + epsilon * perturbation
    
            # Ensure it's still row-stochastic
            row_sums = topic_matrix.sum(axis=1, keepdims=True)
            mask = row_sums > 0
            topic_matrix[mask] = topic_matrix[mask] / row_sums[mask]
    
            self.topic_matrices.append(topic_matrix)
    
        logger.debug(
            "Built transition matrices for {} topics", len(self.topic_matrices)
        )

    def compute_topic_distributions(self) -> None:
        """
        Compute the steady state distribution for each topic-specific
        transition matrix.
        """
        if not self.topic_matrices:
            raise RuntimeError("Topic matrices must be built first")

        self.topic_distributions = []

        for topic_idx, matrix in enumerate(self.topic_matrices):
            # Compute steady state using power iteration
            n = matrix.shape[0]
            pi = np.ones(n) / n  # Initial uniform distribution

            for _ in range(1000):  # Max iterations
                pi_next = pi @ matrix
                if np.linalg.norm(pi_next - pi, 1) < 1e-8:  # Convergence threshold
                    break
                pi = pi_next

            self.topic_distributions.append(pi)

        logger.debug(
            "Computed steady state distributions for {} topics",
            len(self.topic_distributions),
        )

    def get_topic_entropy(self, topic_idx: int) -> Tuple[float, float]:
        """
        Compute entropy measures for a specific topic:
        - Stationary entropy: entropy of the steady state distribution
        - Conditional entropy: average entropy of transitions

        Returns:
            Tuple of (stationary_entropy, conditional_entropy) in bits
        """
        if not self.topic_distributions or not self.topic_matrices:
            raise RuntimeError("Topic model must be built first")

        if not 0 <= topic_idx < self.config.num_topics:
            raise ValueError(f"Invalid topic index: {topic_idx}")

        # Compute stationary entropy
        pi = self.topic_distributions[topic_idx]
        stationary_entropy = -np.sum(pi[pi > 0] * np.log2(pi[pi > 0]))

        # Compute conditional entropy
        matrix = self.topic_matrices[topic_idx]
        row_entropies = np.zeros(self.vocab_size)

        for i in range(self.vocab_size):
            row = matrix[i]
            nonzero = row[row > 0]
            if len(nonzero) > 0:
                row_entropies[i] = -np.sum(nonzero * np.log2(nonzero))

        conditional_entropy = np.dot(pi, row_entropies)

        return stationary_entropy, conditional_entropy
