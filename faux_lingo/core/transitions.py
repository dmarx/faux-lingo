# faux_lingo/core/transitions.py
"""Transition probability matrices combining topic and color constraints."""

from typing import TypeAlias

import torch
from typing_extensions import Self

from .colors import ColorSpace
from .topics import TopicVectorSpace
from .vocabulary import Vocabulary

# Type aliases for dimensions
BatchDim: TypeAlias = int
VocabSize: TypeAlias = int


class TransitionMatrix:
    """
    Manages transition probability matrices that respect both topic and color constraints.

    Core properties:
    1. Matrices are proper probability distributions (row-wise sum to 1)
    2. Color transitions follow specified weights
    3. Global token distributions reflect topic mixtures
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        topic_space: TopicVectorSpace,
        color_space: ColorSpace,
        device: str | None = None,
    ):
        """
        Initialize transition matrix generator.

        Args:
            vocabulary: Vocabulary configuration and mappings
            topic_space: Space of topic vectors
            color_space: Color class definitions and rules
            device: Optional compute device, defaults to CPU

        Raises:
            ValueError: If spaces have incompatible dimensions
        """
        if topic_space.vocab_size != vocabulary.base_vocab_size:
            raise ValueError(
                f"Topic space vocab size ({topic_space.vocab_size}) "
                f"!= base vocab size ({vocabulary.base_vocab_size})"
            )
        if color_space.vocab_size != vocabulary.base_vocab_size:
            raise ValueError(
                f"Color space vocab size ({color_space.vocab_size}) "
                f"!= base vocab size ({vocabulary.base_vocab_size})"
            )

        self.device = device if device else "cpu"
        self.vocabulary = vocabulary
        self.topic_space = topic_space
        self.color_space = color_space

    def generate(
        self,
        topic_mixture: torch.Tensor,
        temperature: float = 1.0,
        min_prob: float = 1e-6,
    ) -> torch.Tensor:
        """
        Generate transition probability matrix for given topic mixture.

        Args:
            topic_mixture: Mixture weights for topics [batch_size, n_topics]
            temperature: Controls entropy of distributions (higher = more uniform)
            min_prob: Minimum probability for valid transitions

        Returns:
            Transition probability matrix [batch_size, vocab_size, vocab_size]

        Notes:
            1. Output[b,i,j] = P(token_j | token_i) for sequence b
            2. Each row sums to 1 (is a valid probability distribution)
            3. Respects both topic and color constraints
        """
        # Get base distributions from topics
        base_probs = self.topic_space.get_distribution(topic_mixture)

        # Convert to transition matrix
        # Each row i is the topic distribution masked by valid transitions from token i
        # Expand base probabilities to transition matrix shape
        base_size = self.vocabulary.base_vocab_size
        transitions = base_probs.unsqueeze(1).expand(-1, base_size, -1)

        # Apply color mask to enforce transition constraints
        color_mask = self.color_space.get_transition_mask()
        transitions = transitions * color_mask

        # Set minimum probability for valid transitions
        transitions = torch.where(
            color_mask > 0,
            torch.maximum(transitions, torch.tensor(min_prob, device=self.device)),
            transitions,
        )

        # Apply temperature scaling to logits before normalization
        if temperature != 1.0:
            transitions = transitions / temperature

        # Normalize each row to get proper probability distributions
        # Small epsilon to avoid division by zero
        row_sums = transitions.sum(dim=-1, keepdim=True) + 1e-10
        transitions = transitions / row_sums

        return transitions

    @classmethod
    def create_uniform(
        cls,
        vocabulary: Vocabulary,
        n_topics: int,
        color_fractions: list[float],
        device: str | None = None,
    ) -> Self:
        """
        Create transition matrix with uniform topic vectors and color transitions.

        Args:
            vocabulary: Vocabulary configuration
            n_topics: Number of topics to use
            color_fractions: Relative sizes of color classes
            device: Optional compute device

        Returns:
            TransitionMatrix instance with uniform parameters
        """
        topic_space = TopicVectorSpace(
            n_topics=n_topics,
            vocab_size=vocabulary.base_vocab_size,
            device=device,
        )
        color_space = ColorSpace(
            color_fractions=color_fractions,
            vocab_size=vocabulary.base_vocab_size,
            device=device,
        )
        return cls(vocabulary, topic_space, color_space, device=device)

    def save(self, path: str) -> None:
        """Save transition parameters."""
        raise NotImplementedError("Saving not yet implemented")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> Self:
        """Load transition parameters."""
        raise NotImplementedError("Loading not yet implemented")
