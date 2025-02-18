# faux_lingo/analysis/entropy.py
"""Information-theoretic analysis of generated sequences."""

from dataclasses import dataclass
from typing import TypeAlias

import torch
from typing_extensions import Self

from ..core.generator import GeneratedSequences
from ..core.transitions import TransitionMatrix

# Type aliases for dimensions
BatchDim: TypeAlias = int
SeqLen: TypeAlias = int
NumTopics: TypeAlias = int


@dataclass
class EntropyMetrics:
    """Container for sequence entropy measurements.

    Attributes:
        color_entropy: Empirical entropy of color transitions
        topic_entropy: Entropy of topic mixtures used in generation
        token_entropy: Empirical entropy of generated token sequences
    """

    color_entropy: float
    topic_entropy: float
    token_entropy: float

    @classmethod
    def zero(cls) -> Self:
        """Create EntropyMetrics initialized to zero."""
        return cls(
            color_entropy=0.0,
            topic_entropy=0.0,
            token_entropy=0.0,
        )


class EntropyAnalyzer:
    """Analyzer for information-theoretic properties of sequences.

    Core functionality:
    1. Computing empirical entropy of generated sequences
    2. Analyzing topic mixture entropy
    3. Tracking color transition patterns
    """

    def __init__(self, transition_model: TransitionMatrix):
        """Initialize analyzer with transition model."""
        self.transition_model = transition_model
        self.device = transition_model.device

    def analyze_sequences(
        self,
        sequences: GeneratedSequences,
    ) -> EntropyMetrics:
        """Compute entropy metrics for sequences.

        Args:
            sequences: Generated token sequences and properties

        Returns:
            EntropyMetrics containing various entropy measures
        """
        metrics = EntropyMetrics(
            color_entropy=self._compute_color_entropy(sequences.tokens),
            topic_entropy=self._compute_topic_entropy(sequences.topic_mixtures),
            token_entropy=self._compute_token_entropy(sequences.tokens),
        )

        return metrics

    def _compute_color_entropy(self, tokens: torch.Tensor) -> float:
        """Compute empirical entropy of color transitions.

        Args:
            tokens: Generated token sequences [batch, seq_len]

        Returns:
            Estimated color transition entropy
        """
        # Convert tokens to colors
        colors = torch.tensor(
            [
                [self.transition_model.color_space.get_color(idx.item()) for idx in seq]
                for seq in tokens
            ],
            device=self.device,
            dtype=torch.long,
        )

        # Count color transitions
        n_colors = self.transition_model.color_space.n_colors
        counts = torch.zeros((n_colors, n_colors), device=self.device)

        for b in range(len(tokens)):
            for t in range(len(tokens[0]) - 1):
                curr_color = colors[b, t]
                next_color = colors[b, t + 1]
                counts[curr_color, next_color] += 1

        # Convert to probabilities with row-wise normalization
        # Add small epsilon to avoid division by zero
        row_sums = counts.sum(dim=1, keepdim=True) + 1e-10
        P = counts / row_sums

        # Compute entropy per row and average
        H = -torch.sum(P * torch.log2(P + 1e-10), dim=1).mean()

        return H.item()

    def _compute_topic_entropy(self, mixtures: torch.Tensor) -> float:
        """Compute entropy of topic mixtures.

        Args:
            mixtures: Topic mixture weights [batch, n_topics]

        Returns:
            Entropy of average topic distribution
        """
        # Average topic distribution across batch
        P = mixtures.mean(0)
        H = -torch.sum(P * torch.log2(P + 1e-10))

        return H.item()

    def _compute_token_entropy(self, tokens: torch.Tensor) -> float:
        """Compute empirical entropy of token sequences.

        Args:
            tokens: Generated token sequences [batch, seq_len]

        Returns:
            Estimated token entropy
        """
        # Count token frequencies
        counts = torch.zeros(self.transition_model.vocab_size, device=self.device)

        for seq in tokens.long():  # Ensure long dtype for indexing
            unique, seq_counts = torch.unique(seq, return_counts=True)
            counts[unique] += seq_counts

        # Convert to probabilities and compute entropy
        P = counts / counts.sum()
        H = -torch.sum(P * torch.log2(P + 1e-10))

        return H.item()
