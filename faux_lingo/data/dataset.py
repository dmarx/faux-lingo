# faux_lingo/data/dataset.py
"""Dataset management for sequence generation."""

from dataclasses import dataclass
from typing import Iterator, TypeAlias, TypedDict

import torch

from ..core.generator import GeneratedSequences, SequenceGenerator

# Type aliases for dimensions
BatchDim: TypeAlias = int
SeqLen: TypeAlias = int

class BatchStats(TypedDict):
    mean_log_prob: float
    topic_weights: list[float]
    color_counts: list[int]


@dataclass
class DatasetConfig:
    """Configuration for dataset generation.

    Attributes:
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence
        n_batches: Total number of batches to generate
        temperature: Controls randomness in generation
        seed: Random seed for reproducibility
    """

    batch_size: int
    seq_length: int
    n_batches: int
    temperature: float = 1.0
    seed: int | None = None


class SequenceDataset:
    """Manages generation and iteration of sequence batches.

    Core functionality:
    1. Batch generation with consistent configuration
    2. Tracking of sequence properties and metadata
    3. Iterator interface for training/validation
    """

    def __init__(
        self,
        generator: SequenceGenerator,
        config: DatasetConfig,
    ):
        """Initialize dataset with generator and configuration.

        Args:
            generator: Sequence generator instance
            config: Dataset generation parameters
        """
        self.generator = generator
        self.config = config
        self.device = generator.device

        # Set random seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Initialize dataset properties
        self.total_sequences = config.batch_size * config.n_batches
        self._current_batch = 0
        self._cached_batch: GeneratedSequences | None = None

    def __len__(self) -> int:
        """Get total number of batches."""
        return self.config.n_batches

    def __iter__(self) -> Iterator[GeneratedSequences]:
        """Create iterator over sequence batches."""
        return self

    def __next__(self) -> GeneratedSequences:
        """Get next batch of sequences."""
        if self._current_batch >= self.config.n_batches:
            self._current_batch = 0
            raise StopIteration

        self._current_batch += 1
        return self.generate_batch()

    def generate_batch(
        self,
        topic_mixtures: torch.Tensor | None = None,
        start_color: int | None = None,
    ) -> GeneratedSequences:
        """Generate a single batch of sequences.

        Args:
            topic_mixtures: Optional pre-specified topic mixtures
            start_color: Optional color index to start sequences with

        Returns:
            GeneratedSequences containing tokens and properties
        """
        if start_color is not None:
            sequences = self.generator.generate_with_color(
                batch_size=self.config.batch_size,
                seq_length=self.config.seq_length,
                start_color=start_color,
                temperature=self.config.temperature,
                topic_mixtures=topic_mixtures,
            )
        else:
            sequences = self.generator.generate(
                batch_size=self.config.batch_size,
                seq_length=self.config.seq_length,
                temperature=self.config.temperature,
                topic_mixtures=topic_mixtures,
            )

        self._cached_batch = sequences
        return sequences

    def get_color_sequences(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert token sequences to color sequences.

        Args:
            tokens: Token sequences [batch_size, seq_length]

        Returns:
            Color sequences [batch_size, seq_length]
        """
        return torch.tensor(
            [
                [
                    self.generator.transition_model.color_space.get_color(idx.item())
                    for idx in seq
                ]
                for seq in tokens
            ],
            device=self.device,
            dtype=torch.long,
        )

    def get_batch_stats(self, batch: GeneratedSequences) -> BatchStats:
        """Compute statistics for a batch of sequences.

        Args:
            batch: Batch of generated sequences

        Returns:
            Dictionary of batch statistics
        """
        color_seqs = self.get_color_sequences(batch.tokens)

        stats: BatchStats = {
            "mean_log_prob": batch.log_probs.mean().item(),
            "topic_weights": batch.topic_mixtures.mean(0).tolist(),
            "color_counts": torch.bincount(
                color_seqs.view(-1),
                minlength=self.generator.transition_model.color_space.n_colors,
            ).tolist(),
        }

        return stats

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size of generator."""
        return self.generator.vocab_size

    @property
    def n_topics(self) -> int:
        """Get number of topics in generator."""
        return self.generator.transition_model.topic_space.n_topics

    @property
    def n_colors(self) -> int:
        """Get number of color classes in generator."""
        return self.generator.transition_model.color_space.n_colors
