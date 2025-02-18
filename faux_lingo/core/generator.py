# faux_lingo/core/generator.py
"""Sequence generator with constrained topic and color structure."""

from dataclasses import dataclass
from typing import TypeAlias

import torch
from typing_extensions import Self

from .transitions import TransitionMatrix

# Type aliases for dimensions
BatchDim: TypeAlias = int
SeqLen: TypeAlias = int


@dataclass
class GeneratedSequences:
    """Container for generated sequences and their properties.

    Attributes:
        tokens: Generated token sequences [batch_size, seq_len]
        topic_mixtures: Topic mixtures used for generation [batch_size, n_topics]
        log_probs: Log probabilities of generated sequences [batch_size]
    """

    tokens: torch.Tensor
    topic_mixtures: torch.Tensor
    log_probs: torch.Tensor


class SequenceGenerator:
    """
    Generates sequences using topic and color-constrained transitions.

    Core functionality:
    1. Sampling sequences from transition matrices
    2. Computing sequence probabilities
    3. Generating with specific topic mixtures or color constraints
    """

    def __init__(
        self,
        transition_model: TransitionMatrix,
        device: str | None = None,
    ):
        """
        Initialize sequence generator.

        Args:
            transition_model: Model for generating transition matrices
            device: Optional compute device, defaults to CPU
        """
        self.device = device if device else "cpu"
        self.transition_model = transition_model
        self.vocab_size = transition_model.vocab_size

    def generate(
        self,
        batch_size: int,
        seq_length: int,
        temperature: float = 1.0,
        topic_mixtures: torch.Tensor | None = None,
        start_tokens: torch.Tensor | None = None,
        min_prob: float = 1e-6,
    ) -> GeneratedSequences:
        """
        Generate batch of sequences.

        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            temperature: Controls randomness in sampling
            topic_mixtures: Optional pre-specified topic mixtures [batch_size, n_topics]
            start_tokens: Optional initial tokens [batch_size]
            min_prob: Minimum probability for valid transitions

        Returns:
            GeneratedSequences containing tokens and properties

        Notes:
            If topic_mixtures not provided, samples from uniform distribution
            If start_tokens not provided, samples initial tokens uniformly
        """
        # Get or generate topic mixtures
        if topic_mixtures is None:
            n_topics = self.transition_model.topic_space.n_topics
            topic_mixtures = torch.ones(batch_size, n_topics, device=self.device)
            topic_mixtures = topic_mixtures / n_topics

        # Validate topic mixture shape
        if topic_mixtures.shape[0] != batch_size:
            raise ValueError(
                f"Topic mixture batch size {topic_mixtures.shape[0]} "
                f"!= requested batch size {batch_size}"
            )

        # Generate transition matrix
        transitions = self.transition_model.generate(
            topic_mixtures,
            temperature=temperature,
            min_prob=min_prob,
        )

        # Initialize sequences
        sequences = torch.zeros(
            (batch_size, seq_length), dtype=torch.long, device=self.device
        )

        # Initialize log probabilities
        log_probs = torch.zeros(batch_size, device=self.device)

        # Sample or use provided start tokens
        if start_tokens is not None:
            if start_tokens.shape != (batch_size,):
                raise ValueError(
                    f"Start tokens shape {start_tokens.shape} "
                    f"!= (batch_size={batch_size},)"
                )
            sequences[:, 0] = start_tokens
        else:
            sequences[:, 0] = torch.randint(
                0, self.vocab_size, (batch_size,), device=self.device
            )

        # Generate rest of sequences
        for t in range(1, seq_length):
            # Get transition probabilities for current tokens
            current_probs = transitions[
                torch.arange(batch_size, device=self.device),
                sequences[:, t - 1],
            ]

            # Sample next tokens
            next_tokens = torch.multinomial(current_probs, 1).squeeze(-1)
            sequences[:, t] = next_tokens

            # Update log probabilities
            log_probs += torch.log(
                torch.gather(
                    current_probs,
                    1,
                    next_tokens.unsqueeze(1),
                )
            ).squeeze(-1)

        return GeneratedSequences(
            tokens=sequences,
            topic_mixtures=topic_mixtures,
            log_probs=log_probs,
        )

    def generate_with_color(
        self,
        batch_size: int,
        seq_length: int,
        start_color: int,
        temperature: float = 1.0,
        topic_mixtures: torch.Tensor | None = None,
    ) -> GeneratedSequences:
        """
        Generate sequences starting with tokens of a specific color.

        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            start_color: Color index to start sequences with
            temperature: Controls randomness in sampling
            topic_mixtures: Optional pre-specified topic mixtures

        Returns:
            GeneratedSequences with tokens starting from specified color
        """
        # Get token range for start color
        start_idx, end_idx = self.transition_model.color_space.get_color_range(
            start_color
        )

        # Sample start tokens from color range
        start_tokens = torch.randint(
            start_idx, end_idx, (batch_size,), device=self.device
        )

        return self.generate(
            batch_size=batch_size,
            seq_length=seq_length,
            temperature=temperature,
            topic_mixtures=topic_mixtures,
            start_tokens=start_tokens,
        )

    @classmethod
    def create_uniform(
        cls,
        vocab_size: int,
        n_topics: int,
        color_fractions: list[float],
        device: str | None = None,
    ) -> Self:
        """
        Create generator with uniform topic and color distributions.

        Args:
            vocab_size: Size of token vocabulary
            n_topics: Number of topics
            color_fractions: Relative sizes of color classes
            device: Optional compute device

        Returns:
            SequenceGenerator with uniform parameters
        """
        transition_model = TransitionMatrix.create_uniform(
            vocab_size=vocab_size,
            n_topics=n_topics,
            color_fractions=color_fractions,
            device=device,
        )
        return cls(transition_model, device=device)
