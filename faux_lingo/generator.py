"""
Constrained topic-based sequence generator with color-based transition rules.

This module provides functionality for generating sequences of tokens that follow:
1. Topic mixture constraints (controlling global token distributions)
2. Color-based transition rules (controlling local token transitions)

Tensor Dimensions Guide:
- batch: Number of sequences/matrices being generated simultaneously
- seq_len: Length of generated sequences
- vocab_size: Total number of possible tokens
- num_topics: Number of topics in the latent space
- num_colors: Number of color classes for tokens

Key Tensor Shapes:
- color_fractions: [num_colors]
- color_transitions: [num_colors, num_colors]
- topic_vectors: [num_topics, vocab_size]
- topic_mixtures: [batch, num_topics]
- sequences: [batch, seq_len]
- transition_matrices: [batch, vocab_size, vocab_size]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import tensorizer
import torch
from jaxtyping import Bool, Float, Int

# Type aliases for tensor dimensions
BatchDim = int
SeqLen = int
VocabSize = int
NumTopics = int
NumColors = int


def validate_shapes(
    tensors: Dict[str, torch.Tensor], expected_shapes: Dict[str, Tuple[int, ...]]
):
    """
    Validate tensor shapes against expected dimensions.

    Args:
        tensors: Dictionary of name -> tensor pairs
        expected_shapes: Dictionary of name -> expected shape pairs

    Raises:
        ValueError: If any tensor shape doesn't match expected shape
    """
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            actual = tensor.shape
            if actual != expected:
                raise ValueError(
                    f"Shape mismatch for {name}: " f"expected {expected}, got {actual}"
                )


@dataclass
class LanguageParams:
    """
    Container for parameters defining the synthetic language.

    Attributes:
        n_topics: Number of topics in latent space
        vocab_size: Total number of possible tokens
        color_fractions: Normalized fraction of vocabulary for each color
        color_transitions: Relative transition weights between colors
        topic_vectors: Orthonormal vectors defining topic distributions
    """

    n_topics: int
    vocab_size: int
    color_fractions: Float[torch.Tensor, "num_colors"]
    color_transitions: Float[torch.Tensor, "num_colors num_colors"]
    topic_vectors: Float[torch.Tensor, "num_topics vocab_size"]

    def __post_init__(self):
        """Validate tensor shapes and properties."""
        n_colors = len(self.color_fractions)
        validate_shapes(
            {
                "color_transitions": self.color_transitions,
                "topic_vectors": self.topic_vectors,
            },
            {
                "color_transitions": (n_colors, n_colors),
                "topic_vectors": (self.n_topics, self.vocab_size),
            },
        )

        # Validate properties
        if not torch.allclose(self.color_fractions.sum(), torch.tensor(1.0)):
            raise ValueError("Color fractions must sum to 1")
        if not torch.all(self.color_transitions >= 0):
            raise ValueError("Color transitions must be non-negative")

    def save(self, path: Union[str, Path]):
        """Save language parameters using torch.save."""
        data = {
            "n_topics": self.n_topics,
            "vocab_size": self.vocab_size,
            "color_fractions": self.color_fractions,
            "color_transitions": self.color_transitions,
            "topic_vectors": self.topic_vectors,
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cuda"):
        """Load language parameters using torch.load."""
        data = torch.load(path, map_location=device)
        return cls(
            n_topics=data["n_topics"],
            vocab_size=data["vocab_size"],
            color_fractions=data["color_fractions"],
            color_transitions=data["color_transitions"],
            topic_vectors=data["topic_vectors"],
        )


class ProbColorConstrainedGenerator:
    """
    Generator for sequences constrained by topics and color transitions.

    The generator creates sequences that respect both:
    1. Global distribution constraints via topic mixtures
    2. Local transition constraints via color-based rules

    Attributes:
        n_topics: Number of topics in latent space
        vocab_size: Total number of possible tokens
        n_colors: Number of color classes
        color_fractions: Normalized fraction of vocabulary for each color
        color_transitions: Weight matrix for transitions between colors
        topic_vectors: Orthonormal vectors defining topic distributions
        boundaries: Token index boundaries for each color
        transition_mask: Vocabulary-sized mask encoding allowed transitions
    """

    def __init__(
        self,
        n_topics: int,
        vocab_size: int,
        color_fractions: Union[torch.Tensor, List[float]],
        color_transitions: Float[torch.Tensor, "num_colors num_colors"],
        topic_vectors: Optional[Float[torch.Tensor, "num_topics vocab_size"]] = None,
        device: str = "cuda",
    ):
        """
        Initialize generator with language parameters.

        Args:
            n_topics: Number of topics in latent space
            vocab_size: Total number of possible tokens
            color_fractions: Relative sizes of color classes (will be normalized)
            color_transitions: Weight matrix for transitions between colors
            topic_vectors: Optional pre-defined topic vectors
            device: Compute device for tensors

        Raises:
            ValueError: If tensor shapes or properties are invalid
        """
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.device = device

        # Convert and normalize color fractions
        if isinstance(color_fractions, list):
            color_fractions = torch.tensor(color_fractions, device=device)
        self.color_fractions = color_fractions / color_fractions.sum()
        self.n_colors = len(self.color_fractions)

        # Validate shapes
        validate_shapes(
            {"color_transitions": color_transitions},
            {"color_transitions": (self.n_colors, self.n_colors)},
        )

        if not torch.all(color_transitions >= 0):
            raise ValueError("Color transitions must be non-negative")

        self.color_transitions = color_transitions.to(device)

        # Compute vocabulary ranges for each color
        counts = (self.color_fractions * self.vocab_size).long()
        self.boundaries = torch.zeros(
            self.n_colors + 1, dtype=torch.long, device=device
        )
        torch.cumsum(counts, dim=0, out=self.boundaries[1:])

        # Build vocabulary-sized transition mask
        self.transition_mask = self._build_block_mask()

        # Generate or load topic vectors
        if topic_vectors is not None:
            validate_shapes(
                {"topic_vectors": topic_vectors},
                {"topic_vectors": (self.n_topics, self.vocab_size)},
            )
            self.topic_vectors = topic_vectors.to(device)
        else:
            self.topic_vectors = self._init_topic_vectors()

    def _build_block_mask(self) -> Float[torch.Tensor, "vocab_size vocab_size"]:
        """
        Build vocabulary-sized mask from color transition matrix.

        Returns:
            Tensor containing transition weights expanded to vocabulary size
        """
        mask = torch.zeros((self.vocab_size, self.vocab_size), device=self.device)

        for i in range(self.n_colors):
            i_start, i_end = self.boundaries[i], self.boundaries[i + 1]
            for j in range(self.n_colors):
                j_start, j_end = self.boundaries[j], self.boundaries[j + 1]
                if self.color_transitions[i, j] > 0:
                    mask[i_start:i_end, j_start:j_end] = self.color_transitions[i, j]

        return mask

    def _init_topic_vectors(self) -> Float[torch.Tensor, "num_topics vocab_size"]:
        """
        Initialize random orthonormal topic vectors.

        Returns:
            Tensor of orthonormal vectors defining topic distributions
        """
        vectors = torch.randn(self.n_topics, self.vocab_size, device=self.device)
        Q, _ = torch.linalg.qr(vectors.T)
        return Q.T

    def generate_transitions(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        min_prob: float = 1e-6,
        mixtures: Optional[Float[torch.Tensor, "batch num_topics"]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch vocab_size vocab_size"],  # transition matrices
        Float[torch.Tensor, "batch num_topics"],  # topic mixtures
    ]:
        """
        Generate batch of transition matrices with probabilistic color constraints.

        Args:
            batch_size: Number of matrices to generate
            temperature: Controls sharpness of distributions
            min_prob: Minimum probability for valid transitions
            mixtures: Optional pre-specified topic mixtures

        Returns:
            Tuple of (transition_matrices, topic_mixtures) where:
            - transition_matrices: Batch of probability matrices for token transitions
            - topic_mixtures: Batch of mixture weights used to generate matrices
        """
        # Validate or generate mixtures
        if mixtures is not None:
            validate_shapes(
                {"mixtures": mixtures}, {"mixtures": (batch_size, self.n_topics)}
            )
        else:
            mixtures = torch.rand(batch_size, self.n_topics, device=self.device)
            mixtures = mixtures / mixtures.sum(dim=-1, keepdim=True)

        Lambda = torch.diag_embed(mixtures)

        # Generate base transition matrices
        Q = self.topic_vectors  # shape: [n_topics, vocab_size]

        # First multiply Lambda with Q for each batch
        # Lambda: [batch, n_topics, n_topics]
        # Q: [n_topics, vocab_size]
        # Result: [batch, n_topics, vocab_size]
        temp = torch.matmul(Lambda, Q)

        # Then multiply with Q.T for each batch
        # temp: [batch, n_topics, vocab_size]
        # Q.T: [vocab_size, n_topics]
        # Result: [batch, vocab_size, vocab_size]
        M = torch.matmul(temp, Q.T)

        if temperature != 1.0:
            M = M / temperature

        M = torch.relu(M)
        M = M * self.transition_mask.unsqueeze(0)
        M = torch.where(
            self.transition_mask > 0,
            torch.maximum(M, torch.tensor(min_prob, device=self.device)),
            M,
        )

        # Normalize
        M = M / M.sum(dim=-1, keepdim=True)

        return M, mixtures

    def sample_sequences(
        self,
        batch_size: int = 32,
        seq_length: int = 100,
        temperature: float = 1.0,
        start_color: Optional[int] = None,
        mixtures: Optional[Float[torch.Tensor, "batch num_topics"]] = None,
    ) -> Tuple[
        Int[torch.Tensor, "batch seq_len"],  # sequences
        Float[torch.Tensor, "batch num_topics"],  # topic mixtures
    ]:
        """
        Sample sequences and return with their topic mixtures.

        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            temperature: Controls randomness in token selection
            start_color: Optional color index to start sequences with
            mixtures: Optional pre-specified topic mixtures

        Returns:
            Tuple of (sequences, topic_mixtures) where:
            - sequences: Batch of token sequences
            - topic_mixtures: Topic mixture weights used to generate sequences
        """
        # Generate transition matrices and get mixtures
        trans_matrices, mixtures = self.generate_transitions(
            batch_size=batch_size, temperature=temperature, mixtures=mixtures
        )

        # Initialize sequences
        sequences = torch.zeros(
            (batch_size, seq_length), dtype=torch.long, device=self.device
        )

        # Sample initial tokens
        if start_color is not None:
            if not 0 <= start_color < self.n_colors:
                raise ValueError(f"Invalid start_color {start_color}")
            start, end = self.get_color_range(start_color)
            sequences[:, 0] = torch.randint(
                start, end, (batch_size,), device=self.device
            )
        else:
            sequences[:, 0] = torch.randint(
                0, self.vocab_size, (batch_size,), device=self.device
            )

        # Generate sequences
        for t in range(1, seq_length):
            current_probs = trans_matrices[
                torch.arange(batch_size, device=self.device), sequences[:, t - 1]
            ]
            sequences[:, t] = torch.multinomial(current_probs, 1).squeeze(-1)

        return sequences, mixtures

    def get_color_range(self, color_idx: int) -> Tuple[int, int]:
        """
        Get the vocabulary index range for a given color.

        Args:
            color_idx: Index of the color

        Returns:
            Tuple of (start_idx, end_idx) for the color's vocabulary range

        Raises:
            ValueError: If color_idx is invalid
        """
        if not 0 <= color_idx < self.n_colors:
            raise ValueError(f"Invalid color_idx {color_idx}")
        return (
            self.boundaries[color_idx].item(),
            self.boundaries[color_idx + 1].item(),
        )

    def get_color(self, vocab_idx: int) -> int:
        """
        Get color index for a vocabulary index.

        Args:
            vocab_idx: Index in vocabulary

        Returns:
            Index of the color that vocab_idx belongs to

        Raises:
            ValueError: If vocab_idx is invalid
        """
        if not 0 <= vocab_idx < self.vocab_size:
            raise ValueError(f"Invalid vocab_idx {vocab_idx}")
        return torch.searchsorted(self.boundaries, vocab_idx).item() - 1

    def save_language(self, path: Union[str, Path]):
        """Save the current language parameters."""
        params = LanguageParams(
            n_topics=self.n_topics,
            vocab_size=self.vocab_size,
            color_fractions=self.color_fractions,
            color_transitions=self.color_transitions,
            topic_vectors=self.topic_vectors,
        )
        params.save(path)

    @classmethod
    def load_language(cls, path: Union[str, Path], device: str = "cuda"):
        """Load a previously saved language."""
        params = LanguageParams.load(path, device)
        return cls(
            n_topics=params.n_topics,
            vocab_size=params.vocab_size,
            color_fractions=params.color_fractions,
            color_transitions=params.color_transitions,
            topic_vectors=params.topic_vectors,
            device=device,
        )


# Example usage and testing
if __name__ == "__main__":
    # Create with unnormalized color fractions
    color_fractions = [3, 5, 2]  # Will be normalized to [0.3, 0.5, 0.2]
    color_transitions = torch.tensor(
        [
            [1.0, 0.5, 0.1],  # Strong self-transitions, weaker cross-transitions
            [0.4, 1.0, 0.7],
            [0.2, 0.6, 1.0],
        ]
    )

    # Create generator with 10 topics
    generator = ProbColorConstrainedGenerator(
        n_topics=10,
        vocab_size=1000,
        color_fractions=color_fractions,
        color_transitions=color_transitions,
    )

    # Test shape validation
    try:
        bad_mixtures = torch.rand(32, 15)  # Wrong number of topics
        sequences, _ = generator.sample_sequences(mixtures=bad_mixtures)
    except ValueError as e:
        print("Successfully caught shape validation error:", e)

    # Generate sequences with specific topic mixture
    specific_mixture = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1] + [0.0] * 6]  # Focusing on first 4 topics
    ).repeat(
        32, 1
    )  # Batch size of 32

    sequences, mixtures = generator.sample_sequences(
        batch_size=32, seq_length=100, temperature=0.8, mixtures=specific_mixture
    )

    # Verify color transitions
    color_sequence = torch.tensor([generator.get_color(idx) for idx in sequences[0]])
    print("Sample color sequence:", color_sequence[:20].tolist())

    # Save and reload language
    generator.save_language("synthetic_language.tensors")
    loaded_generator = ProbColorConstrainedGenerator.load_language(
        "synthetic_language.tensors"
    )

    # Verify loaded generator produces same structure with same mixtures
    new_sequences, _ = loaded_generator.sample_sequences(
        batch_size=32,
        seq_length=100,
        temperature=0.8,
        mixtures=mixtures,  # Reuse previous mixtures
    )
