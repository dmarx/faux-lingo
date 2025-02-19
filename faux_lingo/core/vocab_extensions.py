# faux_lingo/core/vocab_extensions.py
"""Extensions to vocabulary system for multiple mappings and augmentations."""

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import torch

from .vocab_mapping import TokenIdx, TokenSeq, VocabHierarchy

# Type aliases
Probability: TypeAlias = float
AugmentedSeq: TypeAlias = tuple[TokenSeq, Probability]


@dataclass
class MultiMappingLevel:
    """Vocabulary level supporting multiple mappings.

    Attributes:
        vocab_size: Number of tokens at this level
        chunk_size: Number of tokens from parent level per token
        sequences: Mapping of token to list of possible sequences with probabilities
    """

    vocab_size: int
    chunk_size: int
    sequences: dict[TokenIdx, list[AugmentedSeq]]

    def __post_init__(self):
        """Validate level properties."""
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if len(self.sequences) != self.vocab_size:
            raise ValueError(
                f"Number of sequences ({len(self.sequences)}) "
                f"!= vocab_size ({self.vocab_size})"
            )

        # Validate sequence probabilities
        for token, seqs in self.sequences.items():
            if not seqs:
                raise ValueError(f"No sequences defined for token {token}")
            probs = [prob for _, prob in seqs]
            if not torch.allclose(torch.tensor(sum(probs)), torch.tensor(1.0)):
                raise ValueError(
                    f"Sequence probabilities for token {token} do not sum to 1"
                )


class MultiMappingHierarchy:
    """Hierarchical vocabulary with multiple possible mappings.

    Core functionality:
    1. Support for multiple sequences mapping to same token
    2. Probabilistic sequence selection during decoding
    3. Integration with existing vocabulary system
    """

    def __init__(
        self,
        levels: Sequence[MultiMappingLevel],
        device: str | None = None,
    ):
        """Initialize hierarchy with multiple mapping levels.

        Args:
            levels: Sequence of vocabulary levels from lowest to highest
            device: Optional compute device, defaults to CPU
        """
        self.device = device if device else "cpu"
        self.levels = list(levels)

    def decode_sequence(
        self,
        tokens: torch.Tensor,
        start_level: int,
        target_level: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Decode token sequence with probabilistic mapping selection.

        Args:
            tokens: Input token sequence [batch_size, seq_len]
            start_level: Index of starting vocabulary level
            target_level: Index of target vocabulary level
            seed: Optional random seed for reproducible decoding

        Returns:
            Decoded token sequences at target level [batch_size, new_seq_len]
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not (0 <= start_level < len(self.levels)):
            raise ValueError(f"Invalid start_level: {start_level}")
        if not (0 <= target_level < len(self.levels)):
            raise ValueError(f"Invalid target_level: {target_level}")
        if target_level > start_level:
            raise ValueError("Can only decode to same or lower levels")

        # Return input tokens if no decoding needed
        if target_level == start_level:
            return tokens

        # Start with input tokens
        current = tokens

        # Decode through intermediate levels
        for level in range(start_level - 1, target_level - 1, -1):
            # Get all possible sequences for each token
            level_seqs = self.levels[level].sequences
            max_seq_len = max(
                len(seq) for seqs in level_seqs.values() for seq, _ in seqs
            )

            # Create output tensor with padding
            result = torch.full(
                (
                    current.shape[0],
                    current.shape[1] * max_seq_len,
                ),
                -1,
                dtype=torch.long,
                device=self.device,
            )

            # Process each token in batch
            for b in range(current.shape[0]):
                pos = 0
                for t in range(current.shape[1]):
                    # Ensure integer token index
                    token_idx = int(current[b, t].item())
                    if token_idx == -1:  # Skip padding
                        continue

                    # Get possible sequences and probabilities
                    seqs = level_seqs[token_idx]
                    probs = torch.tensor([p for _, p in seqs], device=self.device)

                    # Sample sequence based on probabilities and convert to int
                    seq_idx = int(torch.multinomial(probs, 1).item())
                    seq, _ = seqs[seq_idx]

                    # Add sequence to result
                    result[b, pos : pos + len(seq)] = torch.tensor(
                        seq, device=self.device
                    )
                    pos += len(seq)

            current = result

        return current


@dataclass
class AugmentationConfig:
    """Configuration for sequence augmentation.

    Attributes:
        deletion_prob: Probability of character deletion
        insertion_prob: Probability of random character insertion
        substitution_prob: Probability of character substitution
        transposition_prob: Probability of adjacent character transposition
        seed: Optional random seed for reproducibility
    """

    deletion_prob: float = 0.05
    insertion_prob: float = 0.05
    substitution_prob: float = 0.05
    transposition_prob: float = 0.05
    seed: int | None = None

    def __post_init__(self):
        """Validate configuration."""
        probs = [
            self.deletion_prob,
            self.insertion_prob,
            self.substitution_prob,
            self.transposition_prob,
        ]
        if any(p < 0 or p > 1 for p in probs):
            raise ValueError("All probabilities must be between 0 and 1")
        if sum(probs) > 1:
            raise ValueError("Sum of probabilities must not exceed 1")


class SequenceAugmenter:
    """Applies random perturbations to token sequences.

    Core functionality:
    1. Character-level augmentations (deletion, insertion, etc.)
    2. Controlled randomization based on probabilities
    3. Vocabulary-aware modifications
    """

    def __init__(
        self,
        vocab_size: int,
        config: AugmentationConfig,
        device: str | None = None,
    ):
        """Initialize augmenter with vocabulary and configuration.

        Args:
            vocab_size: Size of token vocabulary
            config: Augmentation parameters
            device: Optional compute device, defaults to CPU
        """
        self.vocab_size = vocab_size
        self.config = config
        self.device = device if device else "cpu"

        if config.seed is not None:
            torch.manual_seed(config.seed)

    def augment_sequence(self, sequence: TokenSeq) -> TokenSeq:
        """Apply random augmentations to token sequence.

        Args:
            sequence: Input token sequence

        Returns:
            Augmented token sequence
        """
        seq = list(sequence)

        # Apply augmentations in random order
        ops = [
            (self._delete, self.config.deletion_prob),
            (self._insert, self.config.insertion_prob),
            (self._substitute, self.config.substitution_prob),
            (self._transpose, self.config.transposition_prob),
        ]

        for op, prob in ops:
            if torch.rand(1).item() < prob:
                seq = op(seq)

        return tuple(seq)

    def _delete(self, seq: list[int]) -> list[int]:
        """Randomly delete a token."""
        if len(seq) <= 1:
            return seq
        idx = torch.randint(len(seq), (1,)).item()
        return seq[:idx] + seq[idx + 1 :]

    def _insert(self, seq: list[int]) -> list[int]:
        """Insert random token."""
        idx = torch.randint(len(seq) + 1, (1,)).item()
        token = torch.randint(self.vocab_size, (1,)).item()
        return seq[:idx] + [token] + seq[idx:]

    def _substitute(self, seq: list[int]) -> list[int]:
        """Replace token with a different random token."""
        if not seq:
            return seq
        idx = torch.randint(len(seq), (1,)).item()
        current = seq[idx]
        # Generate new token until it's different from current
        while True:
            token = torch.randint(self.vocab_size, (1,)).item()
            if token != current:
                break
        seq[idx] = token
        return seq

    def _transpose(self, seq: list[int]) -> list[int]:
        """Swap adjacent tokens."""
        if len(seq) <= 1:
            return seq
        idx = torch.randint(len(seq) - 1, (1,)).item()
        seq[idx], seq[idx + 1] = seq[idx + 1], seq[idx]
        return seq


def convert_to_multi_mapping(
    hierarchy: VocabHierarchy,
    augmenter: SequenceAugmenter | None = None,
    n_variants: int = 3,
) -> MultiMappingHierarchy:
    """Convert standard hierarchy to multi-mapping hierarchy.

    Args:
        hierarchy: Standard vocabulary hierarchy
        augmenter: Optional sequence augmenter for variants
        n_variants: Number of variants to generate per sequence

    Returns:
        MultiMappingHierarchy with original and variant sequences
    """
    levels = []

    for level in hierarchy:
        multi_sequences: dict[TokenIdx, list[AugmentedSeq]] = {}

        for token, base_seq in level.sequences.items():
            variants: list[AugmentedSeq] = [
                (base_seq, 0.6)
            ]  # Original sequence gets higher weight

            if augmenter:
                # Generate variants with augmentation
                n_aug = min(n_variants - 1, 5)  # Cap number of variants
                prob_per_variant = (1.0 - 0.6) / n_aug

                for _ in range(n_aug):
                    variant = augmenter.augment_sequence(base_seq)
                    variants.append((variant, prob_per_variant))

            multi_sequences[token] = variants

        multi_level = MultiMappingLevel(
            vocab_size=level.vocab_size,
            chunk_size=level.chunk_size,
            sequences=multi_sequences,
        )
        levels.append(multi_level)

    return MultiMappingHierarchy(levels)
