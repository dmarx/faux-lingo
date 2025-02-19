# faux_lingo/core/vocab_mapping.py
"""Hierarchical vocabulary mapping and decoding."""

from dataclasses import dataclass
from typing import Iterator, Sequence, TypeAlias

import torch
from typing_extensions import Self

# Type aliases for clarity
TokenIdx: TypeAlias = int
TokenSeq: TypeAlias = tuple[int, ...]
Shape: TypeAlias = tuple[int, ...]


@dataclass
class VocabLevel:
    """
    A single level in the vocabulary hierarchy.
    
    Attributes:
        vocab_size: Number of tokens at this level
        chunk_size: Number of tokens from parent level per token
        sequences: Mapping of each token to its constituent sequence
    """

    vocab_size: int
    chunk_size: int
    sequences: dict[TokenIdx, TokenSeq]

    def __post_init__(self) -> None:
        """Validate vocabulary level properties."""
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if len(self.sequences) != self.vocab_size:
            raise ValueError(
                f"Number of sequences ({len(self.sequences)}) "
                f"!= vocab_size ({self.vocab_size})"
            )
        # Validate all sequences are proper tuples of integers
        for token, seq in self.sequences.items():
            if not isinstance(seq, tuple):
                raise ValueError(f"Sequence for token {token} must be a tuple")
            if not all(isinstance(t, int) for t in seq):
                raise ValueError(
                    f"All elements in sequence for token {token} must be integers"
                )

    @property
    def max_sequence_length(self) -> int:
        """Maximum length of any sequence in this level."""
        return max(len(seq) for seq in self.sequences.values())


class VocabHierarchy:
    """
    Manages hierarchical relationships between vocabulary levels.

    Note: VocabLevels represent mappings BETWEEN levels, not the levels themselves.
    With n VocabLevels, we actually have n+1 vocabulary levels total.
    Level indexing goes from most abstract (0) to most concrete (n):
    Level 0 -> Level 1 (Mapping A)
    Level 1 -> Level 2 (Mapping B)
    """

    def __init__(
        self,
        levels: Sequence[VocabLevel],
        device: str | None = None,
    ) -> None:
        """Initialize vocabulary hierarchy.

        Args:
            levels: Sequence of vocabulary mappings from highest to lowest abstraction
            device: Optional compute device, defaults to CPU
        """
        self.device = device if device else "cpu"
        self.levels = list(levels)
        self.num_levels = len(self.levels) + 1
        self.decode_tables = self._build_decode_tables()
        
        # Calculate total expansion ratio
        self.expansion_ratio = 1
        for level in self.levels:
            self.expansion_ratio *= level.chunk_size

    def compute_decoded_length(self, latent_length: int) -> int:
        """
        Compute the length of decoded sequences for a given latent length.

        Args:
            latent_length: Length of input sequence before decoding

        Returns:
            Expected length of decoded sequence
        """
        return latent_length * self.expansion_ratio

    def compute_latent_length(self, decoded_length: int) -> int:
        """
        Compute required latent sequence length to achieve desired output length.

        Args:
            decoded_length: Desired length after decoding

        Returns:
            Required length of input sequence

        Notes:
            Returns max(1, decoded_length // expansion_ratio) to ensure
            at least one token is generated.
        """
        return max(1, decoded_length // self.expansion_ratio)

    def decode_sequence(
        self,
        tokens: torch.Tensor,
        start_level: int | None = None,
        target_level: int | None = None,
    ) -> torch.Tensor:
        """Decode token sequence from one level to another.

        Args:
            tokens: Input token sequence [batch_size, seq_len]
            start_level: Optional starting level (defaults to 0)
            target_level: Optional target level (defaults to max level)
        Returns:
            Decoded token sequences at target level [batch_size, new_seq_len]
        """
        # Default to decoding from top level to bottom level
        if start_level is None:
            start_level = 0
        if target_level is None:
            target_level = self.num_levels - 1

        if not (0 <= start_level < self.num_levels):
            raise ValueError(f"Invalid start_level: {start_level}")
        if not (0 <= target_level < self.num_levels):
            raise ValueError(f"Invalid target_level: {target_level}")
        if target_level < start_level:
            raise ValueError("Can only decode to same or higher levels")

        # Return input tokens if no decoding needed
        if target_level == start_level:
            return tokens

        # Start with input tokens
        current = tokens

        # Decode through intermediate levels
        for level in range(start_level, target_level):
            table = self.decode_tables[level]
            decoded = table[current]

            # Remove padding and flatten sequence
            mask = decoded != -1
            lengths = mask.sum(dim=-1)
            max_length = int(lengths.sum(dim=-1).max().item())

            # Create output tensor with proper shape and type
            result = torch.full(
                size=(decoded.shape[0], max_length),
                fill_value=-1,
                dtype=torch.long,
                device=self.device,
            )

            # Fill in decoded sequences
            pos = 0
            for i in range(decoded.shape[1]):
                seq_lengths = lengths[:, i]
                for b in range(decoded.shape[0]):
                    length = int(seq_lengths[b].item())
                    if length > 0:
                        result[b, pos : pos + length] = decoded[b, i, :length]
                pos += int(seq_lengths.max().item())

            current = result

        return current[current != -1].view(tokens.shape[0], -1)

    def _build_decode_tables(self) -> list[torch.Tensor]:
        """Build lookup tables for decoding between levels.

        Returns:
            List of tensors mapping level i tokens to level i+1 sequences
            Each tensor has shape [parent_vocab_size, max_child_sequence_length]
            with padded sequences for consistent shape
        """
        tables = []
        for level in self.levels:
            max_length = max(len(seq) for seq in level.sequences.values())

            table = torch.full(
                size=(level.vocab_size, max_length),
                fill_value=-1,
                dtype=torch.long,
                device=self.device,
            )

            for token, sequence in level.sequences.items():
                table[token, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=self.device
                )

            tables.append(table)

        return tables

    @classmethod
    def from_sequences(
        cls,
        sequences: list[dict[TokenIdx, TokenSeq]],
        chunk_sizes: list[int],
        device: str | None = None,
    ) -> Self:
        """Create hierarchy from sequence mappings.
        Args:
            sequences: Mappings for each level
            chunk_sizes: Number of tokens per chunk at each level
            device: Optional compute device
        Returns:
            Initialized VocabHierarchy
        """
        if len(sequences) != len(chunk_sizes):
            raise ValueError("Must provide chunk size for each level")

        levels = []
        for seq_map, chunk_size in zip(sequences, chunk_sizes):
            level = VocabLevel(
                vocab_size=len(seq_map),
                chunk_size=chunk_size,
                sequences=seq_map,
            )
            levels.append(level)

        return cls(levels, device=device)

    def __getitem__(self, level: int) -> VocabLevel:
        """Get vocabulary level by index."""
        return self.levels[level]

    def __len__(self) -> int:
        """Get number of vocabulary levels."""
        return len(self.levels)

    def __iter__(self) -> Iterator[VocabLevel]:
        """Iterate over vocabulary levels."""
        return iter(self.levels)
