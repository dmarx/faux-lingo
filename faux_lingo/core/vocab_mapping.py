# faux_lingo/core/vocab_mapping.py
"""Hierarchical vocabulary mapping and decoding."""

from dataclasses import dataclass
from typing import Iterator, Sequence, TypeAlias

import torch
from typing_extensions import Self

# Type aliases for clarity
TokenIdx: TypeAlias = int
TokenSeq: TypeAlias = tuple[int, ...]


@dataclass
class VocabLevel:
    """A single level in the vocabulary hierarchy.
    
    Attributes:
        vocab_size: Number of tokens at this level
        chunk_size: Number of tokens from parent level per token
        sequences: Mapping of each token to its constituent sequence
    """
    vocab_size: int
    chunk_size: int
    sequences: dict[TokenIdx, TokenSeq]
    
    def __post_init__(self):
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
    """Manages hierarchical relationships between vocabulary levels.
    
    Core functionality:
    1. Deterministic mapping between vocabulary levels
    2. Token sequence decoding to lower levels
    3. Integration with token generation systems
    """

    def __init__(
        self,
        levels: Sequence[VocabLevel],
        device: str | None = None,
    ):
        """Initialize vocabulary hierarchy.
        
        Args:
            levels: Sequence of vocabulary levels from lowest to highest
            device: Optional compute device, defaults to CPU
        """
        self.device = device if device else "cpu"
        self.levels = list(levels)
        
        # Create lookup tables for efficient decoding
        self.decode_tables = self._build_decode_tables()

    def _build_decode_tables(self) -> list[torch.Tensor]:
        """Build lookup tables for decoding between levels.
        
        Returns:
            List of tensors mapping parent tokens to child sequences
            Each tensor has shape [parent_vocab_size, max_child_sequence_length]
            with padded sequences for consistent shape
        """
        tables = []
        for i in range(len(self.levels) - 1):
            parent = self.levels[i+1]
            child = self.levels[i]
            max_length = max(len(seq) for seq in parent.sequences.values())
            
            # Create table with padding
            table = torch.full(
                (parent.vocab_size, max_length),
                -1,  # Use -1 as padding token
                dtype=torch.long,
                device=self.device,
            )
            
            # Fill in sequences
            for token, sequence in parent.sequences.items():
                table[token, :len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=self.device
                )
            
            tables.append(table)
            
        return tables

    def decode_sequence(
        self,
        tokens: torch.Tensor,
        start_level: int,
        target_level: int,
    ) -> torch.Tensor:
        """Decode token sequence from one level to another.
        
        Args:
            tokens: Input token sequence [batch_size, seq_len]
            start_level: Index of starting vocabulary level
            target_level: Index of target vocabulary level
            
        Returns:
            Decoded token sequences at target level [batch_size, new_seq_len]
            where new_seq_len accounts for sequence expansion
        """
        if not (0 <= start_level < len(self.levels)):
            raise ValueError(f"Invalid start_level: {start_level}")
        if not (0 <= target_level < len(self.levels)):
            raise ValueError(f"Invalid target_level: {target_level}")
        if target_level >= start_level:
            raise ValueError("Can only decode to lower levels")
        
        # Start with input tokens
        current = tokens
        
        # Decode through intermediate levels
        for level in range(start_level - 1, target_level - 1, -1):
            # Get decode table for this level
            table = self.decode_tables[level]
            
            # Look up sequences for each token
            decoded = table[current]  # [batch, seq_len, max_child_seq_len]
            
            # Remove padding and flatten sequence
            mask = decoded != -1
            lengths = mask.sum(dim=-1)  # [batch, seq_len]
            max_length = lengths.sum(dim=-1).max().item()
            
            # Create output tensor
            result = torch.full(
                (decoded.shape[0], max_length),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            
            # Fill in decoded sequences
            pos = 0
            for i in range(decoded.shape[1]):
                seq_lengths = lengths[:, i]
                for b in range(decoded.shape[0]):
                    length = seq_lengths[b].item()
                    if length > 0:
                        result[b, pos:pos+length] = decoded[b, i, :length]
                pos += seq_lengths.max().item()
            
            current = result
            
        return current[current != -1].view(tokens.shape[0], -1)

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
