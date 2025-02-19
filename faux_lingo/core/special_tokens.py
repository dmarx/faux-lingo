# faux_lingo/core/special_tokens.py
"""Special token management for output sequences."""

from dataclasses import dataclass
from typing import TypeAlias

TokenIdx: TypeAlias = int


@dataclass
class SpecialTokens:
    """Special tokens for output sequences.

    These tokens are only used at the most concrete vocabulary level
    and are managed by the sequence generator.

    Attributes:
        pad_token: Token used for padding sequences
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        unk_token: Token for unknown/rare tokens
        base_vocab_size: Size of base vocabulary before special tokens
    """

    base_vocab_size: int
    pad_token: TokenIdx | None = None
    bos_token: TokenIdx | None = None
    eos_token: TokenIdx | None = None
    unk_token: TokenIdx | None = None

    def __post_init__(self) -> None:
        """Initialize special token indices after base vocab."""
        current_idx = self.base_vocab_size
        if self.pad_token is not None:
            self.pad_token = current_idx
            current_idx += 1
        if self.bos_token is not None:
            self.bos_token = current_idx
            current_idx += 1
        if self.eos_token is not None:
            self.eos_token = current_idx
            current_idx += 1
        if self.unk_token is not None:
            self.unk_token = current_idx
            current_idx += 1

    @property
    def total_vocab_size(self) -> int:
        """Get total vocabulary size including special tokens."""
        return self.base_vocab_size + self.num_special()

    def num_special(self) -> int:
        """Get number of special tokens in use."""
        return sum(
            1
            for token in [
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token,
            ]
            if token is not None
        )

    @classmethod
    def from_base_vocab(
        cls,
        base_vocab_size: int,
        *,
        pad: bool = False,
        bos: bool = False,
        eos: bool = False,
        unk: bool = False,
    ) -> "SpecialTokens":
        """Create special tokens configuration from base vocabulary.

        Args:
            base_vocab_size: Size of base vocabulary
            pad: Whether to include padding token
            bos: Whether to include beginning of sequence token
            eos: Whether to include end of sequence token
            unk: Whether to include unknown token

        Returns:
            SpecialTokens with requested tokens enabled
        """
        return cls(
            base_vocab_size=base_vocab_size,
            pad_token=0 if pad else None,
            bos_token=0 if bos else None,
            eos_token=0 if eos else None,
            unk_token=0 if unk else None,
        )
