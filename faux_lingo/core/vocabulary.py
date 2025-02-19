# faux_lingo/core/vocabulary.py
"""Centralized vocabulary management system."""

from dataclasses import dataclass
from typing import TypeAlias

from .special_tokens import SpecialTokens
from .vocab_mapping import VocabHierarchy, VocabLevel

TokenIdx: TypeAlias = int


@dataclass
class VocabConfig:
    """Configuration for vocabulary system.

    Attributes:
        base_vocab_size: Size of the base vocabulary for latent transitions
        chunk_sizes: Size of chunks at each hierarchy level
        level_sizes: Vocabulary size at each hierarchy level
        use_special_tokens: Which special tokens to include
    """

    base_vocab_size: int
    chunk_sizes: list[int]
    level_sizes: list[int]
    use_special_tokens: dict[str, bool] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.chunk_sizes) != len(self.level_sizes):
            raise ValueError("Must specify sizes for each hierarchy level")

        # Default special tokens configuration
        if self.use_special_tokens is None:
            self.use_special_tokens = {
                "pad": False,
                "bos": False,
                "eos": False,
                "unk": False,
            }


class Vocabulary:
    """
    Central manager for vocabulary components.

    Manages:
    1. Base vocabulary for latent transitions
    2. Hierarchical mappings between vocabulary levels
    3. Special tokens at concrete output level
    """

    def __init__(self, config: VocabConfig):
        """Initialize vocabulary system.

        Args:
            config: Vocabulary configuration
        """
        self.config = config
        self._build_hierarchy()
        self._setup_special_tokens()

    def _build_hierarchy(self) -> None:
        """Build vocabulary hierarchy from configuration."""
        levels = []
        current_vocab_size = self.config.base_vocab_size

        # Build each level
        for level_size, chunk_size in zip(
            self.config.level_sizes, self.config.chunk_sizes
        ):
            # Initialize sequences for this level
            sequences: dict[TokenIdx, tuple[int, ...]] = {}
            tokens_per_chunk = chunk_size

            # Generate sequential mappings for simplicity
            # In practice, these would be learned or configured
            for token_idx in range(level_size):
                base_token = token_idx % current_vocab_size
                sequences[token_idx] = tuple(
                    (base_token + i) % current_vocab_size
                    for i in range(tokens_per_chunk)
                )

            # Create vocabulary level
            level = VocabLevel(
                vocab_size=level_size,
                chunk_size=chunk_size,
                sequences=sequences,
            )
            levels.append(level)
            current_vocab_size = level_size

        self.hierarchy = VocabHierarchy(levels)

    def _setup_special_tokens(self) -> None:
        """Initialize special tokens if configured."""
        if any(self.config.use_special_tokens.values()):
            # Use final level vocabulary size as base for special tokens
            concrete_vocab_size = (
                self.hierarchy[-1].vocab_size
                if self.hierarchy
                else self.config.base_vocab_size
            )

            self.special_tokens = SpecialTokens.from_base_vocab(
                base_vocab_size=concrete_vocab_size,
                pad=self.config.use_special_tokens["pad"],
                bos=self.config.use_special_tokens["bos"],
                eos=self.config.use_special_tokens["eos"],
                unk=self.config.use_special_tokens["unk"],
            )
        else:
            self.special_tokens = None

    @property
    def base_vocab_size(self) -> int:
        """Size of base vocabulary for latent transitions."""
        return self.config.base_vocab_size

    @property
    def concrete_vocab_size(self) -> int:
        """Total size of concrete vocabulary including special tokens."""
        base_size = (
            self.hierarchy[-1].vocab_size
            if self.hierarchy
            else self.config.base_vocab_size
        )
        if self.special_tokens:
            return self.special_tokens.total_vocab_size
        return base_size

    @property
    def has_hierarchy(self) -> bool:
        """Whether vocabulary uses hierarchical mappings."""
        return len(self.hierarchy) > 0 if self.hierarchy else False

    @classmethod
    def create_simple(
        cls,
        base_vocab_size: int,
        *,
        pad: bool = False,
        bos: bool = False,
        eos: bool = False,
        unk: bool = False,
    ) -> "Vocabulary":
        """Create simple vocabulary without hierarchy.

        Args:
            base_vocab_size: Size of base vocabulary
            pad: Whether to include padding token
            bos: Whether to include beginning of sequence token
            eos: Whether to include end of sequence token
            unk: Whether to include unknown token

        Returns:
            Vocabulary with just base tokens and optionally special tokens
        """
        config = VocabConfig(
            base_vocab_size=base_vocab_size,
            chunk_sizes=[],
            level_sizes=[],
            use_special_tokens={
                "pad": pad,
                "bos": bos,
                "eos": eos,
                "unk": unk,
            },
        )
        return cls(config)

    @classmethod
    def create_hierarchical(
        cls,
        base_vocab_size: int,
        *,
        level_configs: list[tuple[int, int]],  # List of (size, chunk_size) pairs
        pad: bool = False,
        bos: bool = False,
        eos: bool = False,
        unk: bool = False,
    ) -> "Vocabulary":
        """Create hierarchical vocabulary.

        Args:
            base_vocab_size: Size of base vocabulary
            level_configs: List of (vocab_size, chunk_size) pairs for each level
            pad: Whether to include padding token
            bos: Whether to include beginning of sequence token
            eos: Whether to include end of sequence token
            unk: Whether to include unknown token

        Returns:
            Vocabulary with hierarchical structure
        """
        level_sizes, chunk_sizes = zip(*level_configs)
        config = VocabConfig(
            base_vocab_size=base_vocab_size,
            chunk_sizes=list(chunk_sizes),
            level_sizes=list(level_sizes),
            use_special_tokens={
                "pad": pad,
                "bos": bos,
                "eos": eos,
                "unk": unk,
            },
        )
        return cls(config)
