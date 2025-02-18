# faux_lingo/core/vocab_builder.py
"""Builder for constructing hierarchical vocabularies."""

from dataclasses import dataclass
import random
from typing import TypeAlias

from loguru import logger

from .vocab_mapping import VocabHierarchy, VocabLevel

# Type aliases
TokenIdx: TypeAlias = int
TokenSeq: TypeAlias = tuple[int, ...]


@dataclass
class BuilderConfig:
    """Configuration for vocabulary hierarchy construction.
    
    Attributes:
        token_vocab_size: Size of base token vocabulary
        sequence_lengths: List of sequence lengths for each level
        vocab_sizes: List of vocabulary sizes for each level
        seed: Optional random seed for reproducibility
    """

    token_vocab_size: int
    sequence_lengths: list[int]  # Each level's sequence length
    vocab_sizes: list[int]  # Each level's vocabulary size
    seed: int | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.token_vocab_size < 1:
            raise ValueError("token_vocab_size must be positive")
        if len(self.sequence_lengths) != len(self.vocab_sizes):
            raise ValueError(
                "Must specify sequence length and vocabulary size for each level"
            )
        if any(l < 1 for l in self.sequence_lengths):
            raise ValueError("All sequence lengths must be positive")
        if any(v < 1 for v in self.vocab_sizes):
            raise ValueError("All vocabulary sizes must be positive")

        # Compute and validate potential combinations at each level
        tokens = self.token_vocab_size
        for level, (length, size) in enumerate(
            zip(self.sequence_lengths, self.vocab_sizes)
        ):
            max_combinations = tokens**length
            if size > max_combinations:
                raise ValueError(
                    f"Level {level}: vocab_size ({size}) exceeds maximum "
                    f"possible combinations ({max_combinations})"
                )
            tokens = size  # Next level builds from this vocabulary


class VocabBuilder:
    """Builds hierarchical vocabularies with constrained structure.
    
    Core functionality:
    1. Random sampling of valid token sequences
    2. Building vocabularies level by level
    3. Tracking used sequences to avoid duplicates
    """

    def __init__(self, config: BuilderConfig):
        """Initialize builder with configuration.
        
        Args:
            config: Parameters for vocabulary construction
        """
        self.config = config
        self._rng = random.Random(config.seed)
        
        # Initialize sequence tracking
        self._used_sequences: list[set[TokenSeq]] = [
            set() for _ in range(len(config.vocab_sizes))
        ]
        
        logger.info("Initialized VocabBuilder with config: {}", config)

    def build(self) -> VocabHierarchy:
        """Build complete vocabulary hierarchy.
        
        Returns:
            VocabHierarchy with all levels constructed
        """
        levels = []
        current_vocab_size = self.config.token_vocab_size

        # Build each level
        for level, (seq_len, vocab_size) in enumerate(
            zip(self.config.sequence_lengths, self.config.vocab_sizes)
        ):
            logger.debug("Building level {} vocabulary...", level)
            
            # Generate valid sequences for this level
            sequences = {}
            while len(sequences) < vocab_size:
                seq = tuple(
                    self._rng.randrange(current_vocab_size)
                    for _ in range(seq_len)
                )
                if seq not in self._used_sequences[level]:
                    token_idx = len(sequences)
                    sequences[token_idx] = seq
                    self._used_sequences[level].add(seq)

            # Create vocabulary level
            level = VocabLevel(
                vocab_size=vocab_size,
                chunk_size=seq_len,
                sequences=sequences,
            )
            levels.append(level)
            
            # Update for next level
            current_vocab_size = vocab_size
            
            logger.debug(
                "Built level {} with {} sequences of length {}",
                level,
                vocab_size,
                seq_len,
            )

        return VocabHierarchy(levels)

    @classmethod
    def create_default_config(cls) -> BuilderConfig:
        """Create configuration with reasonable defaults.
        
        Returns:
            BuilderConfig for simple three-level hierarchy
        """
        return BuilderConfig(
            token_vocab_size=10,  # Base tokens (0-9)
            sequence_lengths=[2, 3, 2],  # Length at each level
            vocab_sizes=[20, 15, 10],  # Vocabulary sizes
        )


def create_word_hierarchy(
    token_vocab_size: int = 10,
    n_chars: int = 20,
    n_words: int = 100,
    chars_per_word: int = 3,
    seed: int | None = None,
) -> VocabHierarchy:
    """Convenience function to create character-word vocabulary.
    
    Args:
        token_vocab_size: Size of base token vocabulary
        n_chars: Number of unique characters
        n_words: Number of unique words
        chars_per_word: Number of characters per word
        seed: Optional random seed
        
    Returns:
        Two-level hierarchy mapping words to character sequences
    """
    config = BuilderConfig(
        token_vocab_size=token_vocab_size,
        sequence_lengths=[2, chars_per_word],  # tokens->chars, chars->words
        vocab_sizes=[n_chars, n_words],
        seed=seed,
    )
    return VocabBuilder(config).build()
