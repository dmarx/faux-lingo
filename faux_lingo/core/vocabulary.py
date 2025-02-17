# faux_lingo/core/vocabulary.py

"""
Vocabulary generation system for creating hierarchical synthetic language structures.
"""

import random
from dataclasses import dataclass
from typing import List, Set, Tuple

from loguru import logger


@dataclass
class VocabConfig:
    """Configuration for vocabulary generation parameters."""

    token_vocab_size: int
    rune_vocab_size: int
    char_vocab_size: int
    word_vocab_size: int
    tokens_per_rune: int
    runes_per_char: int
    chars_per_word: int
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.token_vocab_size < 1:
            raise ValueError("token_vocab_size must be positive")
        if self.tokens_per_rune < 1:
            raise ValueError("tokens_per_rune must be positive")
        if self.runes_per_char < 1:
            raise ValueError("runes_per_char must be positive")
        if self.chars_per_word < 1:
            raise ValueError("chars_per_word must be positive")
            
        # Calculate maximum possible combinations at each level
        max_runes = self.token_vocab_size ** self.tokens_per_rune
        if self.rune_vocab_size > max_runes:
            raise ValueError(
                f"rune_vocab_size ({self.rune_vocab_size}) exceeds maximum "
                f"possible combinations ({max_runes})"
            )
            
        max_chars = self.rune_vocab_size ** self.runes_per_char
        if self.char_vocab_size > max_chars:
            raise ValueError(
                f"char_vocab_size ({self.char_vocab_size}) exceeds maximum "
                f"possible combinations ({max_chars})"
            )
            
        max_words = self.char_vocab_size ** self.chars_per_word
        if self.word_vocab_size > max_words:
            raise ValueError(
                f"word_vocab_size ({self.word_vocab_size}) exceeds maximum "
                f"possible combinations ({max_words})"
            )


class VocabBuilder:
    """Builds hierarchical vocabularies for synthetic language generation."""

    def __init__(self, config: VocabConfig, seed: int | None = None):
        """Initialize the vocabulary builder with configuration parameters."""
        self.config = config
        self.config.validate()
        self._rng = random.Random(seed)

        # Initialize vocabulary containers
        self.token_vocab: List[int] = []
        self.rune_vocab: List[Tuple[int, ...]] = []
        self.char_vocab: List[Tuple[Tuple[int, ...], ...]] = []
        self.word_vocab: List[Tuple[int, ...]] = []

        # Track used combinations
        self._used_runes: Set[Tuple[int, ...]] = set()
        self._used_chars: Set[Tuple[Tuple[int, ...], ...]] = set()
        self._used_words: Set[Tuple[int, ...]] = set()

        logger.info("Initialized VocabBuilder with config: {}", config)

    def build(self) -> dict:
        """Build all vocabulary levels and return them as a dictionary."""
        self.build_token_vocab()
        self.build_rune_vocab()
        self.build_char_vocab()
        self.build_word_vocab()

        return {
            "token_vocab": self.token_vocab,
            "rune_vocab": self.rune_vocab,
            "char_vocab": self.char_vocab,
            "word_vocab": self.word_vocab,
        }

    def build_token_vocab(self) -> None:
        """Build the base token vocabulary (integers from 0 to size-1)."""
        self.token_vocab = list(range(self.config.token_vocab_size))
        logger.debug("Built token vocabulary of size {}", len(self.token_vocab))

    def build_rune_vocab(self) -> None:
        """Build vocabulary of runes (tuples of tokens)."""
        if not self.token_vocab:
            raise RuntimeError("Token vocabulary must be built first")

        while len(self.rune_vocab) < self.config.rune_vocab_size:
            rune = tuple(
                self._rng.choices(self.token_vocab, k=self.config.tokens_per_rune)
            )
            if rune not in self._used_runes:
                self._used_runes.add(rune)
                self.rune_vocab.append(rune)

        logger.debug("Built rune vocabulary of size {}", len(self.rune_vocab))

    def build_char_vocab(self) -> None:
        """Build vocabulary of characters (tuples of runes)."""
        if not self.rune_vocab:
            raise RuntimeError("Rune vocabulary must be built first")

        while len(self.char_vocab) < self.config.char_vocab_size:
            char = tuple(
                self._rng.choices(self.rune_vocab, k=self.config.runes_per_char)
            )
            if char not in self._used_chars:
                self._used_chars.add(char)
                self.char_vocab.append(char)

        logger.debug("Built character vocabulary of size {}", len(self.char_vocab))

    def build_word_vocab(self) -> None:
        """Build vocabulary of words (flattened tuples of tokens)."""
        if not self.char_vocab:
            raise RuntimeError("Character vocabulary must be built first")

        while len(self.word_vocab) < self.config.word_vocab_size:
            # Build word as sequence of characters
            word_chars = tuple(
                self._rng.choices(self.char_vocab, k=self.config.chars_per_word)
            )

            # Flatten into sequence of tokens
            word_tokens = []
            for char in word_chars:
                for rune in char:
                    word_tokens.extend(rune)

            word = tuple(word_tokens)
            if word not in self._used_words:
                self._used_words.add(word)
                self.word_vocab.append(word)

        logger.debug("Built word vocabulary of size {}", len(self.word_vocab))


def create_default_config() -> VocabConfig:
    """Create a VocabConfig with reasonable default values."""
    return VocabConfig(
        token_vocab_size=10,  # Base tokens (0-9)
        rune_vocab_size=30,  # Combinations of tokens
        char_vocab_size=20,  # Combinations of runes
        word_vocab_size=100,  # Final vocabulary size
        tokens_per_rune=1,  # Simplest case: one token per rune
        runes_per_char=3,  # Three runes per character
        chars_per_word=3,  # Three characters per word
    )
