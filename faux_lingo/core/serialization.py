# faux_lingo/core/serialization.py
"""Serialization utilities for vocabulary and generation metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .colors import ColorSpace
from .topics import TopicVectorSpace
from .transitions import TransitionMatrix
from .vocab_mapping import VocabHierarchy

# Type aliases
ConfigDict: TypeAlias = dict[str, Any]


@dataclass
class GenerationMetadata:
    """Metadata for tracking generation state and configuration.

    Attributes:
        config: Generation configuration
        vocab_hierarchy: Current vocabulary state
        transition_model: Current transition model state
        sequences_generated: Number of sequences generated
        last_batch_id: ID of last generated batch
    """

    config: DictConfig
    vocab_hierarchy: VocabHierarchy
    transition_model: TransitionMatrix
    sequences_generated: int = 0
    last_batch_id: int = 0

    def save(self, path: Path) -> None:
        """Save generation metadata to disk.

        Args:
            path: Directory to save metadata files
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = path / "config.yaml"
        OmegaConf.save(self.config, config_path)
        logger.info(f"Saved configuration to {config_path}")

        # Save vocabulary hierarchy
        vocab_path = path / "vocab"
        vocab_path.mkdir(exist_ok=True)
        for i, level in enumerate(self.vocab_hierarchy):
            level_path = vocab_path / f"level_{i}.pt"
            torch.save(level.sequences, level_path)
        logger.info(f"Saved vocabulary hierarchy to {vocab_path}")

        # Save transition model components
        model_path = path / "model"
        model_path.mkdir(exist_ok=True)
        
        topic_path = model_path / "topic_vectors.pt"
        self.transition_model.topic_space.save(topic_path)
        
        color_path = model_path / "color_space.pt"
        self.transition_model.color_space.save(color_path)
        logger.info(f"Saved transition model to {model_path}")

        # Save generation state
        state_path = path / "state.pt"
        torch.save(
            {
                "sequences_generated": self.sequences_generated,
                "last_batch_id": self.last_batch_id,
            },
            state_path,
        )
        logger.info(f"Saved generation state to {state_path}")

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "GenerationMetadata":
        """Load generation metadata from disk.

        Args:
            path: Directory containing metadata files
            device: Optional device for loading model components

        Returns:
            Loaded GenerationMetadata instance
        """
        if not path.is_dir():
            raise ValueError(f"Metadata directory {path} does not exist")

        # Load configuration
        config_path = path / "config.yaml"
        config = OmegaConf.load(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Load vocabulary hierarchy
        vocab_path = path / "vocab"
        if not vocab_path.is_dir():
            raise ValueError(f"Vocabulary directory {vocab_path} does not exist")

        level_paths = sorted(vocab_path.glob("level_*.pt"))
        sequences = []
        for level_path in level_paths:
            sequences.append(torch.load(level_path))
        
        # Get chunk sizes from config
        chunk_sizes = config.vocab.chunk_sizes
        vocab_hierarchy = VocabHierarchy.from_sequences(
            sequences, chunk_sizes, device=device
        )
        logger.info(f"Loaded vocabulary hierarchy from {vocab_path}")

        # Load transition model components
        model_path = path / "model"
        if not model_path.is_dir():
            raise ValueError(f"Model directory {model_path} does not exist")

        topic_space = TopicVectorSpace.load(
            model_path / "topic_vectors.pt", device=device
        )
        color_space = ColorSpace.load(
            model_path / "color_space.pt", device=device
        )
        transition_model = TransitionMatrix(topic_space, color_space, device=device)
        logger.info(f"Loaded transition model from {model_path}")

        # Load generation state
        state_path = path / "state.pt"
        if state_path.exists():
            state = torch.load(state_path)
            sequences_generated = state["sequences_generated"]
            last_batch_id = state["last_batch_id"]
            logger.info(f"Loaded generation state from {state_path}")
        else:
            sequences_generated = 0
            last_batch_id = 0
            logger.warning(f"No generation state found at {state_path}")

        return cls(
            config=config,
            vocab_hierarchy=vocab_hierarchy,
            transition_model=transition_model,
            sequences_generated=sequences_generated,
            last_batch_id=last_batch_id,
        )
