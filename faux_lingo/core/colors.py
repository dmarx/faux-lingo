# faux_lingo/core/colors.py
"""Color class management and transition rules for token sequences."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias
import torch

# Type aliases for dimensions
NumColors: TypeAlias = int
VocabSize: TypeAlias = int

@dataclass
class ColorMapping:
    """Maps between token indices and color classes.
    
    Attributes:
        boundaries: Tensor of token index boundaries for each color
        fractions: Normalized fraction of vocabulary for each color
    """
    boundaries: torch.Tensor  # [num_colors + 1]
    fractions: torch.Tensor   # [num_colors]

class ColorSpace:
    """
    Manages color classes and their transition rules.
    
    Core properties:
    1. Each token belongs to exactly one color class
    2. Color classes partition the vocabulary space
    3. Transitions between colors follow specified rules
    """
    
    def __init__(
        self,
        color_fractions: list[float] | torch.Tensor,
        vocab_size: int,
        transition_weights: torch.Tensor | None = None,
        device: str | None = None
    ):
        """
        Initialize color space with fractions and transition rules.
        
        Args:
            color_fractions: Relative sizes of color classes
            vocab_size: Total vocabulary size
            transition_weights: Optional matrix of color transition weights
            device: Optional compute device, defaults to CPU
            
        Notes:
            - Color fractions will be normalized to sum to 1
            - If transition_weights not provided, defaults to all-ones matrix
        """
        self.device = device if device else "cpu"
        
        # Convert and normalize color fractions
        if isinstance(color_fractions, list):
            color_fractions = torch.tensor(color_fractions, dtype=torch.float32)
        self.n_colors = len(color_fractions)
        
        # Compute normalized fractions and token boundaries
        self.mapping = self._compute_mapping(color_fractions, vocab_size)
        self.vocab_size = vocab_size
        
        # Setup transition weights
        if transition_weights is not None:
            self._validate_transitions(transition_weights)
            self.transition_weights = transition_weights.to(self.device)
        else:
            self.transition_weights = torch.ones(
                (self.n_colors, self.n_colors), 
                device=self.device
            )
    
    def _compute_mapping(
        self, 
        fractions: torch.Tensor,
        vocab_size: int
    ) -> ColorMapping:
        """
        Compute normalized fractions and token boundaries.
        
        Args:
            fractions: Raw color fractions
            vocab_size: Total vocabulary size
            
        Returns:
            ColorMapping with normalized fractions and boundaries
        """
        # Normalize fractions
        fractions = fractions.to(self.device)
        normalized = fractions / fractions.sum()
        
        # Compute token counts and boundaries
        counts = (normalized * vocab_size).long()
        
        # Adjust last count to ensure total = vocab_size
        total = counts.sum()
        if total < vocab_size:
            counts[-1] += vocab_size - total
            
        # Compute boundaries
        boundaries = torch.zeros(self.n_colors + 1, 
                               dtype=torch.long,
                               device=self.device)
        torch.cumsum(counts, dim=0, out=boundaries[1:])
        
        return ColorMapping(boundaries=boundaries, fractions=normalized)
    
    def _validate_transitions(self, weights: torch.Tensor) -> None:
        """
        Validate transition weight matrix.
        
        Args:
            weights: Color transition weight matrix
            
        Raises:
            ValueError: If weights have invalid shape or values
        """
        if weights.shape != (self.n_colors, self.n_colors):
            raise ValueError(
                f"Transition weights shape {weights.shape} "
                f"doesn't match n_colors {self.n_colors}"
            )
        if not torch.all(weights >= 0):
            raise ValueError("Transition weights must be non-negative")
    
    def get_color(self, token_idx: int) -> int:
        """
        Get color index for a token index.
        
        Args:
            token_idx: Index in vocabulary
            
        Returns:
            Index of the color that token_idx belongs to
            
        Raises:
            ValueError: If token_idx is invalid
        """
        if not 0 <= token_idx < self.vocab_size:
            raise ValueError(f"Invalid token_idx {token_idx}")
        # searchsorted gives us the index where token_idx would be inserted
        # which is exactly the color index we want
        return torch.searchsorted(self.mapping.boundaries[1:], token_idx).item()
    
    def get_color_range(self, color_idx: int) -> tuple[int, int]:
        """
        Get token index range for a color.
        
        Args:
            color_idx: Index of the color
            
        Returns:
            Tuple of (start_idx, end_idx) for color's token range
            
        Raises:
            ValueError: If color_idx is invalid
        """
        if not 0 <= color_idx < self.n_colors:
            raise ValueError(f"Invalid color_idx {color_idx}")
        return (
            self.mapping.boundaries[color_idx].item(),
            self.mapping.boundaries[color_idx + 1].item()
        )
    
    def get_transition_mask(self) -> torch.Tensor:
        """
        Get vocabulary-sized mask from color transition weights.
        
        Returns:
            Boolean mask of shape [vocab_size, vocab_size]
        """
        mask = torch.zeros(
            (self.vocab_size, self.vocab_size),
            device=self.device
        )
        
        for i in range(self.n_colors):
            i_start, i_end = self.get_color_range(i)
            for j in range(self.n_colors):
                j_start, j_end = self.get_color_range(j)
                if self.transition_weights[i, j] > 0:
                    mask[i_start:i_end, j_start:j_end] = self.transition_weights[i, j]
                    
        return mask
    
    def save(self, path: Path) -> None:
        """Save color space parameters."""
        data = {
            "fractions": self.mapping.fractions.cpu(),
            "boundaries": self.mapping.boundaries.cpu(),
            "transition_weights": self.transition_weights.cpu(),
            "vocab_size": self.vocab_size
        }
        torch.save(data, path)
        
    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "ColorSpace":
        """Load color space from saved parameters."""
        data = torch.load(path)
        color_space = cls(
            color_fractions=data["fractions"],
            vocab_size=data["vocab_size"],
            transition_weights=data["transition_weights"],
            device=device
        )
        return color_space
