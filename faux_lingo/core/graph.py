# faux_lingo/core/graph.py

"""
Graph-based transition system with color-coded nodes for synthetic language generation.
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np
import random
from loguru import logger

@dataclass
class GraphConfig:
    """Configuration for colored graph generation."""
    num_colors: int
    avg_degree: int
    vocab_size: int
    sigma: float = 1.0
    epsilon: float = 0.1
    random_color_transitions: bool = False
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_colors < 1:
            raise ValueError("num_colors must be positive")
        if self.avg_degree < 1:
            raise ValueError("avg_degree must be positive")
        if self.avg_degree > self.num_colors:
            raise ValueError("avg_degree cannot exceed number of colors")
        if self.vocab_size < self.num_colors:
            raise ValueError("vocab_size must be at least num_colors")

class ColoredGraph:
    """
    Manages a graph where nodes (words) have assigned colors and transitions
    follow color-based constraints.
    """
    
    def __init__(self, config: GraphConfig, seed: int | None = None):
        """Initialize the colored graph system."""
        self.config = config
        self.config.validate()
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        
        # Initialize containers
        self.word_colors: Dict[int, int] = {}
        self.transition_matrix: np.ndarray | None = None
        self.color_matrix: np.ndarray | None = None
        
        # Cache for efficiency
        self._color_to_words: Dict[int, List[int]] = {}
        
        logger.info("Initialized ColoredGraph with config: {}", config)
    
    def build(self) -> dict:
        """Build the complete graph system and return relevant matrices."""
        self.assign_colors()
        self.build_color_transitions()
        self.build_transition_matrix()
        
        return {
            "word_colors": self.word_colors,
            "transition_matrix": self.transition_matrix,
            "color_matrix": self.color_matrix
        }
    
    def assign_colors(self) -> None:
        """Assign a random color to each word index."""
        self.word_colors = {
            i: self._rng.randint(0, self.config.num_colors - 1)
            for i in range(self.config.vocab_size)
        }
        
        # Build reverse mapping for efficiency
        self._color_to_words = {}
        for word, color in self.word_colors.items():
            self._color_to_words.setdefault(color, []).append(word)
        
        # Verify we have words of each color
        used_colors = set(self.word_colors.values())
        if len(used_colors) < self.config.num_colors:
            logger.warning(
                "Not all colors were assigned. Expected {} colors, got {}",
                self.config.num_colors, len(used_colors)
            )
            # Force at least one word of each color
            for color in range(self.config.num_colors):
                if color not in used_colors:
                    word = self._rng.randint(0, self.config.vocab_size - 1)
                    self.word_colors[word] = color
                    self._color_to_words.setdefault(color, []).append(word)
        
        logger.debug("Assigned colors to {} words", len(self.word_colors))
    
    def build_color_transitions(self) -> None:
        """
        Build the color transition matrix. If random_color_transitions is True,
        sample from a Dirichlet distribution with parameters based on color distance.
        Otherwise, use uniform transitions.
        """
        n_colors = self.config.num_colors
        if self.config.random_color_transitions:
            # Sample transition probabilities based on color distance
            self.color_matrix = np.zeros((n_colors, n_colors))
            for i in range(n_colors):
                # Set alpha parameters based on distance between colors
                alphas = np.array([
                    np.exp(-((i - j) ** 2) / (2 * self.config.sigma**2)) + self.config.epsilon
                    for j in range(n_colors)
                ])
                # Sample from Dirichlet distribution
                self.color_matrix[i] = self._np_rng.dirichlet(alphas)
        else:
            # Use uniform transitions between colors
            self.color_matrix = np.full((n_colors, n_colors), 1.0 / n_colors)
        
        logger.debug("Built color transition matrix of shape {}", self.color_matrix.shape)
    
    def build_transition_matrix(self) -> None:
        """
        Build the word transition matrix using color constraints.
        For each word, sample avg_degree target colors according to the color matrix,
        then choose random words of those colors as targets.
        """
        if self.color_matrix is None:
            raise RuntimeError("Color transition matrix must be built first")
        
        # Initialize sparse transition matrix
        self.transition_matrix = np.zeros((self.config.vocab_size, self.config.vocab_size))
        
        for i in range(self.config.vocab_size):
            src_color = self.word_colors[i]
            
            # Sample target colors according to color transition probabilities
            p_colors = self.color_matrix[src_color]
            available_colors = np.arange(self.config.num_colors)
            target_colors = self._np_rng.choice(
                available_colors,
                size=self.config.avg_degree,
                replace=False,
                p=p_colors
            )
            
            # For each target color, choose a random word of that color
            chosen_words = []
            for color in target_colors:
                word = self._rng.choice(self._color_to_words[color])
                chosen_words.append(word)
            
            # Assign random weights to transitions
            weights = self._np_rng.rand(self.config.avg_degree)
            weights /= weights.sum()  # Normalize to create probability distribution
            
            # Set transition probabilities
            for idx, j in enumerate(chosen_words):
                self.transition_matrix[i, j] = weights[idx]
        
        logger.debug("Built transition matrix of shape {}", self.transition_matrix.shape)
    
    def get_steady_state(self, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
        """
        Compute the steady state distribution of the transition matrix
        using power iteration.
        """
        if self.transition_matrix is None:
            raise RuntimeError("Transition matrix must be built first")
        
        n = self.transition_matrix.shape[0]
        pi = np.ones(n) / n  # Initial uniform distribution
        
        for _ in range(max_iter):
            pi_next = pi @ self.transition_matrix
            if np.linalg.norm(pi_next - pi, 1) < tol:
                return pi_next
            pi = pi_next
        
        logger.warning("Steady state computation did not converge after {} iterations", max_iter)
        return pi
