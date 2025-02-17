# faux_lingo/core/topics.py
"""Core functionality for topic-based sequence generation."""

from pathlib import Path
from typing import TypeAlias
import torch

# Type aliases for tensor dimensions
BatchDim: TypeAlias = int
VocabSize: TypeAlias = int 
NumTopics: TypeAlias = int

class TopicVectorSpace:
    """
    Manages a set of orthonormal topic vectors that define token distributions.
    
    Core mathematical properties:
    1. Each topic vector is unit length
    2. All topic vectors are orthogonal to each other
    3. Topic vectors form a basis for generating token distributions
    """
    
    def __init__(
        self,
        n_topics: int,
        vocab_size: int,
        vectors: torch.Tensor | None = None,
        device: str = "cuda"
    ):
        """
        Initialize topic vector space.
        
        Args:
            n_topics: Number of topics (must be <= vocab_size)
            vocab_size: Size of token vocabulary
            vectors: Optional pre-defined topic vectors
            device: Compute device for tensors
        """
        if n_topics > vocab_size:
            raise ValueError(f"n_topics ({n_topics}) must be <= vocab_size ({vocab_size})")
            
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.device = device
        
        if vectors is not None:
            self._validate_vectors(vectors)
            self.vectors = vectors.to(device)
        else:
            self.vectors = self._init_random_vectors()
            
    def _validate_vectors(self, vectors: torch.Tensor) -> None:
        """
        Validate topic vector properties.
        
        Args:
            vectors: Topic vectors to validate
            
        Raises:
            ValueError: If vectors don't meet required properties
        """
        if vectors.shape != (self.n_topics, self.vocab_size):
            raise ValueError(
                f"Vector shape {vectors.shape} doesn't match "
                f"({self.n_topics}, {self.vocab_size})"
            )
            
        # Check unit length
        norms = torch.linalg.norm(vectors, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms)):
            raise ValueError("Topic vectors must have unit length")
            
        # Check orthogonality
        gram = vectors @ vectors.T
        should_be_identity = torch.eye(self.n_topics, device=vectors.device)
        if not torch.allclose(gram, should_be_identity, atol=1e-6):
            raise ValueError("Topic vectors must be orthogonal")
            
    def _init_random_vectors(self) -> torch.Tensor:
        """
        Initialize random orthonormal topic vectors.
        
        Returns:
            Tensor of orthonormal vectors
        """
        vectors = torch.randn(self.n_topics, self.vocab_size, device=self.device)
        # Use QR decomposition to get orthonormal basis
        Q, _ = torch.linalg.qr(vectors.T)
        return Q.T
        
    def get_distribution(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Get token distribution for a topic mixture.
        
        Args:
            mixture: Topic mixture weights [batch_size, n_topics]
            
        Returns:
            Token probabilities [batch_size, vocab_size]
            
        Notes:
            Probabilities may need further processing (e.g., ReLU, normalization)
            to get final transition probabilities
        """
        # Validate mixture
        if mixture.shape[-1] != self.n_topics:
            raise ValueError(
                f"Mixture shape {mixture.shape} doesn't match n_topics {self.n_topics}"
            )
            
        # Project mixture onto topic vectors
        return mixture @ self.vectors
        
    def save(self, path: Path) -> None:
        """Save topic vectors."""
        torch.save(self.vectors, path)
        
    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> "TopicVectorSpace":
        """Load topic vectors and construct space."""
        vectors = torch.load(path, map_location=device)
        n_topics, vocab_size = vectors.shape
        return cls(n_topics, vocab_size, vectors=vectors, device=device)
