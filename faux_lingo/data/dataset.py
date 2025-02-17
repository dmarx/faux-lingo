# faux_lingo/data/dataset.py

"""
PyTorch dataset wrappers for synthetic language generation.
"""

from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, IterableDataset
from loguru import logger

from ..core.generator import DocumentGenerator

class GenerativeCorpusDataset(Dataset):
    """
    A map-style dataset that generates documents on demand using fixed artifacts.
    """
    
    def __init__(
        self,
        artifacts: Dict,
        doc_count: int,
        doc_length: int,
        doc_topic_alpha: float = 0.5,
        include_whitespace: bool = True,
        include_markers: bool = True,
        return_entropy: bool = False,
        seed: int | None = None
    ):
        """
        Initialize the dataset.
        
        Args:
            artifacts: Dictionary of fixed artifacts from ArtifactGenerator
            doc_count: Total number of documents to generate
            doc_length: Number of words per document
            doc_topic_alpha: Concentration parameter for topic mixture
            include_whitespace: Whether to add whitespace tokens between words
            include_markers: Whether to add BOD/EOD markers
            return_entropy: Whether to return entropy measures with documents
            seed: Random seed for reproducibility
        """
        self.doc_count = doc_count
        self.doc_length = doc_length
        self.return_entropy = return_entropy
        
        # Initialize document generator
        self.generator = DocumentGenerator(
            artifacts=artifacts,
            doc_topic_alpha=doc_topic_alpha,
            include_whitespace=include_whitespace,
            include_markers=include_markers,
            seed=seed
        )
        
        logger.info(
            "Initialized GenerativeCorpusDataset with {} documents of length {}",
            doc_count, doc_length
        )
    
    def __len__(self) -> int:
        return self.doc_count
    
    def __getitem__(self, idx: int) -> torch.Tensor | Tuple[torch.Tensor, float, float]:
        """
        Generate a document.
        
        Returns:
            If return_entropy is False:
                torch.Tensor of token indices
            If return_entropy is True:
                tuple of (tokens, avg_entropy, perplexity)
        """
        result = self.generator.generate(
            self.doc_length,
            return_entropy=self.return_entropy
        )
        
        if self.return_entropy:
            tokens, entropy, perplexity = result
            return torch.tensor(tokens, dtype=torch.long), entropy, perplexity
        
        return torch.tensor(result, dtype=torch.long)

class StreamingCorpusDataset(IterableDataset):
    """
    An iterable-style dataset that generates an infinite stream of documents.
    Useful for large-scale training where you don't need a fixed dataset size.
    """
    
    def __init__(
        self,
        artifacts: Dict,
        doc_length: int,
        doc_topic_alpha: float = 0.5,
        include_whitespace: bool = True,
        include_markers: bool = True,
        return_entropy: bool = False,
        seed: int | None = None
    ):
        """
        Initialize the streaming dataset.
        
        Args:
            artifacts: Dictionary of fixed artifacts from ArtifactGenerator
            doc_length: Number of words per document
            doc_topic_alpha: Concentration parameter for topic mixture
            include_whitespace: Whether to add whitespace tokens between words
            include_markers: Whether to add BOD/EOD markers
            return_entropy: Whether to return entropy measures with documents
            seed: Random seed for reproducibility
        """
        self.doc_length = doc_length
        self.return_entropy = return_entropy
        
        # Initialize document generator
        self.generator = DocumentGenerator(
            artifacts=artifacts,
            doc_topic_alpha=doc_topic_alpha,
            include_whitespace=include_whitespace,
            include_markers=include_markers,
            seed=seed
        )
        
        logger.info(
            "Initialized StreamingCorpusDataset generating documents of length {}",
            doc_length
        )
    
    def __iter__(self):
        """Generate an infinite stream of documents."""
        while True:
            result = self.generator.generate(
                self.doc_length,
                return_entropy=self.return_entropy
            )
            
            if self.return_entropy:
                tokens, entropy, perplexity = result
                yield torch.tensor(tokens, dtype=torch.long), entropy, perplexity
            else:
                yield torch.tensor(result, dtype=torch.long)
