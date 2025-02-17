import torch
from typing import Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass
from jaxtyping import Float, Int

@dataclass
class EntropyMetrics:
    """Container for various entropy-related metrics"""
    transition_entropy: float  # Average entropy of transition distributions
    color_entropy: float      # Entropy of color transitions
    token_entropy: float      # Empirical entropy of token sequences
    topic_entropy: float      # Entropy of topic mixtures
    mutual_info: Dict[str, float]  # Mutual information between various components

class InfoTheoryAnalyzer:
    """Analyzer for information-theoretic properties of generated sequences"""
    
    def __init__(self, generator):
        """
        Args:
            generator: ProbColorConstrainedGenerator instance
        """
        self.generator = generator
        
    def compute_transition_entropy(
        self,
        transition_matrix: Float[torch.Tensor, "batch vocab vocab"]
    ) -> Float[torch.Tensor, "batch"]:
        """Compute entropy of transition distributions.
        
        Args:
            transition_matrix: Batch of transition probability matrices
            
        Returns:
            Entropy for each batch element
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        P = transition_matrix + eps
        H = -torch.sum(P * torch.log2(P), dim=-1)  # Sum over transitions
        return torch.mean(H, dim=-1)  # Average over states
    
    def compute_color_entropy(
        self,
        sequences: Int[torch.Tensor, "batch seq_len"],
        window_size: int = 2
    ) -> float:
        """Compute empirical entropy of color transitions.
        
        Args:
            sequences: Batch of token sequences
            window_size: Size of n-gram window for transition stats
            
        Returns:
            Estimated color transition entropy
        """
        # Convert tokens to colors
        colors = torch.tensor([
            [self.generator.get_color(idx.item()) for idx in seq]
            for seq in sequences
        ], device=sequences.device)
        
        # Count n-gram transitions
        counts = torch.zeros(
            (self.generator.n_colors,) * window_size,
            device=sequences.device
        )
        
        for b in range(len(sequences)):
            for i in range(len(sequences[0]) - window_size + 1):
                ngram = tuple(colors[b, i:i+window_size].tolist())
                counts[ngram] += 1
                
        # Convert to probabilities and compute entropy
        P = counts / counts.sum()
        H = -torch.sum(P * torch.log2(P + 1e-10))
        return H.item()
    
    def estimate_mutual_information(
        self,
        sequences: Int[torch.Tensor, "batch seq_len"],
        mixtures: Float[torch.Tensor, "batch n_topics"],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Estimate mutual information between various components.
        
        Args:
            sequences: Generated token sequences
            mixtures: Topic mixtures used for generation
            n_bins: Number of bins for discretizing continuous values
            
        Returns:
            Dictionary of MI estimates between different components
        """
        # Get color sequences
        colors = torch.tensor([
            [self.generator.get_color(idx.item()) for idx in seq]
            for seq in sequences
        ], device=sequences.device)
        
        # Compute token frequencies per sequence
        token_freqs = torch.zeros(
            (len(sequences), self.generator.vocab_size),
            device=sequences.device
        )
        for b in range(len(sequences)):
            unique, counts = torch.unique(sequences[b], return_counts=True)
            token_freqs[b, unique] = counts.float() / len(sequences[b])
        
        # Discretize topic mixtures
        disc_mixtures = torch.tensor([
            np.digitize(mix, np.linspace(0, 1, n_bins))
            for mix in mixtures.cpu().numpy()
        ], device=sequences.device)
        
        # Compute mutual information estimates
        mi_estimates = {
            'color_topic': self._mutual_info_discrete(
                colors.view(-1), disc_mixtures.view(-1)
            ),
            'token_topic': self._mutual_info_discrete(
                sequences.view(-1), disc_mixtures.view(-1)
            ),
            'color_token': self._mutual_info_discrete(
                colors.view(-1), sequences.view(-1)
            )
        }
        
        return mi_estimates
    
    def _mutual_info_discrete(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> float:
        """Compute mutual information between discrete variables.
        
        Args:
            X: First variable
            Y: Second variable
            
        Returns:
            Mutual information estimate
        """
        # Joint distribution
        joint = torch.zeros(
            (X.max() + 1, Y.max() + 1),
            device=X.device
        )
        for x, y in zip(X, Y):
            joint[x, y] += 1
        joint = joint / joint.sum()
        
        # Marginal distributions
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)
        
        # Compute MI
        eps = 1e-10
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint[i,j] > eps:
                    mi += joint[i,j] * torch.log2(
                        joint[i,j] / (p_x[i] * p_y[j]) + eps
                    )
        
        return mi.item()
    
    def analyze_sequence_entropy(
        self,
        batch_size: int = 1000,
        seq_length: int = 100,
        temperature: float = 1.0,
        window_size: int = 2
    ) -> EntropyMetrics:
        """Perform comprehensive entropy analysis of generated sequences.
        
        Args:
            batch_size: Number of sequences to analyze
            seq_length: Length of sequences
            temperature: Generation temperature
            window_size: Window size for n-gram analysis
            
        Returns:
            EntropyMetrics containing various entropy measures
        """
        # Generate sequences
        sequences, mixtures = self.generator.sample_sequences(
            batch_size=batch_size,
            seq_length=seq_length,
            temperature=temperature
        )
        
        # Get transition matrices
        trans_matrices, _ = self.generator.generate_transitions(
            batch_size=batch_size,
            temperature=temperature,
            mixtures=mixtures
        )
        
        # Compute various entropy measures
        transition_entropy = self.compute_transition_entropy(
            trans_matrices
        ).mean().item()
        
        color_entropy = self.compute_color_entropy(
            sequences,
            window_size=window_size
        )
        
        # Empirical token entropy
        token_counts = torch.zeros(self.generator.vocab_size, device=sequences.device)
        for seq in sequences:
            unique, counts = torch.unique(seq, return_counts=True)
            token_counts[unique] += counts
        
        P = token_counts / token_counts.sum()
        token_entropy = -torch.sum(P * torch.log2(P + 1e-10)).item()
        
        # Topic mixture entropy
        topic_entropy = -torch.sum(
            mixtures.mean(0) * torch.log2(mixtures.mean(0) + 1e-10)
        ).item()
        
        # Mutual information estimates
        mutual_info = self.estimate_mutual_information(
            sequences, mixtures, n_bins=10
        )
        
        return EntropyMetrics(
            transition_entropy=transition_entropy,
            color_entropy=color_entropy,
            token_entropy=token_entropy,
            topic_entropy=topic_entropy,
            mutual_info=mutual_info
        )

# Example usage:
if __name__ == "__main__":
    from prob_color_gen import ProbColorConstrainedGenerator
    
    # Create a simple generator
    color_fractions = [0.3, 0.5, 0.2]
    color_transitions = torch.tensor([
        [1.0, 0.5, 0.1],
        [0.4, 1.0, 0.7],
        [0.2, 0.6, 1.0]
    ])
    
    generator = ProbColorConstrainedGenerator(
        n_topics=5,
        vocab_size=100,
        color_fractions=color_fractions,
        color_transitions=color_transitions
    )
    
    # Create analyzer
    analyzer = InfoTheoryAnalyzer(generator)
    
    # Analyze entropy at different temperatures
    for temp in [0.5, 1.0, 2.0]:
        metrics = analyzer.analyze_sequence_entropy(
            batch_size=1000,
            temperature=temp
        )
        print(f"\nTemperature {temp}:")
        print(f"Transition entropy: {metrics.transition_entropy:.3f}")
        print(f"Color entropy: {metrics.color_entropy:.3f}")
        print(f"Token entropy: {metrics.token_entropy:.3f}")
        print(f"Topic entropy: {metrics.topic_entropy:.3f}")
        print("Mutual information:")
        for k, v in metrics.mutual_info.items():
            print(f"  {k}: {v:.3f}")
