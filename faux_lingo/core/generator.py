# faux_lingo/core/generator.py
"""Sequence generator with constrained topic and color structure."""

from dataclasses import dataclass
from typing import TypeAlias

import torch
from typing_extensions import Self

from .transitions import TransitionMatrix
from .vocabulary import Vocabulary

# Type aliases for dimensions
BatchDim: TypeAlias = int
SeqLen: TypeAlias = int


@dataclass
class GeneratedSequences:
    """Container for generated sequences and their properties.

    Attributes:
        tokens: Generated token sequences [batch_size, seq_len]
        topic_mixtures: Topic mixtures used for generation [batch_size, n_topics]
        log_probs: Log probabilities of generated sequences [batch_size]
        latent_tokens: Optional latent transition sequences before decoding
    """

    tokens: torch.Tensor
    topic_mixtures: torch.Tensor
    log_probs: torch.Tensor
    latent_tokens: torch.Tensor | None = None


class SequenceGenerator:
    """
    Generates sequences using topic and color-constrained transitions.

    Core functionality:
    1. Sampling sequences from transition matrices
    2. Computing sequence probabilities
    3. Generating with specific topic mixtures or color constraints
    4. Decoding sequences through vocabulary hierarchy when provided
    """

    def __init__(
        self,
        transition_model: TransitionMatrix,
        device: str | None = None,
    ):
        """
        Initialize sequence generator.

        Args:
            transition_model: Model for generating transition matrices
            device: Optional compute device, defaults to CPU
        """
        self.device = device if device else "cpu"
        self.transition_model = transition_model
        self.vocabulary = transition_model.vocabulary

    def _pad_sequence(
        self,
        sequence: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Pad sequence to target length using padding token.

        Args:
            sequence: Input sequence to pad
            target_length: Desired sequence length

        Returns:
            Padded sequence

        Notes:
            If no padding token is defined, raises ValueError
        """
        current_length = sequence.shape[1]
        if current_length >= target_length:
            return sequence[:, :target_length]

        if (
            self.vocabulary.special_tokens is None
            or self.vocabulary.special_tokens.pad_token is None
        ):
            raise ValueError("Padding token not defined")

        padding = torch.full(
            (sequence.shape[0], target_length - current_length),
            self.vocabulary.special_tokens.pad_token,
            dtype=torch.long,
            device=self.device,
        )
        return torch.cat([sequence, padding], dim=1)
    
    def _add_special_tokens(
        self,
        sequence: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Add special tokens and padding to sequence.
    
        Args:
            sequence: Input sequence to modify
            target_length: Desired final sequence length
    
        Returns:
            Modified sequence with special tokens and padding
    
        Notes:
            - BOS token is added at start if specified
            - EOS token is added at end if specified
            - Padding is added as needed to reach target length
        """
        batch_size = sequence.shape[0]
        current_length = sequence.shape[1]
        special_tokens = self.vocabulary.special_tokens
    
        if special_tokens is None:
            return sequence
            
        # Start with original sequence
        result = sequence
    
        # Add BOS token if specified
        if special_tokens.bos_token is not None:
            bos = torch.full(
                (batch_size, 1),
                special_tokens.bos_token,
                dtype=torch.long,
                device=self.device
            )
            result = torch.cat([bos, result], dim=1)
            current_length += 1
    
        # Add EOS token if specified
        if special_tokens.eos_token is not None:
            eos = torch.full(
                (batch_size, 1),
                special_tokens.eos_token,
                dtype=torch.long,
                device=self.device
            )
            result = torch.cat([result, eos], dim=1)
            current_length += 1
    
        # Add padding if needed
        if current_length < target_length:
            if special_tokens.pad_token is None:
                raise ValueError("Padding token not defined")
                
            padding = torch.full(
                (batch_size, target_length - current_length),
                special_tokens.pad_token,
                dtype=torch.long,
                device=self.device
            )
            result = torch.cat([result, padding], dim=1)
        elif current_length > target_length:
            # Truncate if too long, preserving BOS/EOS if present
            if special_tokens.bos_token is not None:
                # Keep BOS token
                result = torch.cat([
                    result[:, :1],
                    result[:, 1:target_length-1],
                    result[:, -1:] if special_tokens.eos_token is not None else result[:, target_length-1:target_length]
                ], dim=1)
            else:
                # No BOS token
                if special_tokens.eos_token is not None:
                    result = torch.cat([
                        result[:, :target_length-1],
                        result[:, -1:]
                    ], dim=1)
                else:
                    result = result[:, :target_length]
    
        return result
    
    def generate(
        self,
        batch_size: int,
        seq_length: int,
        temperature: float = 1.0,
        topic_mixtures: torch.Tensor | None = None,
        start_tokens: torch.Tensor | None = None,
        min_prob: float = 1e-6,
        return_latent: bool = False,
    ) -> GeneratedSequences:
        """Generate batch of sequences.
    
        Args:
            batch_size: Number of sequences to generate
            seq_length: Desired length of final token sequences
            temperature: Controls randomness in sampling
            topic_mixtures: Optional pre-specified topic mixtures [batch_size, n_topics]
            start_tokens: Optional initial tokens [batch_size]
            min_prob: Minimum probability for valid transitions
            return_latent: Whether to return latent transition sequences
    
        Returns:
            GeneratedSequences containing tokens and properties
        """
        # Account for special tokens in target length
        special_tokens = self.vocabulary.special_tokens
        target_length = seq_length
        if special_tokens is not None:
            # Subtract space needed for BOS/EOS tokens
            if special_tokens.bos_token is not None:
                target_length -= 1
            if special_tokens.eos_token is not None:
                target_length -= 1
    
        # Compute required latent sequence length
        latent_length = (target_length if not self.vocabulary.has_hierarchy 
                        else self.vocabulary.hierarchy.compute_latent_length(target_length))
    
        # Get or generate topic mixtures
        if topic_mixtures is None:
            n_topics = self.transition_model.topic_space.n_topics
            topic_mixtures = torch.ones(batch_size, n_topics, device=self.device)
            topic_mixtures = topic_mixtures / n_topics
    
        # Validate topic mixture shape
        if topic_mixtures.shape[0] != batch_size:
            raise ValueError(
                f"Topic mixture batch size {topic_mixtures.shape[0]} "
                f"!= requested batch size {batch_size}"
            )
    
        # Generate transition matrix
        transitions = self.transition_model.generate(
            topic_mixtures,
            temperature=temperature,
            min_prob=min_prob,
        )
    
        # Initialize sequences
        latent_sequences = torch.zeros(
            (batch_size, latent_length), dtype=torch.long, device=self.device
        )
    
        # Initialize log probabilities
        log_probs = torch.zeros(batch_size, device=self.device)
    
        # Sample or use provided start tokens
        if start_tokens is not None:
            if start_tokens.shape != (batch_size,):
                raise ValueError(
                    f"Start tokens shape {start_tokens.shape} "
                    f"!= (batch_size={batch_size},)"
                )
            latent_sequences[:, 0] = start_tokens
        else:
            latent_sequences[:, 0] = torch.randint(
                0, self.vocabulary.base_vocab_size,
                (batch_size,),
                device=self.device
            )
    
        # Generate rest of sequences
        for t in range(1, latent_length):
            # Get transition probabilities for current tokens
            current_probs = transitions[
                torch.arange(batch_size, device=self.device),
                latent_sequences[:, t - 1],
            ]
    
            # Sample next tokens
            next_tokens = torch.multinomial(current_probs, 1).squeeze(-1)
            latent_sequences[:, t] = next_tokens
    
            # Update log probabilities
            log_probs += torch.log(
                torch.gather(
                    current_probs,
                    1,
                    next_tokens.unsqueeze(1),
                )
            ).squeeze(-1)
    
        # Decode sequences if hierarchy exists, otherwise use latent sequences
        tokens = latent_sequences
        if self.vocabulary.has_hierarchy:
            tokens = self.vocabulary.hierarchy.decode_sequence(
                latent_sequences,
                start_level=0,  # Most abstract level
                target_level=len(self.vocabulary.hierarchy),  # Most concrete level
            )
    
        # Add special tokens and padding
        tokens = self._add_special_tokens(tokens, seq_length)
    
        return GeneratedSequences(
            tokens=tokens,
            topic_mixtures=topic_mixtures,
            log_probs=log_probs,
            latent_tokens=latent_sequences if return_latent else None,
        )

    def generate_with_color(
        self,
        batch_size: int,
        seq_length: int,
        start_color: int,
        temperature: float = 1.0,
        topic_mixtures: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> GeneratedSequences:
        """
        Generate sequences starting with tokens of a specific color.

        Args:
            batch_size: Number of sequences to generate
            seq_length: Desired length of final token sequences
            start_color: Color index to start sequences with
            temperature: Controls randomness in sampling
            topic_mixtures: Optional pre-specified topic mixtures
            return_latent: Whether to return latent transition sequences

        Returns:
            GeneratedSequences with tokens starting from specified color
        """
        # Get token range for start color
        start_idx, end_idx = self.transition_model.color_space.get_color_range(
            start_color
        )

        # Sample start tokens from color range
        start_tokens = torch.randint(
            start_idx, end_idx, (batch_size,), device=self.device
        )

        return self.generate(
            batch_size=batch_size,
            seq_length=seq_length,
            temperature=temperature,
            topic_mixtures=topic_mixtures,
            start_tokens=start_tokens,
            return_latent=return_latent,
        )

    @classmethod
    def create_uniform(
        cls,
        vocabulary: Vocabulary,
        n_topics: int,
        color_fractions: list[float],
        device: str | None = None,
    ) -> Self:
        """
        Create generator with uniform topic and color distributions.

        Args:
            vocabulary: Vocabulary configuration
            n_topics: Number of topics
            color_fractions: Relative sizes of color classes
            device: Optional compute device

        Returns:
            SequenceGenerator with uniform parameters
        """
        transition_model = TransitionMatrix.create_uniform(
            vocabulary=vocabulary,
            n_topics=n_topics,
            color_fractions=color_fractions,
            device=device,
        )
        return cls(transition_model, device=device)
