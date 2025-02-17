# faux_lingo/utils/entropy.py

"""
Utilities for computing various entropy measures and information-theoretic quantities.
"""

from typing import Tuple

import numpy as np
from loguru import logger


def compute_entropy(prob_vector: np.ndarray, eps: float = 1e-12) -> np.float64:
    """
    Compute the Shannon entropy (in bits) of a probability vector.
    Only considers nonzero entries.

    Args:
        prob_vector: Numpy array of probabilities
        eps: Small constant to prevent log(0)

    Returns:
        Entropy in bits
    """
    p = np.array(prob_vector)
    p = p[p > eps]
    return -np.sum(p * np.log2(p + eps))


def compute_cross_entropy(
    p: np.ndarray, q: np.ndarray, eps: float = 1e-12
) -> np.float64:
    """
    Compute the cross entropy H(p,q) = -sum(p_i * log(q_i)) between two
    probability distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Small constant to prevent log(0)

    Returns:
        Cross entropy in bits
    """
    if len(p) != len(q):
        raise ValueError("Probability distributions must have same length")
    return -np.sum(p * np.log2(q + eps))


def compute_kl_divergence(
    p: np.ndarray, q: np.ndarray, eps: float = 1e-12
) -> np.float64:
    """
    Compute the Kullback-Leibler divergence D(p||q) between two probability
    distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Small constant to prevent log(0)

    Returns:
        KL divergence in bits
    """
    if len(p) != len(q):
        raise ValueError("Probability distributions must have same length")
    p = p + eps
    q = q + eps
    return np.sum(p * np.log2(p / q))


def analyze_transition_matrix(
    T: np.ndarray, tokens_per_word: int, eps: float = 1e-12
) -> dict:
    """
    Compute various entropy measures for a transition matrix.

    Args:
        T: Row-stochastic transition matrix
        tokens_per_word: Number of tokens per word (for per-token normalization)
        eps: Small constant to prevent log(0)

    Returns:
        Dictionary containing:
            - stationary_entropy: Entropy of steady state distribution
            - conditional_entropy: Average row entropy
            - perplexity: 2^(entropy)
            - normalized_entropies: Per-token versions of above
    """
    n = T.shape[0]

    # Compute steady state distribution using power iteration
    pi = np.ones(n) / n
    for _ in range(1000):
        pi_next = pi @ T
        if np.linalg.norm(pi_next - pi, 1) < eps:
            break
        pi = pi_next

    # Compute stationary entropy
    stationary_entropy = compute_entropy(pi, eps)

    # Compute conditional entropy (average row entropy)
    row_entropies = np.zeros(n)
    for i in range(n):
        row = T[i]
        if row.sum() > eps:  # Only compute for rows with transitions
            row_entropies[i] = compute_entropy(row, eps)
    conditional_entropy = np.dot(pi, row_entropies)

    # Compute perplexity
    perplexity = 2.0**stationary_entropy
    conditional_perplexity = 2.0**conditional_entropy

    # Normalize by tokens per word
    normalized = {
        "per_token_stationary": stationary_entropy / tokens_per_word,
        "per_token_conditional": conditional_entropy / tokens_per_word,
        "per_token_perplexity": 2.0 ** (stationary_entropy / tokens_per_word),
        "per_token_conditional_perplexity": 2.0
        ** (conditional_entropy / tokens_per_word),
    }

    return {
        "stationary_entropy": stationary_entropy,
        "conditional_entropy": conditional_entropy,
        "perplexity": perplexity,
        "conditional_perplexity": conditional_perplexity,
        "normalized": normalized,
    }


def compute_topic_diversity(
    topic_distributions: list[np.ndarray], eps: float = 1e-12
) -> Tuple[np.float64, np.ndarray]:
    """
    Compute diversity measures between topics using KL divergence.

    Args:
        topic_distributions: List of topic distributions
        eps: Small constant to prevent log(0)

    Returns:
        Tuple of:
            - average_kl: Mean KL divergence between all topic pairs
            - kl_matrix: Matrix of pairwise KL divergences
    """
    n_topics = len(topic_distributions)
    kl_matrix = np.zeros((n_topics, n_topics))

    # Compute pairwise KL divergences
    for i in range(n_topics):
        for j in range(n_topics):
            if i != j:
                kl_matrix[i, j] = compute_kl_divergence(
                    topic_distributions[i], topic_distributions[j], eps
                )

    # Compute average KL divergence (excluding diagonal)
    avg_kl = np.sum(kl_matrix) / (n_topics * (n_topics - 1))

    return avg_kl, kl_matrix
