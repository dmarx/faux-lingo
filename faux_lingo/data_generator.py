import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

##############################
# Utility: Sample a Color Transition Matrix
##############################

def sample_color_transition_matrix(num_colors, sigma=1.0, epsilon=0.1, seed=None):
    """
    Sample a color-to-color transition matrix from a Dirichlet distribution.
    
    For each source color i, we set the concentration parameter for target color j as:
      alpha_i[j] = exp(- (|i - j|^2) / (2 * sigma^2)) + epsilon.
    
    Returns a (num_colors x num_colors) numpy array where each row sums to 1.
    """
    if seed is not None:
        np.random.seed(seed)
    color_matrix = np.zeros((num_colors, num_colors))
    for i in range(num_colors):
        alphas = np.array([np.exp(-((i - j) ** 2) / (2 * sigma**2)) + epsilon for j in range(num_colors)])
        row = np.random.dirichlet(alphas)
        color_matrix[i] = row
    return color_matrix

##############################
# Hierarchical Vocabulary Functions
##############################

def generate_token_vocab(token_vocab_size):
    """Generate token vocabulary as a list of integers: 0,...,token_vocab_size-1."""
    return list(range(token_vocab_size))

def generate_rune_vocab(token_vocab, tokens_per_rune, rune_vocab_size):
    """Generate a fixed set of valid runes (tuples of tokens)."""
    runes = set()
    rune_list = []
    while len(rune_list) < rune_vocab_size:
        candidate = tuple(random.choices(token_vocab, k=tokens_per_rune))
        if candidate not in runes:
            runes.add(candidate)
            rune_list.append(candidate)
    return rune_list

def generate_char_vocab(rune_vocab, runes_per_character, char_vocab_size):
    """Generate a fixed set of valid characters (tuples of runes)."""
    chars = set()
    char_list = []
    while len(char_list) < char_vocab_size:
        candidate = tuple(random.choices(rune_vocab, k=runes_per_character))
        if candidate not in chars:
            chars.add(candidate)
            char_list.append(candidate)
    return char_list

def flatten_character(character):
    """Flatten a character (tuple of runes) into a list of tokens."""
    tokens = []
    for rune in character:
        tokens.extend(rune)
    return tokens

def generate_word_vocab(char_vocab, characters_per_word, word_vocab_size):
    """
    Generate a fixed vocabulary of valid words.
    Each word is built as a tuple of characters (of length characters_per_word)
    and then flattened into a tuple of tokens.
    """
    words = set()
    word_list = []
    while len(word_list) < word_vocab_size:
        candidate = tuple(random.choices(char_vocab, k=characters_per_word))
        if candidate not in words:
            words.add(candidate)
            word_tokens = []
            for character in candidate:
                word_tokens.extend(flatten_character(character))
            word_list.append(tuple(word_tokens))
    return word_list

##############################
# Graph-based Topic Generation Functions
##############################

def assign_colors(word_vocab, num_colors):
    """
    Assign each word a random color (from 0 to num_colors-1).
    Returns a dictionary mapping word index -> color.
    """
    return {i: random.randint(0, num_colors - 1) for i in range(len(word_vocab))}

def sample_sparse_transition_matrix_with_colors(vocab_size, avg_degree, word_colors,
                                                 color_transition_matrix=None, seed=None):
    """
    Sample a sparse, row-stochastic transition matrix with a color constraint.
    
    For each source word (row), use its color to weight the probability of transitioning
    to target colors via the provided `color_transition_matrix`.
    Then, sample `avg_degree` distinct target colors (using those probabilities)
    and, for each, randomly choose a target word with that color.
    
    If no color_transition_matrix is provided, a uniform distribution over colors is used.
    """
    if seed is not None:
        np.random.seed(seed)
    T = np.zeros((vocab_size, vocab_size))
    
    # Build mapping: color -> list of word indices.
    color_to_words = {}
    for word, color in word_colors.items():
        color_to_words.setdefault(color, []).append(word)
    
    num_colors = max(word_colors.values()) + 1
    if color_transition_matrix is None:
        # Use uniform distribution over colors.
        color_transition_matrix = np.full((num_colors, num_colors), 1.0/num_colors)
    else:
        color_transition_matrix = np.array(color_transition_matrix)
        color_transition_matrix = color_transition_matrix / color_transition_matrix.sum(axis=1, keepdims=True)
    
    for i in range(vocab_size):
        src_color = word_colors[i]
        p_colors = color_transition_matrix[src_color]
        available_colors = np.arange(num_colors)
        if avg_degree > num_colors:
            raise ValueError("avg_degree cannot exceed the number of colors.")
        # Sample distinct target colors according to p_colors.
        target_colors = np.random.choice(available_colors, size=avg_degree, replace=False, p=p_colors)
        chosen_words = []
        for c in target_colors:
            chosen_words.append(random.choice(color_to_words[c]))
        weights = np.random.rand(avg_degree)
        weights /= weights.sum()
        for idx, j in enumerate(chosen_words):
            T[i, j] = weights[idx]
    return T

def steady_state_distribution(T, tol=1e-8, max_iter=1000):
    """
    Compute the stationary distribution (left eigenvector) of the row-stochastic matrix T
    via power iteration.
    """
    n = T.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_next = pi @ T
        if np.linalg.norm(pi_next - pi, 1) < tol:
            return pi_next
        pi = pi_next
    return pi

def sample_topic_modes(num_topics, word_colors, num_colors, modes_per_color):
    """
    For each topic and for each color, sample a set of mode words.
    Returns a list (length num_topics) of dictionaries mapping color -> set(mode word indices).
    """
    color_to_words = {}
    for word, color in word_colors.items():
        color_to_words.setdefault(color, []).append(word)
    
    topic_modes = []
    for _ in range(num_topics):
        mode_dict = {}
        for c in range(num_colors):
            words_in_color = color_to_words.get(c, [])
            if len(words_in_color) < modes_per_color:
                raise ValueError(f"Not enough words with color {c} to sample {modes_per_color} modes.")
            mode_nodes = random.sample(words_in_color, modes_per_color)
            mode_dict[c] = set(mode_nodes)
        topic_modes.append(mode_dict)
    return topic_modes

def sample_topic_transition_matrix_with_modes(T_background, word_colors, topic_mode_dict, attachment_bias):
    """
    Generate a topic-specific transition matrix by boosting transitions toward mode words.
    
    For each nonzero entry T[i,j] in T_background, if word j (with color c) is in the mode set,
    multiply its weight by (1 + attachment_bias) and renormalize the row.
    """
    T_topic = T_background.copy()
    vocab_size = T_topic.shape[0]
    for i in range(vocab_size):
        row = T_topic[i].copy()
        indices = np.nonzero(row)[0]
        for j in indices:
            c = word_colors[j]
            if j in topic_mode_dict.get(c, set()):
                row[j] *= (1 + attachment_bias)
        if row.sum() > 0:
            row /= row.sum()
        T_topic[i] = row
    return T_topic

def generate_topic_transition_matrices_with_modes(num_topics, T_background, word_colors, topic_modes, attachment_bias):
    """
    For each topic, generate its topic-specific transition matrix (via preferential attachment)
    and compute its stationary distribution.
    
    Returns a tuple (T_topics, topics) where:
      - T_topics: list of topic-specific transition matrices.
      - topics: list of stationary distributions (one per topic).
    """
    T_topics = []
    topics = []
    for t in range(num_topics):
        T_topic = sample_topic_transition_matrix_with_modes(T_background, word_colors, topic_modes[t], attachment_bias)
        T_topics.append(T_topic)
        topics.append(steady_state_distribution(T_topic))
    return T_topics, topics

##############################
# Artifact Generation
##############################

def generate_artifacts(
    num_topics,
    word_vocab_size,
    characters_per_word,
    runes_per_character,
    tokens_per_rune,
    token_vocab_size,
    rune_vocab_size,
    char_vocab_size,
    topic_word_alpha,   # (Kept for compatibility; topics now come from graphs)
    num_colors,
    avg_degree,
    modes_per_color,
    attachment_bias,
    random_color_transition=False,
    color_transition_matrix=None,
    sigma=1.0,
    epsilon=0.1,
    seed=None
):
    """
    Generate all fixed artifacts for the corpus.
    
    This includes:
      - Hierarchical vocabularies,
      - Word colors,
      - A background transition matrix (with color-to-color probabilities),
      - Topic mode words,
      - And the topic-specific transition matrices (with their stationary distributions).
    
    If a user-provided color_transition_matrix is given, it is used (and takes priority).
    Otherwise, if random_color_transition is True, a random color transition matrix is sampled
    using `sample_color_transition_matrix`. If False, a uniform matrix is used.
    """
    # Hierarchical vocabularies.
    token_vocab = generate_token_vocab(token_vocab_size)
    rune_vocab = generate_rune_vocab(token_vocab, tokens_per_rune, rune_vocab_size)
    char_vocab = generate_char_vocab(rune_vocab, runes_per_character, char_vocab_size)
    word_vocab = generate_word_vocab(char_vocab, characters_per_word, word_vocab_size)
    
    # Graph-related artifacts.
    word_colors = assign_colors(word_vocab, num_colors)
    if color_transition_matrix is None:
        if random_color_transition:
            color_transition_matrix = sample_color_transition_matrix(num_colors, sigma=sigma, epsilon=epsilon, seed=seed)
        else:
            color_transition_matrix = np.full((num_colors, num_colors), 1.0/num_colors)
    T_background = sample_sparse_transition_matrix_with_colors(word_vocab_size, avg_degree, word_colors, color_transition_matrix, seed=seed)
    topic_modes = sample_topic_modes(num_topics, word_colors, num_colors, modes_per_color)
    T_topics, topics = generate_topic_transition_matrices_with_modes(num_topics, T_background, word_colors, topic_modes, attachment_bias)
    
    artifacts = {
        "token_vocab": token_vocab,
        "rune_vocab": rune_vocab,
        "char_vocab": char_vocab,
        "word_vocab": word_vocab,
        "topics": topics,         # List of stationary distributions (one per topic)
        "T_topics": T_topics,     # List of topic-specific transition matrices
        "num_topics": num_topics,
        "word_colors": word_colors,
        "T_background": T_background,
        "topic_modes": topic_modes,
        "color_transition_matrix": color_transition_matrix,
        # Also store graph-generation parameters for reference.
        "num_colors": num_colors,
        "avg_degree": avg_degree,
        "modes_per_color": modes_per_color,
        "attachment_bias": attachment_bias
    }
    return artifacts

##############################
# Document Generation via Transition Traversal
##############################

def generate_document(doc_length, artifacts, doc_topic_alpha,
                      include_whitespace=True, include_bod=True, include_eod=True):
    """
    Generate one document as a PyTorch tensor of tokens by traversing a topic's transition matrix.
    
    Procedure:
      1. Sample a document-level topic mixture (Dirichlet with parameter doc_topic_alpha).
      2. Select one topic for the document.
      3. For that topic, traverse its transition matrix (T_topic) to generate a Markov chain:
         - Sample the first word from the topic's stationary distribution.
         - Then, for each subsequent word, sample the next word using the current word's row in T_topic.
      4. Convert the sequence of word indices into a flattened sequence of tokens using the word vocabulary.
      5. Optionally, insert WS_TOKEN between words and add BOD/EOD markers.
    """
    num_topics = artifacts["num_topics"]
    topics = artifacts["topics"]      # Stationary distributions for each topic
    T_topics = artifacts["T_topics"]    # Corresponding transition matrices
    word_vocab = artifacts["word_vocab"]
    
    # 1. Sample document-level topic mixture.
    doc_topic_mixture = np.random.dirichlet([doc_topic_alpha] * num_topics)
    # 2. Select one topic for the document.
    topic_idx = np.random.choice(num_topics, p=doc_topic_mixture)
    T_topic = T_topics[topic_idx]
    stationary = topics[topic_idx]
    
    vocab_size = len(word_vocab)
    # 3. Traverse the transition matrix to generate a chain of word indices.
    current_word = np.random.choice(vocab_size, p=stationary)
    word_indices = [current_word]
    for _ in range(doc_length - 1):
        row = T_topic[current_word]
        if row.sum() == 0:
            next_word = np.random.choice(vocab_size, p=stationary)
        else:
            next_word = np.random.choice(vocab_size, p=row)
        word_indices.append(next_word)
        current_word = next_word
    
    # 4. Convert word indices into a sequence of tokens.
    tokens = []
    WS_TOKEN = -3
    if include_whitespace:
        for i, word_idx in enumerate(word_indices):
            tokens.extend(word_vocab[word_idx])
            if i < len(word_indices) - 1:
                tokens.append(WS_TOKEN)
    else:
        for word_idx in word_indices:
            tokens.extend(word_vocab[word_idx])
    
    # 5. Add BOD and EOD tokens if desired.
    if include_bod:
        tokens.insert(0, -1)
    if include_eod:
        tokens.append(-2)
    return torch.tensor(tokens, dtype=torch.long)

##############################
# PyTorch Dataset Wrapper
##############################

class GenerativeCorpusDataset(Dataset):
    """
    A PyTorch Dataset that generates documents on the fly using our richer model.
    Documents are generated by traversing the topic-specific transition matrices.
    """
    def __init__(self, doc_count, doc_length, artifacts, doc_topic_alpha,
                 include_whitespace=True, include_bod=True, include_eod=True):
        self.doc_count = doc_count
        self.doc_length = doc_length
        self.artifacts = artifacts
        self.doc_topic_alpha = doc_topic_alpha
        self.include_whitespace = include_whitespace
        self.include_bod = include_bod
        self.include_eod = include_eod

    def __len__(self):
        return self.doc_count

    def __getitem__(self, idx):
        return generate_document(
            self.doc_length,
            self.artifacts,
            self.doc_topic_alpha,
            include_whitespace=self.include_whitespace,
            include_bod=self.include_bod,
            include_eod=self.include_eod
        )

##############################
# Example Usage with DataLoader
##############################

def create_artifacts_for_dataset():
    # Parameter choices (example values, tune as needed).
    num_topics = 5              # Number of topics.
    word_vocab_size = 100       # Total number of words.
    characters_per_word = 3     # Characters per word.
    runes_per_character = 3     # Runes per character.
    tokens_per_rune = 1         # Tokens per rune.
    token_vocab_size = 10       # Size of token vocabulary.
    rune_vocab_size = 30        # Number of valid runes.
    char_vocab_size = 20        # Number of valid characters.
    topic_word_alpha = 0.1      # (Not directly used now.)
    
    # Graph-based parameters.
    num_colors = 5              # Simulated parts-of-speech.
    avg_degree = 5              # Average out-degree for transition matrix.
    modes_per_color = 2         # Mode words per color for each topic.
    attachment_bias = 0.5       # Boost factor for mode transitions.
    seed = 42                   # For reproducibility.
    
    # Option to sample a random color transition matrix.
    # Set random_color_transition to False for uniform transitions.
    random_color_transition = False
    # If random_color_transition is True and no matrix is provided, a random matrix is sampled.
    color_transition_matrix = None
    
    artifacts = generate_artifacts(
        num_topics,
        word_vocab_size,
        characters_per_word,
        runes_per_character,
        tokens_per_rune,
        token_vocab_size,
        rune_vocab_size,
        char_vocab_size,
        topic_word_alpha,
        num_colors,
        avg_degree,
        modes_per_color,
        attachment_bias,
        random_color_transition=random_color_transition,
        color_transition_matrix=color_transition_matrix,
        sigma=1.0,
        epsilon=0.1,
        seed=seed
    )
    return artifacts

def main():
    # Create fixed artifacts once.
    artifacts = create_artifacts_for_dataset()
    
    # Dataset parameters.
    doc_count = 1000          # Total number of documents.
    doc_length = 50           # Number of words per document.
    doc_topic_alpha = 0.5     # Document-level topic Dirichlet concentration.
    
    dataset = GenerativeCorpusDataset(
        doc_count=doc_count,
        doc_length=doc_length,
        artifacts=artifacts,
        doc_topic_alpha=doc_topic_alpha,
        include_whitespace=True,
        include_bod=True,
        include_eod=True
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    for batch in dataloader:
        print("Batch shape:", batch.shape)  # (batch_size, sequence_length)
        print("Sample batch:", batch)
        break

if __name__ == "__main__":
    main()
