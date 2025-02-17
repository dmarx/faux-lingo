import numpy as np
import random

##############################
# Generation of Colored Transition Matrices with Preferential Attachment
##############################

def assign_colors(word_vocab, num_colors):
    """
    Assign a random color (0 to num_colors-1) to each word in the vocabulary.
    Returns a dictionary mapping word index -> color.
    """
    return {i: random.randint(0, num_colors - 1) for i in range(len(word_vocab))}

def sample_sparse_transition_matrix_with_colors(vocab_size, avg_degree, word_colors, seed=None):
    """
    Sample a sparse row-stochastic transition matrix with a color constraint.
    For each row, select avg_degree distinct colors and then, for each chosen color,
    choose a random target word (column) having that color.
    """
    if seed is not None:
        np.random.seed(seed)
    T = np.zeros((vocab_size, vocab_size))
    
    # Build mapping: color -> list of words with that color.
    color_to_words = {}
    for word, color in word_colors.items():
        color_to_words.setdefault(color, []).append(word)
    
    available_colors = list(color_to_words.keys())
    if avg_degree > len(available_colors):
        raise ValueError("avg_degree cannot exceed the number of colors.")
    
    for i in range(vocab_size):
        # Select avg_degree distinct colors.
        selected_colors = random.sample(available_colors, avg_degree)
        chosen_words = []
        for c in selected_colors:
            chosen_words.append(random.choice(color_to_words[c]))
        # Assign random weights to the chosen words, then normalize.
        weights = np.random.rand(avg_degree)
        weights /= weights.sum()
        for idx, j in enumerate(chosen_words):
            T[i, j] = weights[idx]
    return T

def steady_state_distribution(T, tol=1e-8, max_iter=1000):
    """
    Compute the stationary distribution (left eigenvector) of the row-stochastic matrix T.
    Uses power iteration: pi = pi @ T until convergence.
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
    For each topic and each color, sample a set of mode nodes.
    Returns a list (length num_topics) of dictionaries. Each dictionary maps color -> set(mode word indices).
    """
    # Build mapping from color -> list of words.
    color_to_words = {}
    for word, color in word_colors.items():
        color_to_words.setdefault(color, []).append(word)
    
    topic_modes = []
    for t in range(num_topics):
        mode_dict = {}
        for c in range(num_colors):
            words_in_color = color_to_words.get(c, [])
            if len(words_in_color) < modes_per_color:
                raise ValueError(f"Not enough words with color {c} to sample {modes_per_color} mode nodes.")
            mode_nodes = random.sample(words_in_color, modes_per_color)
            mode_dict[c] = set(mode_nodes)
        topic_modes.append(mode_dict)
    return topic_modes

def sample_topic_transition_matrix_with_modes(T_background, word_colors, topic_mode_dict, attachment_bias):
    """
    Generate a topic-specific transition matrix by modifying T_background.
    For each row, for every nonzero entry at column j (with color c), if j is in the mode set for that color,
    boost the weight by multiplying by (1 + attachment_bias). Then renormalize the row.
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

def generate_topics_with_modes(num_topics, T_background, word_colors, topic_modes, attachment_bias):
    """
    For each topic, modify the background transition matrix using the topic’s mode nodes
    and compute its stationary distribution.
    Returns a list of stationary distributions (one per topic).
    """
    topics = []
    for t in range(num_topics):
        T_topic = sample_topic_transition_matrix_with_modes(T_background, word_colors, topic_modes[t], attachment_bias)
        pi_topic = steady_state_distribution(T_topic)
        topics.append(pi_topic)
    return topics

##############################
# Entropy and Information Density Utilities
##############################

def compute_entropy(prob_vector):
    """
    Compute the Shannon entropy (in bits) of a probability vector.
    Only considers nonzero entries.
    """
    p = np.array(prob_vector)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def row_entropy(row):
    """
    Compute the entropy of a probability distribution represented by a row (ignoring zero entries).
    """
    return compute_entropy(row)

def compute_average_conditional_entropy(T):
    """
    Compute the weighted average of row entropies for a transition matrix T.
    Weight each row's entropy by its stationary probability.
    """
    pi = steady_state_distribution(T)
    row_entropies = np.array([row_entropy(T[i]) for i in range(T.shape[0])])
    return np.dot(pi, row_entropies)

def compute_stationary_entropy(T):
    """
    Compute the entropy (in bits) of the stationary distribution of T.
    """
    pi = steady_state_distribution(T)
    return compute_entropy(pi)

def analyze_transition_matrix(T, vocab_size, tokens_per_word):
    """
    Compute a set of entropy measures for a given transition matrix T.
    
    Returns a dictionary containing:
      - max_entropy: log2(vocab_size) [bits/word]
      - stationary_entropy: entropy of the stationary distribution [bits/word]
      - conditional_entropy: average conditional entropy (weighted row entropies) [bits/word]
      - per_token_stationary: stationary entropy per token
      - per_token_conditional: conditional entropy per token
    """
    max_entropy = np.log2(vocab_size)
    stationary_entropy = compute_stationary_entropy(T)
    conditional_entropy = compute_average_conditional_entropy(T)
    per_token_stationary = stationary_entropy / tokens_per_word
    per_token_conditional = conditional_entropy / tokens_per_word
    return {
        "max_entropy": max_entropy,
        "stationary_entropy": stationary_entropy,
        "conditional_entropy": conditional_entropy,
        "per_token_stationary": per_token_stationary,
        "per_token_conditional": per_token_conditional
    }

def analyze_topic(T_background, word_colors, topic_mode_dict, attachment_bias, vocab_size, tokens_per_word):
    """
    For a given topic (specified by its mode nodes), generate its topic-specific transition matrix,
    and then return the analysis of its information density.
    """
    T_topic = sample_topic_transition_matrix_with_modes(T_background, word_colors, topic_mode_dict, attachment_bias)
    return analyze_transition_matrix(T_topic, vocab_size, tokens_per_word)

##############################
# Main: Generate Transition Matrices, Topics, and Analyze Information Density
##############################

def main():
    # Parameters for the transition matrix and preferential attachment.
    vocab_size = 100      # Total number of words
    avg_degree = 5        # Number of transitions per word (each row has avg_degree targets)
    num_colors = 5        # Number of colors (simulate parts-of-speech)
    num_topics = 3        # Number of topics
    modes_per_color = 2   # For each topic and each color, choose 2 mode words
    attachment_bias = 0.5 # Mode nodes get their weight boosted by (1 + attachment_bias)
    seed = 42

    # Hierarchical parameters for word construction.
    # (Assume words are built deterministically from lower-level symbols.)
    characters_per_word = 3     # C
    runes_per_character = 3     # D
    tokens_per_rune = 1         # E
    tokens_per_word = characters_per_word * runes_per_character * tokens_per_rune

    # For demonstration, we assume word_vocab is just indices 0 .. vocab_size-1.
    word_vocab = list(range(vocab_size))
    
    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Assign each word a random color.
    word_colors = assign_colors(word_vocab, num_colors)
    
    # 2. Sample the background sparse transition matrix with the color constraint.
    T_background = sample_sparse_transition_matrix_with_colors(vocab_size, avg_degree, word_colors, seed=seed)
    
    # 3. For each topic, for each color, choose mode words.
    topic_modes = sample_topic_modes(num_topics, word_colors, num_colors, modes_per_color)
    
    # 4. Analyze the background transition matrix.
    bg_analysis = analyze_transition_matrix(T_background, vocab_size, tokens_per_word)
    print("=== Background Transition Matrix Analysis ===")
    print(f"Max entropy (uniform): {bg_analysis['max_entropy']:.3f} bits/word")
    print(f"Stationary entropy: {bg_analysis['stationary_entropy']:.3f} bits/word")
    print(f"Conditional entropy: {bg_analysis['conditional_entropy']:.3f} bits/word")
    print(f"Per-token stationary entropy: {bg_analysis['per_token_stationary']:.3f} bits/token")
    print(f"Per-token conditional entropy: {bg_analysis['per_token_conditional']:.3f} bits/token")
    print("")

    # 5. Analyze each topic's transition matrix.
    for t in range(num_topics):
        topic_analysis = analyze_topic(T_background, word_colors, topic_modes[t],
                                       attachment_bias, vocab_size, tokens_per_word)
        print(f"--- Topic {t+1} Analysis ---")
        print(f"Stationary entropy: {topic_analysis['stationary_entropy']:.3f} bits/word")
        print(f"Conditional entropy: {topic_analysis['conditional_entropy']:.3f} bits/word")
        print(f"Per-token stationary entropy: {topic_analysis['per_token_stationary']:.3f} bits/token")
        print(f"Per-token conditional entropy: {topic_analysis['per_token_conditional']:.3f} bits/token")
        print("")
    
    # 6. (Optional) List color assignments and mode nodes for inspection.
    print("First 10 word color assignments:")
    for i in range(10):
        print(f"Word {i}: Color {word_colors[i]}")
    print("")
    
    print("Topic Modes (showing mode words per color):")
    for t, mode_dict in enumerate(topic_modes):
        print(f"Topic {t+1}:")
        for c in range(num_colors):
            print(f"  Color {c}: Modes {sorted(mode_dict[c])}")
        print("")


def compute_perplexity_from_entropy(entropy):
    """
    Given an entropy value (in bits), compute the perplexity.
    """
    return 2 ** entropy

def analyze_transition_matrix_with_perplexity(T, vocab_size, tokens_per_word):
    """
    Compute a set of entropy measures for a given transition matrix T
    and then infer perplexity.
    
    Returns a dictionary containing:
      - max_entropy: log2(vocab_size) [bits/word]
      - stationary_entropy: entropy of the stationary distribution [bits/word]
      - conditional_entropy: average conditional entropy (weighted row entropies) [bits/word]
      - per_token_stationary: stationary entropy per token
      - per_token_conditional: conditional entropy per token
      - perplexity_stationary: 2^(stationary_entropy)
      - perplexity_conditional: 2^(conditional_entropy)
    """
    analysis = analyze_transition_matrix(T, vocab_size, tokens_per_word)
    perplexity_stationary = compute_perplexity_from_entropy(analysis["stationary_entropy"])
    perplexity_conditional = compute_perplexity_from_entropy(analysis["conditional_entropy"])
    
    analysis.update({
        "perplexity_stationary": perplexity_stationary,
        "perplexity_conditional": perplexity_conditional
    })
    return analysis

# Example usage within our main analysis routine:
def main_perplexity_example():
    # (Use the same parameters as before)
    vocab_size = 100      # Total number of words
    avg_degree = 5        # Number of transitions per word
    tokens_per_word = 9   # e.g., 3 characters * 3 runes * 1 token per rune

    # We'll assume T_background has been generated by our earlier function.
    # For demonstration, generate a dummy T_background:
    word_colors = {i: random.randint(0, 4) for i in range(vocab_size)}
    T_background = sample_sparse_transition_matrix_with_colors(vocab_size, avg_degree, word_colors, seed=42)
    
    analysis = analyze_transition_matrix_with_perplexity(T_background, vocab_size, tokens_per_word)
    
    print("=== Transition Matrix Analysis with Perplexity ===")
    print(f"Max entropy (uniform): {analysis['max_entropy']:.3f} bits/word")
    print(f"Stationary entropy: {analysis['stationary_entropy']:.3f} bits/word")
    print(f"Conditional entropy: {analysis['conditional_entropy']:.3f} bits/word")
    print(f"Per-token stationary entropy: {analysis['per_token_stationary']:.3f} bits/token")
    print(f"Per-token conditional entropy: {analysis['per_token_conditional']:.3f} bits/token")
    print(f"Perplexity (stationary): {analysis['perplexity_stationary']:.3f}")
    print(f"Perplexity (conditional): {analysis['perplexity_conditional']:.3f}")


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    main_perplexity_example()
