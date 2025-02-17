import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random

def sample_color_transition_matrix(num_colors, sigma=1.0, epsilon=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    color_matrix = np.zeros((num_colors, num_colors))
    for i in range(num_colors):
        alphas = np.array([np.exp(-((i - j) ** 2) / (2 * sigma**2)) + epsilon for j in range(num_colors)])
        row = np.random.dirichlet(alphas)
        color_matrix[i] = row
    return color_matrix

def generate_token_vocab(token_vocab_size):
    return list(range(token_vocab_size))

def generate_rune_vocab(token_vocab, tokens_per_rune, rune_vocab_size):
    runes = set()
    rune_list = []
    while len(rune_list) < rune_vocab_size:
        candidate = tuple(random.choices(token_vocab, k=tokens_per_rune))
        if candidate not in runes:
            runes.add(candidate)
            rune_list.append(candidate)
    return rune_list

def generate_char_vocab(rune_vocab, runes_per_character, char_vocab_size):
    chars = set()
    char_list = []
    while len(char_list) < char_vocab_size:
        candidate = tuple(random.choices(rune_vocab, k=runes_per_character))
        if candidate not in chars:
            chars.add(candidate)
            char_list.append(candidate)
    return char_list

def flatten_character(character):
    tokens = []
    for rune in character:
        tokens.extend(rune)
    return tokens

def generate_word_vocab(char_vocab, characters_per_word, word_vocab_size):
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

def assign_colors(word_vocab, num_colors):
    return {i: random.randint(0, num_colors - 1) for i in range(len(word_vocab))}

def sample_sparse_transition_matrix_with_colors(vocab_size, avg_degree, word_colors,
                                                 color_transition_matrix=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    T = np.zeros((vocab_size, vocab_size))
    color_to_words = {}
    for word, color in word_colors.items():
        color_to_words.setdefault(color, []).append(word)
    num_colors = max(word_colors.values()) + 1
    if color_transition_matrix is None:
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
    n = T.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_next = pi @ T
        if np.linalg.norm(pi_next - pi, 1) < tol:
            return pi_next
        pi = pi_next
    return pi

def sample_topic_modes(num_topics, word_colors, num_colors, modes_per_color):
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
    T_topics = []
    topics = []
    for t in range(num_topics):
        T_topic = sample_topic_transition_matrix_with_modes(T_background, word_colors, topic_modes[t], attachment_bias)
        T_topics.append(T_topic)
        topics.append(steady_state_distribution(T_topic))
    return T_topics, topics

def generate_artifacts(
    num_topics,
    word_vocab_size,
    characters_per_word,
    runes_per_character,
    tokens_per_rune,
    token_vocab_size,
    rune_vocab_size,
    char_vocab_size,
    topic_word_alpha,  # For compatibility; not used directly here.
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
    token_vocab = generate_token_vocab(token_vocab_size)
    rune_vocab = generate_rune_vocab(token_vocab, tokens_per_rune, rune_vocab_size)
    char_vocab = generate_char_vocab(rune_vocab, runes_per_character, char_vocab_size)
    word_vocab = generate_word_vocab(char_vocab, characters_per_word, word_vocab_size)
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
        "topics": topics,
        "T_topics": T_topics,
        "num_topics": num_topics,
        "word_colors": word_colors,
        "T_background": T_background,
        "topic_modes": topic_modes,
        "color_transition_matrix": color_transition_matrix,
        "num_colors": num_colors,
        "avg_degree": avg_degree,
        "modes_per_color": modes_per_color,
        "attachment_bias": attachment_bias
    }
    return artifacts


class GenerativeCorpusDataset(torch.utils.data.Dataset):
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
        return generate_document(self.doc_length, self.artifacts, self.doc_topic_alpha,
                                 include_whitespace=self.include_whitespace,
                                 include_bod=self.include_bod,
                                 include_eod=self.include_eod)

def generate_document(doc_length, artifacts, doc_topic_alpha,
                      include_whitespace=True, include_bod=True, include_eod=True):
    num_topics = artifacts["num_topics"]
    topics = artifacts["topics"]
    T_topics = artifacts["T_topics"]
    word_vocab = artifacts["word_vocab"]
    doc_topic_mixture = np.random.dirichlet([doc_topic_alpha] * num_topics)
    topic_idx = np.random.choice(num_topics, p=doc_topic_mixture)
    T_topic = T_topics[topic_idx]
    stationary = topics[topic_idx]
    vocab_size = len(word_vocab)
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
    if include_bod:
        tokens.insert(0, -1)
    if include_eod:
        tokens.append(-2)
    return torch.tensor(tokens, dtype=torch.long)


class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.lstm(emb, hidden)
        logits = self.fc(output)
        return logits, hidden


def main():
    # --- Create Artifacts ---
    # We'll use default uniform color transitions (random_color_transition=False).
    seed = 42
    artifacts = generate_artifacts(
        num_topics=5,
        word_vocab_size=100,
        characters_per_word=3,
        runes_per_character=3,
        tokens_per_rune=1,
        token_vocab_size=10,   # our token vocab will be 0..9
        rune_vocab_size=30,
        char_vocab_size=20,
        topic_word_alpha=0.1,
        num_colors=5,
        avg_degree=5,
        modes_per_color=2,
        attachment_bias=0.5,
        random_color_transition=False,  # default: uniform transitions
        color_transition_matrix=None,
        sigma=1.0,
        epsilon=0.1,
        seed=seed
    )
    
    # --- Create Dataset and DataLoader ---
    doc_count = 1000
    doc_length = 50  # words per document
    doc_topic_alpha = 0.5
    dataset = GenerativeCorpusDataset(doc_count, doc_length, artifacts, doc_topic_alpha,
                                       include_whitespace=True, include_bod=True, include_eod=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # --- Define Simple Language Model ---
    # Our generative code produces tokens in range 0..9 from vocabulary, and special tokens -1, -2, -3.
    # We'll remap special tokens to additional indices: 
    #  -1 -> 10, -2 -> 11, -3 -> 12.
    base_vocab_size = 10  # from token_vocab_size
    num_special = 3
    overall_vocab_size = base_vocab_size + num_special  # 13
    model = SimpleLanguageModel(vocab_size=overall_vocab_size, embed_size=16, hidden_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    num_epochs = 5
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            # batch shape: (batch_size, seq_length)
            # Remap special tokens: 
            # if token < 0, add overall_vocab_size to it.
            inputs = batch[:, :-1].clone()
            targets = batch[:, 1:].clone()
            inputs[inputs < 0] = inputs[inputs < 0] + overall_vocab_size
            targets[targets < 0] = targets[targets < 0] + overall_vocab_size
            
            logits, _ = model(inputs)  # logits: (batch_size, seq_length-1, overall_vocab_size)
            loss = criterion(logits.view(-1, overall_vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {2**avg_loss:.2f}")
    
    print("Training complete.")

if __name__ == "__main__":
    main()
