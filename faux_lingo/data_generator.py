def generate_document_with_entropy(doc_length, artifacts, doc_topic_alpha,
                                   include_whitespace=True, include_bod=True, include_eod=True):
    num_topics = artifacts["num_topics"]
    topics = artifacts["topics"]      # stationary distributions for each topic
    T_topics = artifacts["T_topics"]    # corresponding transition matrices
    word_vocab = artifacts["word_vocab"]
    
    # 1. Sample document-level topic mixture.
    doc_topic_mixture = np.random.dirichlet([doc_topic_alpha] * num_topics)
    topic_idx = np.random.choice(num_topics, p=doc_topic_mixture)
    T_topic = T_topics[topic_idx]
    stationary = topics[topic_idx]
    
    vocab_size = len(word_vocab)
    # 2. Initialize by sampling the first word from the stationary distribution.
    current_word = np.random.choice(vocab_size, p=stationary)
    word_indices = [current_word]
    
    # Initialize entropy accumulator.
    total_neg_log_prob = 0.0
    # Include the probability for the first word.
    total_neg_log_prob += -np.log2(stationary[current_word] + 1e-12)
    
    # 3. Traverse the transition matrix to generate a chain of word indices.
    for _ in range(doc_length - 1):
        row = T_topic[current_word]
        # In case of a row that sums to zero, fallback to stationary distribution.
        if row.sum() == 0:
            p_dist = stationary
        else:
            p_dist = row
        # Sample the next word.
        next_word = np.random.choice(vocab_size, p=p_dist)
        word_indices.append(next_word)
        # Accumulate negative log probability.
        total_neg_log_prob += -np.log2(p_dist[next_word] + 1e-12)
        current_word = next_word
    
    # 4. Compute average entropy (bits per word).
    avg_entropy = total_neg_log_prob / doc_length
    perplexity = 2 ** avg_entropy

    # 5. Convert word indices to tokens.
    tokens = []
    WS_TOKEN = -3
    for i, word_idx in enumerate(word_indices):
        tokens.extend(word_vocab[word_idx])
        if include_whitespace and i < len(word_indices) - 1:
            tokens.append(WS_TOKEN)
    if include_bod:
        tokens.insert(0, -1)
    if include_eod:
        tokens.append(-2)
    
    # Return both the document tensor and the computed entropy/perplexity.
    doc_tensor = torch.tensor(tokens, dtype=torch.long)
    return doc_tensor, avg_entropy, perplexity
