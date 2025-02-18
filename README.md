# FauxLingo Documentation

## User Guide

### 1. Introduction

FauxLingo is a powerful toolkit for generating structured token sequences with controlled properties. It combines topic-based generation with color-based constraints to produce sequences that exhibit specific statistical and structural patterns.

#### Key Features and Capabilities
- Hierarchical vocabulary system with multiple abstraction levels
- Topic-based sequence generation with controlled mixing
- Color-based constraints for structural patterns
- Information-theoretic analysis tools
- Extensible augmentation system for sequence variants
- Robust serialization for long-running generations

#### Common Use Cases
- Generating synthetic training data with controlled properties
- Testing sequence processing systems
- Studying information-theoretic properties of structured sequences
- Prototyping language models with controlled vocabularies

### 2. Getting Started

#### Basic Concepts

##### Tokens and Vocabularies
FauxLingo organizes tokens into hierarchical vocabularies. A token is the basic unit of sequence generation, and tokens can be organized into higher-level structures. For example:
```python
# Create a simple word-level hierarchy
hierarchy = create_word_hierarchy(
    token_vocab_size=10,  # Base tokens (0-9)
    n_chars=20,          # Number of character-level tokens
    n_words=100,         # Number of word-level tokens
    chars_per_word=3     # Characters per word
)
```

##### Colors and Topics
- Colors are classes that partition the token vocabulary and control transition patterns
- Topics are basis vectors that define token distributions
- Together they create structured yet flexible sequence generation

##### Sequence Generation
Basic sequence generation combines topic mixtures with color constraints:
```python
generator = SequenceGenerator.create_uniform(
    vocab_size=100,
    n_topics=3,
    color_fractions=[0.3, 0.3, 0.4]
)

sequences = generator.generate(
    batch_size=32,
    seq_length=20,
    temperature=1.0
)
```

#### Quick Start Tutorial
1. Install FauxLingo:
   ```bash
   pip install fauxlingo
   ```

2. Create a simple generator:
   ```python
   from faux_lingo import SequenceGenerator
   
   generator = SequenceGenerator.create_uniform(
       vocab_size=10,
       n_topics=2,
       color_fractions=[0.5, 0.5]
   )
   ```

3. Generate sequences:
   ```python
   sequences = generator.generate(
       batch_size=4,
       seq_length=10
   )
   ```

4. Analyze results:
   ```python
   from faux_lingo.analysis import EntropyAnalyzer
   
   analyzer = EntropyAnalyzer(generator.transition_model)
   metrics = analyzer.analyze_sequences(sequences)
   print(f"Token entropy: {metrics.token_entropy:.2f}")
   ```

#### Common Use Patterns
- Generate -> Analyze -> Adjust -> Repeat
- Build vocabulary -> Generate sequences -> Export
- Load existing -> Augment -> Generate variants

### 3. Core Components

#### 3.1 Vocabulary System

##### Understanding Vocabulary Hierarchy
The vocabulary system organizes tokens into levels:
```
Level 2 (Words):    [Word_0] [Word_1] ...
                      ↓         ↓
Level 1 (Chars):  [C_0,C_1] [C_2,C_3] ...
                    ↓   ↓     ↓   ↓
Level 0 (Tokens): [0,1,2] [3,4,5] ...
```

##### Creating Custom Vocabularies
```python
from faux_lingo.core.vocab_builder import BuilderConfig, VocabBuilder

config = BuilderConfig(
    token_vocab_size=10,
    sequence_lengths=[2, 3],  # Length at each level
    vocab_sizes=[20, 30]      # Size of each level
)

builder = VocabBuilder(config)
hierarchy = builder.build()
```

##### Multiple Mappings and Variants
Support for multiple sequences mapping to the same token:
```python
from faux_lingo.core.vocab_extensions import (
    AugmentationConfig,
    SequenceAugmenter,
    convert_to_multi_mapping
)

augmenter = SequenceAugmenter(
    vocab_size=10,
    config=AugmentationConfig(
        deletion_prob=0.1,
        insertion_prob=0.1
    )
)

multi_hierarchy = convert_to_multi_mapping(
    hierarchy,
    augmenter=augmenter,
    n_variants=3
)
```

##### Best Practices for Vocabulary Design
- Keep hierarchy levels focused and logical
- Balance vocabulary sizes between levels
- Consider sequence length impact on combinations
- Use meaningful partitioning for colors

#### 3.2 Generation System

##### Basic Sequence Generation
```python
# Simple generation with default settings
sequences = generator.generate(
    batch_size=32,
    seq_length=20
)

# Control randomness with temperature
sequences = generator.generate(
    batch_size=32,
    seq_length=20,
    temperature=0.8  # Lower = more deterministic
)
```

##### Topic-based Generation
```python
# Generate with specific topic mixture
topic_mixture = torch.tensor([[0.7, 0.3]])  # Favor first topic
sequences = generator.generate(
    batch_size=1,
    seq_length=20,
    topic_mixtures=topic_mixture
)
```

##### Color Constraints
```python
# Generate sequences starting with specific color
sequences = generator.generate_with_color(
    batch_size=32,
    seq_length=20,
    start_color=1  # Start with second color class
)
```

#### 3.3 Analysis Tools

##### Entropy Metrics
```python
analyzer = EntropyAnalyzer(generator.transition_model)
metrics = analyzer.analyze_sequences(sequences)

print(f"Color entropy: {metrics.color_entropy:.2f}")
print(f"Topic entropy: {metrics.topic_entropy:.2f}")
print(f"Token entropy: {metrics.token_entropy:.2f}")
```

##### Sequence Analysis
```python
# Get color sequences
dataset = SequenceDataset(generator, config)
color_seqs = dataset.get_color_sequences(sequences.tokens)

# Get batch statistics
stats = dataset.get_batch_stats(sequences)
print(f"Mean log probability: {stats['mean_log_prob']:.2f}")
print(f"Color distribution: {stats['color_counts']}")
```

### 4. Advanced Usage

#### Configuration System
```python
from faux_lingo.core.serialization import GenerationMetadata
from omegaconf import OmegaConf

# Create configuration
config = OmegaConf.create({
    "generation": {
        "batch_size": 32,
        "seq_length": 20,
        "temperature": 0.8
    },
    "vocab": {
        "token_size": 100,
        "n_topics": 3,
        "color_fractions": [0.3, 0.3, 0.4]
    }
})

# Create metadata container
metadata = GenerationMetadata(
    config=config,
    vocab_hierarchy=hierarchy,
    transition_model=generator.transition_model
)
```

#### Serialization and State Management
```python
# Save state
metadata.save(Path("generation_state"))

# Load state
metadata = GenerationMetadata.load(
    Path("generation_state"),
    device="cuda"
)
```

#### Augmentation System
```python
config = AugmentationConfig(
    deletion_prob=0.1,
    insertion_prob=0.1,
    substitution_prob=0.1,
    transposition_prob=0.1
)

augmenter = SequenceAugmenter(
    vocab_size=100,
    config=config
)

# Augment sequence
sequence = (0, 1, 2, 3)
variant = augmenter.augment_sequence(sequence)
```

### 5. Examples

#### Basic Examples

##### Simple Vocabulary Creation
```python
# Create word-level hierarchy
hierarchy = create_word_hierarchy(
    token_vocab_size=10,
    n_chars=20,
    n_words=100,
    chars_per_word=3
)
```

##### Basic Generation
```python
# Create generator
generator = SequenceGenerator.create_uniform(
    vocab_size=100,
    n_topics=3,
    color_fractions=[0.3, 0.3, 0.4]
)

# Generate sequences
sequences = generator.generate(
    batch_size=32,
    seq_length=20
)
```

##### Analysis Examples
```python
# Analyze entropy
analyzer = EntropyAnalyzer(generator.transition_model)
metrics = analyzer.analyze_sequences(sequences)

# Get batch statistics
dataset = SequenceDataset(generator, config)
stats = dataset.get_batch_stats(sequences)
```

#### Advanced Examples

##### Complex Hierarchies
```python
# Multi-level hierarchy with custom configuration
config = BuilderConfig(
    token_vocab_size=10,
    sequence_lengths=[2, 3, 2],  # Three levels
    vocab_sizes=[20, 30, 15]     # Sizes at each level
)

builder = VocabBuilder(config)
hierarchy = builder.build()
```

##### Custom Constraints
```python
# Create custom color transition weights
weights = torch.tensor([
    [1.0, 0.5, 0.0],  # Color 0 transitions
    [0.5, 1.0, 0.5],  # Color 1 transitions
    [0.0, 0.5, 1.0]   # Color 2 transitions
])

color_space = ColorSpace(
    color_fractions=[0.3, 0.3, 0.4],
    vocab_size=100,
    transition_weights=weights
)
```

##### Integration Examples
```python
# Integrate with training loop
for epoch in range(n_epochs):
    for batch in SequenceDataset(generator, config):
        # Process batch
        tokens = batch.tokens
        topic_mixtures = batch.topic_mixtures
        log_probs = batch.log_probs
        
        # Training step
        loss = model(tokens, topic_mixtures)
        loss.backward()
```

### 6. Troubleshooting

#### Common Error Messages

1. "Vocab size mismatch":
   - Check that vocabulary sizes match between components
   - Verify color space and topic space use same vocab size

2. "Invalid topic mixture":
   - Ensure topic mixtures sum to 1.0
   - Check batch dimension matches requested batch size

3. "Transition weights shape mismatch":
   - Verify transition weight matrix matches number of colors
   - Check that weights are non-negative

4. "Sequence probabilities do not sum to 1":
   - In multi-mapping, check variant probabilities sum to 1.0
   - Verify normalization in transition matrices

For additional support and resources:
- Check the GitHub repository
- Review test cases for examples
- File issues for bugs or feature requests

---

# Appendix H: Mathematics

## H.1 Probability Models

### Topic Space
The topic space is constructed using orthonormal vectors that form a basis for generating token distributions. For a vocabulary of size V and T topics:

1. **Topic Vectors**: Each topic i is represented by a unit vector vi ∈ ℝᵛ where:
   - ‖vi‖₂ = 1 (unit length)
   - vi · vj = 0 for i ≠ j (orthogonality)

2. **Topic Mixtures**: A topic mixture w = (w₁, ..., wₜ) satisfies:
   - wᵢ ≥ 0 for all i
   - Σwᵢ = 1

3. **Token Distribution**: For a topic mixture w, the base token distribution p is:
   p = Σ(wᵢvᵢ) for i = 1 to T

### Color-Constrained Transitions

1. **Color Classes**: The vocabulary is partitioned into C color classes where:
   - Each token belongs to exactly one color
   - Color c contains nc tokens
   - Σnc = V (total vocabulary size)

2. **Transition Matrix**: For a token distribution p, the transition matrix P is:
   P(j|i) = p(j) * M(c(i),c(j)) / Z(i)
   where:
   - c(i) is the color of token i
   - M is the color transition weight matrix
   - Z(i) is the normalization factor

3. **Normalization**: Z(i) ensures each row sums to 1:
   Z(i) = Σ(p(j) * M(c(i),c(j))) for all j

### Temperature Scaling
Temperature T modifies the transition probabilities:
P'(j|i) = P(j|i)^(1/T) / Z'(i)
where Z'(i) is the new normalization factor.

## H.2 Entropy Calculations

### Color Entropy
The empirical entropy of color transitions is:
H(C) = -Σ P(j|i) log₂ P(j|i)
averaged over all color pairs (i,j).

### Topic Entropy
For a batch of topic mixtures {wᵦ}, the entropy is:
H(T) = -Σ w̄ᵢ log₂ w̄ᵢ
where w̄ᵢ is the mean weight for topic i.

### Token Entropy
The empirical entropy of token sequences:
H(V) = -Σ f(v) log₂ f(v)
where f(v) is the observed frequency of token v.

## H.3 Sequence Generation

### Sampling Process
1. Given a topic mixture w and current token i:
   a. Compute base distribution p = Σ(wᵢvᵢ)
   b. Apply color constraints to get P(j|i)
   c. Apply temperature scaling
   d. Sample next token from resulting distribution

2. Batch Generation:
   - Independent samples for each sequence
   - Shared topic mixture within batch
   - Parallel computation of transitions

# Appendix B: Configuration Reference

## B.1 Available Settings

### Generation Configuration
```yaml
generation:
  batch_size: 32          # Number of sequences per batch
  seq_length: 20          # Length of each sequence
  temperature: 1.0        # Sampling temperature (default: 1.0)
  min_prob: 1e-6         # Minimum transition probability
  device: "cuda"         # Compute device (optional)
```

### Vocabulary Configuration
```yaml
vocab:
  token_vocab_size: 100   # Base vocabulary size
  sequence_lengths:       # Length of sequences at each level
    - 2                  # Level 1
    - 3                  # Level 2
  vocab_sizes:           # Vocabulary size at each level
    - 20                # Level 1
    - 30                # Level 2
  chunk_sizes:           # Tokens per chunk at each level
    - 2                 # Level 1
    - 3                 # Level 2
```

### Topic Configuration
```yaml
topics:
  n_topics: 3            # Number of topic vectors
  init_method: "random"  # Initialization method
  orthogonalize: true    # Force orthogonality
```

### Color Configuration
```yaml
colors:
  fractions:             # Relative size of color classes
    - 0.3               # Color 1
    - 0.3               # Color 2
    - 0.4               # Color 3
  transition_weights:    # Optional color transition matrix
    - [1.0, 0.5, 0.0]  # Color 1 transitions
    - [0.5, 1.0, 0.5]  # Color 2 transitions
    - [0.0, 0.5, 1.0]  # Color 3 transitions
```

### Augmentation Configuration
```yaml
augmentation:
  deletion_prob: 0.05    # Character deletion probability
  insertion_prob: 0.05   # Character insertion probability
  substitution_prob: 0.05 # Character substitution probability
  transposition_prob: 0.05 # Character transposition probability
  seed: null             # Random seed (optional)
```

## B.2 Default Values

### Core Defaults
```yaml
generation:
  batch_size: 32
  seq_length: 20
  temperature: 1.0
  min_prob: 1e-6
  device: "cpu"

vocab:
  token_vocab_size: 10
  sequence_lengths: [2, 3]
  vocab_sizes: [20, 15]
  chunk_sizes: [2, 3]

topics:
  n_topics: 2
  init_method: "random"
  orthogonalize: true

colors:
  fractions: [0.5, 0.5]
  transition_weights: null  # Defaults to all-ones matrix
```

### Augmentation Defaults
```yaml
augmentation:
  deletion_prob: 0.05
  insertion_prob: 0.05
  substitution_prob: 0.05
  transposition_prob: 0.05
  seed: null
```

## B.3 Environment Variables

The following environment variables can override configuration settings:

```bash
FAUXLINGO_DEVICE="cuda"          # Override compute device
FAUXLINGO_BATCH_SIZE="64"        # Override batch size
FAUXLINGO_TEMPERATURE="0.8"      # Override temperature
FAUXLINGO_SEED="42"             # Set random seed
```

## B.4 Configuration Validation

The configuration system validates:
1. Numeric ranges (e.g., probabilities between 0 and 1)
2. Compatibility between components
3. Resource requirements (e.g., memory constraints)
4. Device availability

Configuration errors include detailed messages explaining the validation failure and suggested fixes.

# Mathematics and Code Correlation

## 1. Topic Space

### Mathematical Definition
A topic space consists of T orthonormal vectors in ℝᵛ where V is the vocabulary size.

**Topic Vector Properties**:
- Unit length: ‖vi‖₂ = 1
- Orthogonality: vi · vj = 0 for i ≠ j

### Code Implementation
In `core/topics.py`:
```python
class TopicVectorSpace:
    def _validate_vectors(self, vectors: torch.Tensor) -> None:
        # Check unit length
        norms = torch.linalg.norm(vectors, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms)):
            raise ValueError("Topic vectors must have unit length")

        # Check orthogonality
        gram = vectors @ vectors.T
        should_be_identity = torch.eye(self.n_topics, device=vectors.device)
        if not torch.allclose(gram, should_be_identity, atol=1e-6):
            raise ValueError("Topic vectors must be orthogonal")
```

### Distribution Generation
**Math**: p = Σ(wᵢvᵢ) for i = 1 to T

**Code**:
```python
def get_distribution(self, mixture: torch.Tensor) -> torch.Tensor:
    """Project mixture onto topic vectors."""
    return mixture @ self.vectors
```

## 2. Color-Constrained Transitions

### Mathematical Definition
For token distribution p and color transition matrix M:
P(j|i) = p(j) * M(c(i),c(j)) / Z(i)

where Z(i) = Σ(p(j) * M(c(i),c(j))) is the normalization factor.

### Code Implementation
In `core/transitions.py`:
```python
class TransitionMatrix:
    def generate(
        self,
        topic_mixture: torch.Tensor,
        temperature: float = 1.0,
        min_prob: float = 1e-6,
    ) -> torch.Tensor:
        # Get base distributions from topics
        base_probs = self.topic_space.get_distribution(topic_mixture)

        # Convert to transition matrix
        transitions = base_probs.unsqueeze(1).expand(-1, self.vocab_size, -1)

        # Apply color mask
        color_mask = self.color_space.get_transition_mask()
        transitions = transitions * color_mask

        # Set minimum probability for valid transitions
        transitions = torch.where(
            color_mask > 0,
            torch.maximum(transitions, torch.tensor(min_prob)),
            transitions
        )

        # Normalize probabilities
        row_sums = transitions.sum(dim=-1, keepdim=True) + 1e-10
        transitions = transitions / row_sums
```

### Color Space Implementation
In `core/colors.py`:
```python
class ColorSpace:
    def get_transition_mask(self) -> torch.Tensor:
        """Create vocabulary-sized mask from color transitions."""
        mask = torch.zeros(
            (self.vocab_size, self.vocab_size),
            device=self.device
        )

        for i in range(self.n_colors):
            i_start, i_end = self.get_color_range(i)
            for j in range(self.n_colors):
                j_start, j_end = self.get_color_range(j)
                if self.transition_weights[i, j] > 0:
                    mask[i_start:i_end, j_start:j_end] = (
                        self.transition_weights[i, j]
                    )
```

## 3. Entropy Calculations

### Color Entropy
**Math**: H(C) = -Σ P(j|i) log₂ P(j|i)

**Code** in `analysis/entropy.py`:
```python
def _compute_color_entropy(self, tokens: torch.Tensor) -> float:
    # Convert tokens to colors
    colors = torch.tensor(
        [[self.transition_model.color_space.get_color(idx.item())
          for idx in seq] for seq in tokens],
        device=self.device,
    )

    # Count transitions
    counts = torch.zeros(
        (self.transition_model.color_space.n_colors,
         self.transition_model.color_space.n_colors),
        device=self.device
    )

    for b in range(len(tokens)):
        for t in range(len(tokens[0]) - 1):
            curr_color = colors[b, t]
            next_color = colors[b, t + 1]
            counts[curr_color, next_color] += 1

    # Convert to probabilities
    row_sums = counts.sum(dim=1, keepdim=True) + 1e-10
    P = counts / row_sums

    # Compute entropy
    H = -torch.sum(P * torch.log2(P + 1e-10), dim=1).mean()
    return H.item()
```

### Topic Entropy
**Math**: H(T) = -Σ w̄ᵢ log₂ w̄ᵢ

**Code**:
```python
def _compute_topic_entropy(self, mixtures: torch.Tensor) -> float:
    # Average topic distribution
    P = mixtures.mean(0)
    H = -torch.sum(P * torch.log2(P + 1e-10))
    return H.item()
```

### Token Entropy
**Math**: H(V) = -Σ f(v) log₂ f(v)

**Code**:
```python
def _compute_token_entropy(self, tokens: torch.Tensor) -> float:
    # Count frequencies
    counts = torch.zeros(
        self.transition_model.vocab_size,
        device=self.device
    )

    for seq in tokens.long():
        unique, seq_counts = torch.unique(seq, return_counts=True)
        counts[unique] += seq_counts

    # Convert to probabilities and compute entropy
    P = counts / counts.sum()
    H = -torch.sum(P * torch.log2(P + 1e-10))
    return H.item()
```

## 4. Sequence Generation

### Sampling Process
**Math**: 
1. p = Σ(wᵢvᵢ)  # Base distribution
2. P(j|i) = p(j) * M(c(i),c(j)) / Z(i)  # Apply constraints
3. P'(j|i) = P(j|i)^(1/T) / Z'(i)  # Temperature scaling

### Code Implementation
In `core/generator.py`:
```python
class SequenceGenerator:
    def generate(
        self,
        batch_size: int,
        seq_length: int,
        temperature: float = 1.0,
        topic_mixtures: torch.Tensor | None = None,
        start_tokens: torch.Tensor | None = None,
        min_prob: float = 1e-6,
    ) -> GeneratedSequences:
        # Get transition matrix
        transitions = self.transition_model.generate(
            topic_mixtures,
            temperature=temperature,
            min_prob=min_prob,
        )

        sequences = torch.zeros(
            (batch_size, seq_length),
            dtype=torch.long,
            device=self.device
        )
        log_probs = torch.zeros(batch_size, device=self.device)

        # Initial tokens
        if start_tokens is None:
            sequences[:, 0] = torch.randint(
                0, self.vocab_size,
                (batch_size,),
                device=self.device
            )
        else:
            sequences[:, 0] = start_tokens

        # Generate sequence
        for t in range(1, seq_length):
            current_probs = transitions[
                torch.arange(batch_size, device=self.device),
                sequences[:, t - 1],
            ]

            # Sample next tokens
            next_tokens = torch.multinomial(current_probs, 1).squeeze(-1)
            sequences[:, t] = next_tokens

            # Update log probabilities
            log_probs += torch.log(
                torch.gather(
                    current_probs,
                    1,
                    next_tokens.unsqueeze(1),
                )
            ).squeeze(-1)
```

This implementation shows how:
1. Topic vectors define the base distribution
2. Color constraints are applied via masking
3. Temperature scaling controls randomness
4. Sequences are generated through sequential sampling

The code carefully handles:
- Numerical stability (adding small epsilon)
- Efficient batch processing
- Device placement
- Memory management
- Edge cases in probability distributions

Each mathematical operation has corresponding validation in the code to ensure theoretical properties are maintained during computation.
