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
