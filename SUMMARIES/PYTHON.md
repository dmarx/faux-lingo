# Python Project Structure

## faux_lingo/analysis/entropy.py
```python
@dataclass
class EntropyMetrics
    """
    Container for sequence entropy measurements.
    Attributes:
        color_entropy: Empirical entropy of color transitions
        topic_entropy: Entropy of topic mixtures used in generation
        token_entropy: Empirical entropy of generated token sequences
    """

    @classmethod
    def zero(cls) -> Self
        """Create EntropyMetrics initialized to zero."""


class EntropyAnalyzer
    """
    Analyzer for information-theoretic properties of sequences.
    Core functionality:
    1. Computing empirical entropy of generated sequences
    2. Analyzing topic mixture entropy
    3. Tracking color transition patterns
    """

    def __init__(self, transition_model: TransitionMatrix)
        """Initialize analyzer with transition model."""

    def analyze_sequences(self, sequences: GeneratedSequences) -> EntropyMetrics
        """
        Compute entropy metrics for sequences.
        Args:
            sequences: Generated token sequences and properties
        Returns:
            EntropyMetrics containing various entropy measures
        """

    def _compute_color_entropy(self, tokens: Any) -> float
        """
        Compute empirical entropy of color transitions.
        Args:
            tokens: Generated token sequences [batch, seq_len]
        Returns:
            Estimated color transition entropy
        """

    def _compute_topic_entropy(self, mixtures: Any) -> float
        """
        Compute entropy of topic mixtures.
        Args:
            mixtures: Topic mixture weights [batch, n_topics]
        Returns:
            Entropy of average topic distribution
        """

    def _compute_token_entropy(self, tokens: Any) -> float
        """
        Compute empirical entropy of token sequences.
        Args:
            tokens: Generated token sequences [batch, seq_len]
        Returns:
            Estimated token entropy
        """


```

## faux_lingo/core/colors.py
```python
@dataclass
class ColorMapping
    """
    Maps between token indices and color classes.
    Attributes:
        boundaries: Tensor of token index boundaries for each color
        fractions: Normalized fraction of vocabulary for each color
    """

class ColorSpace
    """
    Manages color classes and their transition rules.
    Core properties:
    1. Each token belongs to exactly one color class
    2. Color classes partition the vocabulary space
    3. Transitions between colors follow specified rules
    """

    def __init__(self, color_fractions: list[float] | Any, vocab_size: int, transition_weights: Any | None, device: str | None)
        """
        Initialize color space with fractions and transition rules.
        Args:
            color_fractions: Relative sizes of color classes
            vocab_size: Total vocabulary size
            transition_weights: Optional matrix of color transition weights
            device: Optional compute device, defaults to CPU
        Notes:
            - Color fractions will be normalized to sum to 1
            - If transition_weights not provided, defaults to all-ones matrix
        """

    def _compute_mapping(self, fractions: Any, vocab_size: int) -> ColorMapping
        """
        Compute normalized fractions and token boundaries.
        Args:
            fractions: Raw color fractions
            vocab_size: Total vocabulary size
        Returns:
            ColorMapping with normalized fractions and boundaries
        """

    def _validate_transitions(self, weights: Any) -> None
        """
        Validate transition weight matrix.
        Args:
            weights: Color transition weight matrix
        Raises:
            ValueError: If weights have invalid shape or values
        """

    def get_color(self, token_idx: int) -> int
        """
        Get color index for a token index.
        Args:
            token_idx: Index in vocabulary
        Returns:
            Index of the color that token_idx belongs to
        Raises:
            ValueError: If token_idx is invalid
        """

    def get_color_range(self, color_idx: int) -> tuple[[int, int]]
        """
        Get token index range for a color.
        Args:
            color_idx: Index of the color
        Returns:
            Tuple of (start_idx, end_idx) for color's token range
        Raises:
            ValueError: If color_idx is invalid
        """

    def get_transition_mask(self) -> Any
        """
        Get vocabulary-sized mask from color transition weights.
        Returns:
            Boolean mask of shape [vocab_size, vocab_size]
        """

    def save(self, path: Path) -> None
        """Save color space parameters."""

    @classmethod
    def load(cls, path: Path, device: str | None) -> 'ColorSpace'
        """Load color space from saved parameters."""


```

## faux_lingo/core/generator.py
```python
@dataclass
class GeneratedSequences
    """
    Container for generated sequences and their properties.
    Attributes:
        tokens: Generated token sequences [batch_size, seq_len]
        topic_mixtures: Topic mixtures used for generation [batch_size, n_topics]
        log_probs: Log probabilities of generated sequences [batch_size]
    """

class SequenceGenerator
    """
    Generates sequences using topic and color-constrained transitions.
    Core functionality:
    1. Sampling sequences from transition matrices
    2. Computing sequence probabilities
    3. Generating with specific topic mixtures or color constraints
    """

    def __init__(self, transition_model: TransitionMatrix, device: str | None)
        """
        Initialize sequence generator.
        Args:
            transition_model: Model for generating transition matrices
            device: Optional compute device, defaults to CPU
        """

    def generate(self, batch_size: int, seq_length: int, temperature: float, topic_mixtures: Any | None, start_tokens: Any | None, min_prob: float) -> GeneratedSequences
        """
        Generate batch of sequences.
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            temperature: Controls randomness in sampling
            topic_mixtures: Optional pre-specified topic mixtures [batch_size, n_topics]
            start_tokens: Optional initial tokens [batch_size]
            min_prob: Minimum probability for valid transitions
        Returns:
            GeneratedSequences containing tokens and properties
        Notes:
            If topic_mixtures not provided, samples from uniform distribution
            If start_tokens not provided, samples initial tokens uniformly
        """

    def generate_with_color(self, batch_size: int, seq_length: int, start_color: int, temperature: float, topic_mixtures: Any | None) -> GeneratedSequences
        """
        Generate sequences starting with tokens of a specific color.
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            start_color: Color index to start sequences with
            temperature: Controls randomness in sampling
            topic_mixtures: Optional pre-specified topic mixtures
        Returns:
            GeneratedSequences with tokens starting from specified color
        """

    @classmethod
    def create_uniform(cls, vocab_size: int, n_topics: int, color_fractions: list[float], device: str | None) -> Self
        """
        Create generator with uniform topic and color distributions.
        Args:
            vocab_size: Size of token vocabulary
            n_topics: Number of topics
            color_fractions: Relative sizes of color classes
            device: Optional compute device
        Returns:
            SequenceGenerator with uniform parameters
        """


```

## faux_lingo/core/serialization.py
```python
@dataclass
class GenerationMetadata
    """
    Metadata for tracking generation state and configuration.
    Attributes:
        config: Generation configuration
        vocab_hierarchy: Current vocabulary state
        transition_model: Current transition model state
        sequences_generated: Number of sequences generated
        last_batch_id: ID of last generated batch
    """

    def save(self, path: Path) -> None
        """
        Save generation metadata to disk.
        Args:
            path: Directory to save metadata files
        """

    @classmethod
    def load(cls, path: Path, device: str | None) -> 'GenerationMetadata'
        """
        Load generation metadata from disk.
        Args:
            path: Directory containing metadata files
            device: Optional device for loading model components
        Returns:
            Loaded GenerationMetadata instance
        """


```

## faux_lingo/core/topics.py
```python
class TopicVectorSpace
    """
    Manages a set of orthonormal topic vectors that define token distributions.
    Core mathematical properties:
    1. Each topic vector is unit length
    2. All topic vectors are orthogonal to each other
    3. Topic vectors form a basis for generating token distributions
    """

    def __init__(self, n_topics: int, vocab_size: int, vectors: Any | None, device: str | None)
        """
        Initialize topic vector space.
        Args:
            n_topics: Number of topics (must be <= vocab_size)
            vocab_size: Size of token vocabulary
            vectors: Optional pre-defined topic vectors
            device: Optional compute device for tensors, defaults to CPU
        """

    def _validate_vectors(self, vectors: Any) -> None
        """
        Validate topic vector properties.
        Args:
            vectors: Topic vectors to validate
        Raises:
            ValueError: If vectors don't meet required properties
        """

    def _init_random_vectors(self) -> Any
        """
        Initialize random orthonormal topic vectors.
        Returns:
            Tensor of orthonormal vectors
        """

    def get_distribution(self, mixture: Any) -> Any
        """
        Get token distribution for a topic mixture.
        Args:
            mixture: Topic mixture weights [batch_size, n_topics]
        Returns:
            Token probabilities [batch_size, vocab_size]
        Notes:
            Probabilities may need further processing (e.g., ReLU, normalization)
            to get final transition probabilities
        """

    def save(self, path: Path) -> None
        """Save topic vectors."""

    @classmethod
    def load(cls, path: Path, device: str | None) -> 'TopicVectorSpace'
        """Load topic vectors and construct space."""


```

## faux_lingo/core/transitions.py
```python
class TransitionMatrix
    """
    Manages transition probability matrices that respect both topic and color constraints.
    Core properties:
    1. Matrices are proper probability distributions (row-wise sum to 1)
    2. Color transitions follow specified weights
    3. Global token distributions reflect topic mixtures
    """

    def __init__(self, topic_space: TopicVectorSpace, color_space: ColorSpace, device: str | None)
        """
        Initialize transition matrix generator.
        Args:
            topic_space: Space of topic vectors
            color_space: Color class definitions and rules
            device: Optional compute device, defaults to CPU
        Raises:
            ValueError: If spaces have incompatible dimensions
        """

    def generate(self, topic_mixture: Any, temperature: float, min_prob: float) -> Any
        """
        Generate transition probability matrix for given topic mixture.
        Args:
            topic_mixture: Mixture weights for topics [batch_size, n_topics]
            temperature: Controls entropy of distributions (higher = more uniform)
            min_prob: Minimum probability for valid transitions
        Returns:
            Transition probability matrix [batch_size, vocab_size, vocab_size]
        Notes:
            1. Output[b,i,j] = P(token_j | token_i) for sequence b
            2. Each row sums to 1 (is a valid probability distribution)
            3. Respects both topic and color constraints
        """

    @classmethod
    def create_uniform(cls, vocab_size: int, n_topics: int, color_fractions: list[float], device: str | None) -> Self
        """
        Create transition matrix with uniform topic vectors and color transitions.
        Args:
            vocab_size: Size of token vocabulary
            n_topics: Number of topics to use
            color_fractions: Relative sizes of color classes
            device: Optional compute device
        Returns:
            TransitionMatrix instance with uniform parameters
        """

    def save(self, path: str) -> None
        """Save transition parameters."""

    @classmethod
    def load(cls, path: str, device: str | None) -> Self
        """Load transition parameters."""


```

## faux_lingo/core/vocab_builder.py
```python
@dataclass
class BuilderConfig
    """
    Configuration for vocabulary hierarchy construction.
    Attributes:
        token_vocab_size: Size of base token vocabulary
        sequence_lengths: List of sequence lengths for each level
        vocab_sizes: List of vocabulary sizes for each level
        seed: Optional random seed for reproducibility
    """

    def __post_init__(self) -> None
        """Validate configuration."""


class VocabBuilder
    """
    Builds hierarchical vocabularies with constrained structure.
    Core functionality:
    1. Random sampling of valid token sequences
    2. Building vocabularies level by level
    3. Tracking used sequences to avoid duplicates
    """

    def __init__(self, config: BuilderConfig)
        """
        Initialize builder with configuration.
        Args:
            config: Parameters for vocabulary construction
        """

    def build(self) -> VocabHierarchy
        """
        Build complete vocabulary hierarchy.
        Returns:
            VocabHierarchy with all levels constructed
        """

    @classmethod
    def create_default_config(cls) -> BuilderConfig
        """
        Create configuration with reasonable defaults.
        Returns:
            BuilderConfig for simple three-level hierarchy
        """


def create_word_hierarchy(token_vocab_size: int, n_chars: int, n_words: int, chars_per_word: int, seed: int | None) -> VocabHierarchy
    """
    Convenience function to create character-word vocabulary.
    Args:
        token_vocab_size: Size of base token vocabulary
        n_chars: Number of unique characters
        n_words: Number of unique words
        chars_per_word: Number of characters per word
        seed: Optional random seed
    Returns:
        Two-level hierarchy mapping words to character sequences
    """

```

## faux_lingo/core/vocab_extensions.py
```python
@dataclass
class MultiMappingLevel
    """
    Vocabulary level supporting multiple mappings.
    Attributes:
        vocab_size: Number of tokens at this level
        chunk_size: Number of tokens from parent level per token
        sequences: Mapping of token to list of possible sequences with probabilities
    """

    def __post_init__(self) -> None
        """Validate level properties."""


class MultiMappingHierarchy
    """
    Hierarchical vocabulary with multiple possible mappings.
    Core functionality:
    1. Support for multiple sequences mapping to same token
    2. Probabilistic sequence selection during decoding
    3. Integration with existing vocabulary system
    """

    def __init__(self, levels: Sequence[MultiMappingLevel], device: str | None)
        """
        Initialize hierarchy with multiple mapping levels.
        Args:
            levels: Sequence of vocabulary levels from lowest to highest
            device: Optional compute device, defaults to CPU
        """

    def decode_sequence(self, tokens: Any, start_level: int, target_level: int, seed: int | None) -> Any
        """
        Decode token sequence with probabilistic mapping selection.
        Args:
            tokens: Input token sequence [batch_size, seq_len]
            start_level: Index of starting vocabulary level
            target_level: Index of target vocabulary level
            seed: Optional random seed for reproducible decoding
        Returns:
            Decoded token sequences at target level [batch_size, new_seq_len]
        """


@dataclass
class AugmentationConfig
    """
    Configuration for sequence augmentation.
    Attributes:
        deletion_prob: Probability of character deletion
        insertion_prob: Probability of random character insertion
        substitution_prob: Probability of character substitution
        transposition_prob: Probability of adjacent character transposition
        seed: Optional random seed for reproducibility
    """

    def __post_init__(self) -> None
        """Validate configuration."""


class SequenceAugmenter
    """
    Applies random perturbations to token sequences.
    Core functionality:
    1. Character-level augmentations (deletion, insertion, etc.)
    2. Controlled randomization based on probabilities
    3. Vocabulary-aware modifications
    """

    def __init__(self, vocab_size: int, config: AugmentationConfig, device: str | None)
        """
        Initialize augmenter with vocabulary and configuration.
        Args:
            vocab_size: Size of token vocabulary
            config: Augmentation parameters
            device: Optional compute device, defaults to CPU
        """

    def augment_sequence(self, sequence: TokenSeq) -> TokenSeq
        """
        Apply random augmentations to token sequence.
        Args:
            sequence: Input token sequence
        Returns:
            Augmented token sequence
        """

    def _delete(self, seq: list[int]) -> list[int]
        """Randomly delete a token."""

    def _insert(self, seq: list[int]) -> list[int]
        """Insert random token."""

    def _substitute(self, seq: list[int]) -> list[int]
        """Replace token with a different random token."""

    def _transpose(self, seq: list[int]) -> list[int]
        """Swap adjacent tokens."""


def convert_to_multi_mapping(hierarchy: VocabHierarchy, augmenter: SequenceAugmenter | None, n_variants: int) -> MultiMappingHierarchy
    """
    Convert standard hierarchy to multi-mapping hierarchy.
    Args:
        hierarchy: Standard vocabulary hierarchy
        augmenter: Optional sequence augmenter for variants
        n_variants: Number of variants to generate per sequence
    Returns:
        MultiMappingHierarchy with original and variant sequences
    """

```

## faux_lingo/core/vocab_mapping.py
```python
@dataclass
class VocabLevel
    """
    A single level in the vocabulary hierarchy.
        Attributes:
        vocab_size: Number of tokens at this level
        chunk_size: Number of tokens from parent level per token
        sequences: Mapping of each token to its constituent sequence
    """

    def __post_init__(self) -> None
        """Validate vocabulary level properties."""

    @property
    def max_sequence_length(self) -> int
        """Maximum length of any sequence in this level."""


class VocabHierarchy
    """
    Manages hierarchical relationships between vocabulary levels.
    Note: VocabLevels represent mappings BETWEEN levels, not the levels themselves.
    With n VocabLevels, we actually have n+1 vocabulary levels total.
    Level indexing goes from most abstract (0) to most concrete (n):
    Level 0 -> Level 1 (Mapping A)
    Level 1 -> Level 2 (Mapping B)
    """

    def __init__(self, levels: Sequence[VocabLevel], device: str | None) -> None
        """
        Initialize vocabulary hierarchy.
        Args:
            levels: Sequence of vocabulary mappings from highest to lowest abstraction
            device: Optional compute device, defaults to CPU
        """

    def decode_sequence(self, tokens: Any, start_level: int | None, target_level: int | None) -> Any
        """
        Decode token sequence from one level to another.
        Args:
            tokens: Input token sequence [batch_size, seq_len]
            start_level: Optional starting level (defaults to 0)
            target_level: Optional target level (defaults to max level)
        Returns:
            Decoded token sequences at target level [batch_size, new_seq_len]
        """

    def _build_decode_tables(self) -> list[Any]
        """
        Build lookup tables for decoding between levels.
        Returns:
            List of tensors mapping level i tokens to level i+1 sequences
            Each tensor has shape [parent_vocab_size, max_child_sequence_length]
            with padded sequences for consistent shape
        """

    @classmethod
    def from_sequences(cls, sequences: list[dict[[TokenIdx, TokenSeq]]], chunk_sizes: list[int], device: str | None) -> Self
        """
        Create hierarchy from sequence mappings.
        Args:
            sequences: Mappings for each level
            chunk_sizes: Number of tokens per chunk at each level
            device: Optional compute device
        Returns:
            Initialized VocabHierarchy
        """

    def __getitem__(self, level: int) -> VocabLevel
        """Get vocabulary level by index."""

    def __len__(self) -> int
        """Get number of vocabulary levels."""

    def __iter__(self) -> Iterator[VocabLevel]
        """Iterate over vocabulary levels."""


```

## faux_lingo/data/dataset.py
```python
class BatchStats(TypedDict)

@dataclass
class DatasetConfig
    """
    Configuration for dataset generation.
    Attributes:
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence
        n_batches: Total number of batches to generate
        temperature: Controls randomness in generation
        seed: Random seed for reproducibility
    """

class SequenceDataset
    """
    Manages generation and iteration of sequence batches.
    Core functionality:
    1. Batch generation with consistent configuration
    2. Tracking of sequence properties and metadata
    3. Iterator interface for training/validation
    """

    def __init__(self, generator: SequenceGenerator, config: DatasetConfig)
        """
        Initialize dataset with generator and configuration.
        Args:
            generator: Sequence generator instance
            config: Dataset generation parameters
        """

    def __len__(self) -> int
        """Get total number of batches."""

    def __iter__(self) -> Iterator[GeneratedSequences]
        """Create iterator over sequence batches."""

    def __next__(self) -> GeneratedSequences
        """Get next batch of sequences."""

    def generate_batch(self, topic_mixtures: Any | None, start_color: int | None) -> GeneratedSequences
        """
        Generate a single batch of sequences.
        Args:
            topic_mixtures: Optional pre-specified topic mixtures
            start_color: Optional color index to start sequences with
        Returns:
            GeneratedSequences containing tokens and properties
        """

    def get_color_sequences(self, tokens: Any) -> Any
        """
        Convert token sequences to color sequences.
        Args:
            tokens: Token sequences [batch_size, seq_length]
        Returns:
            Color sequences [batch_size, seq_length]
        """

    def get_batch_stats(self, batch: GeneratedSequences) -> BatchStats
        """
        Compute statistics for a batch of sequences.
        Args:
            batch: Batch of generated sequences
        Returns:
            Dictionary of batch statistics
        """

    @property
    def vocab_size(self) -> int
        """Get vocabulary size of generator."""

    @property
    def n_topics(self) -> int
        """Get number of topics in generator."""

    @property
    def n_colors(self) -> int
        """Get number of color classes in generator."""


```

## tests/test_colors.py
```python
def test_normalization()
    """Test that color fractions are properly normalized."""

def test_boundaries()
    """Test token boundary calculations."""

def test_color_lookup()
    """Test token to color mapping."""

def test_transition_weights()
    """Test transition weight validation and mask creation."""

def test_save_load(tmp_path)
    """Test serialization of color space."""

def test_device_handling()
    """Test device placement and movement."""

```

## tests/test_dataset.py
```python
def simple_generator()
    """Create a simple generator for testing."""

def simple_config()
    """Create a basic dataset configuration."""

def test_dataset_iteration(simple_generator, simple_config)
    """Test basic dataset iteration."""

def test_reproducibility(simple_generator, simple_config)
    """Test that sequence generation is reproducible with same seed."""

def test_color_sequence_conversion(simple_generator, simple_config)
    """Test conversion of tokens to color sequences."""

def test_batch_stats(simple_generator, simple_config)
    """Test batch statistics computation."""

def test_color_constrained_generation(simple_generator, simple_config)
    """Test generation with specific start color."""

def test_topic_constrained_generation(simple_generator, simple_config)
    """Test generation with specific topic mixtures."""

def test_device_handling(simple_generator, simple_config)
    """Test device placement and consistency."""

```

## tests/test_entropy.py
```python
def simple_analyzer()
    """Create analyzer with simple uniform generator."""

def sample_sequences(simple_analyzer)
    """Generate sample sequences for testing."""

def test_metrics_zero()
    """Test zero initialization of metrics."""

def test_color_entropy(simple_analyzer)
    """Test color entropy computation with different transition rules."""

def test_topic_entropy(simple_analyzer)
    """Test topic entropy computation."""

def test_token_entropy(simple_analyzer, sample_sequences)
    """Test token entropy computation."""

def test_device_handling(simple_analyzer, sample_sequences)
    """Test device placement and consistency."""

```

## tests/test_generator.py
```python
def simple_generator()
    """Create a simple generator for testing."""

def test_sequence_shapes(simple_generator)
    """Test output shapes from generation."""

def test_token_ranges(simple_generator)
    """Test that generated tokens are within vocabulary."""

def test_color_start(simple_generator)
    """Test generation with specific start color."""

def test_temperature_effect(simple_generator)
    """Test that temperature effect is consistent across runs."""

def test_topic_mixture_validation(simple_generator)
    """Test validation of topic mixture inputs."""

def test_start_token_validation(simple_generator)
    """Test validation of start token inputs."""

def test_color_validation(simple_generator)
    """Test validation of color inputs."""

def test_log_probability_consistency(simple_generator)
    """Test that log probabilities are consistent with transitions."""

def test_reproducibility(simple_generator)
    """Test that sequences are reproducible with same seed."""

def get_transition_counts(tokens: Any) -> Any
    """Get counts of token-to-token transitions."""

def get_entropy(probs: Any) -> float
    """Calculate average entropy of transition distributions."""

```

## tests/test_topics.py
```python
def test_init_validation()
    """Test input validation during initialization."""

def test_vector_properties()
    """Test that topic vectors have required mathematical properties."""

def test_distribution_shape()
    """Test output shape of get_distribution."""

def test_save_load(tmp_path)
    """Test serialization of topic vectors."""

def test_device_handling()
    """Test device placement and movement."""

def test_random_state()
    """Test reproducibility of random initialization."""

```

## tests/test_transitions.py
```python
def simple_matrix()
    """Create a simple transition matrix for testing."""

def test_initialization()
    """Test constructor validation."""

def test_uniform_creation()
    """Test creation of uniform transition matrix."""

def test_probability_properties(simple_matrix)
    """Test that generated matrices have valid probability properties."""

def test_color_constraints(simple_matrix)
    """Test that color transition constraints are respected."""

def test_temperature_effect(simple_matrix)
    """Test that temperature affects distribution entropy."""

def test_batch_generation(simple_matrix)
    """Test generation of multiple matrices simultaneously."""

def test_min_probability(simple_matrix)
    """Test that minimum probability is respected for valid transitions."""

def test_device_consistency(simple_matrix)
    """Test that all tensors stay on the same device."""

def test_invalid_mixture_shape(simple_matrix)
    """Test validation of topic mixture shape."""

def test_reproducibility()
    """Test that results are reproducible with same random seed."""

```

## tests/test_vocab_builder.py
```python
def simple_config()
    """Create simple builder configuration."""

def test_config_validation()
    """Test builder configuration validation."""

def test_builder_reproducibility(simple_config)
    """Test that building is reproducible with same seed."""

def test_sequence_uniqueness(simple_config)
    """Test that generated sequences are unique within levels."""

def test_sequence_validity(simple_config)
    """Test that sequences use valid tokens from previous level."""

def test_create_word_hierarchy()
    """Test convenience function for word hierarchy creation."""

def test_default_config()
    """Test default configuration creation."""

```

## tests/test_vocab_extensions.py
```python
def simple_multi_level()
    """Create simple multi-mapping level for testing."""

def simple_augmenter()
    """Create sequence augmenter with test configuration."""

def test_multi_level_validation()
    """Test validation of multi-mapping level properties."""

def test_multi_hierarchy_decoding(simple_multi_level)
    """Test sequence decoding with multiple mappings."""

def test_augmenter_operations(simple_augmenter)
    """Test individual augmentation operations."""

def test_augmenter_sequence_handling(simple_augmenter)
    """Test sequence augmentation edge cases."""

def test_hierarchy_conversion()
    """Test conversion from standard to multi-mapping hierarchy."""

def test_augmentation_config_validation()
    """Test validation of augmentation configuration."""

def test_device_handling()
    """Test device placement and consistency."""

```

## tests/test_vocab_mapping.py
```python
def simple_hierarchy()
    """
    Create a simple 3-level hierarchy for testing.
    Structure:
    - Level 0 (most abstract) tokens map to 2 level 1 tokens
    - Level 1 tokens map to 2 level 2 (most concrete) tokens
    Example mappings:
    Level 0 -> Level 1:
    - 0 -> (0, 1)
    - 1 -> (1, 2)
    Level 1 -> Level 2:
    - 0 -> (0, 1)
    - 1 -> (2, 3)
    - 2 -> (3, 4)
    """

def test_vocab_level_validation()
    """Test validation of vocabulary level properties."""

def test_hierarchy_respected()

def test_single_token_decoding(simple_hierarchy)
    """Test decoding of individual tokens."""

def test_sequence_decoding(simple_hierarchy)
    """Test decoding of token sequences."""

def test_batch_decoding(simple_hierarchy)
    """Test decoding of batched sequences."""

def test_invalid_level_decoding(simple_hierarchy)
    """Test validation of decoding levels."""

def test_default_decoding(simple_hierarchy)
    """Test default decoding behavior."""

def test_from_sequences()
    """Test creation of hierarchy from sequence mappings."""

def test_device_handling()
    """Test device placement and movement of tensors."""

```
