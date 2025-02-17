# Faux-Lingo

Faux-Lingo is a toolkit for generating synthetic language data using a constructed language framework. It simulates low-entropy, structured corpora via a graph-based topic model with color-coded (i.e. grammar-inspired) transitions, preferential attachment, and hierarchical vocabulary construction.

## Features

- **Synthetic Data Generation**: Create corpora with controlled entropy and structure.
- **Graph-Based Topic Modeling**: Model grammatical constraints using a color transition matrix.
- **On-the-Fly Data Streaming**: Integrate seamlessly with PyTorch's DataLoader for training language models.
- **Customizable Parameters**: Control vocabulary size, topic concentration, attachment biases, and more.

## Installation

You can install FauxLingo via pip (when released on PyPI):

```bash
pip install faux_lingo
```

Or clone this repository and install locally:

```bash
git clone https://github.com/yourusername/faux_lingo.git
cd faux_lingo
pip install .
```

## Usage Example

Here's a quick example of how to use FauxLingo with a PyTorch DataLoader:

```python
from faux_lingo.dataset import GenerativeCorpusDataset
from faux_lingo.data_generator import generate_artifacts
from torch.utils.data import DataLoader

# Generate fixed artifacts for the synthetic corpus.
artifacts = generate_artifacts(
    num_topics=5,
    word_vocab_size=100,
    characters_per_word=3,
    runes_per_character=3,
    tokens_per_rune=1,
    token_vocab_size=10,
    rune_vocab_size=30,
    char_vocab_size=20,
    topic_word_alpha=0.1,
    num_colors=5,
    avg_degree=5,
    modes_per_color=2,
    attachment_bias=0.5,
    random_color_transition=False,  # uniform color transitions by default
    color_transition_matrix=None,     # user-supplied matrix takes priority if provided
    sigma=1.0,
    epsilon=0.1,
    seed=42
)

# Create the dataset and DataLoader.
dataset = GenerativeCorpusDataset(doc_count=1000, doc_length=50, artifacts=artifacts, doc_topic_alpha=0.5)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate over a batch.
for batch in dataloader:
    print(batch.shape)  # e.g., (16, sequence_length)
    break
```

## Package Structure

```
faux_lingo/                  # Top-level project folder
├── faux_lingo/              # Main package directory
│   ├── __init__.py
│   ├── data_generator.py    # Contains our synthetic corpus & transition matrix generators
│   ├── dataset.py           # PyTorch Dataset wrapper and related utilities
│   ├── models.py            # (Optional) Example models or helper code (e.g., for demo training)
│   └── utils.py             # Any extra helper functions
├── tests/                   # Unit tests for the package
│   ├── __init__.py
│   └── test_data_generator.py
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions workflow for CI (e.g., running tests)
├── pyproject.toml           # Build system configuration
├── setup.cfg              # Package metadata and options
├── MANIFEST.in              # Specifies non-code files to include in the package
├── README.md                # Package overview and usage instructions
└── LICENSE                  # License file (e.g., MIT)
```

