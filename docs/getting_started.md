# Getting Started

Welcome to distribution_perturbation (`dist_pert`)! This guide will help you get
started with applying first-order distribution perturbations to text, image, and
numeric data.

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- git (for version control)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AI-ModCon/BaseEval_distribution_perturbation.git
cd BaseEval_distribution_perturbation
git submodule update --init --recursive
```

### 2. Install Dependencies

The numeric perturbers only require numpy. Text and image perturbers are gated
behind optional extras so you install only what you need:

```bash
uv sync                                   # numeric only (numpy)
uv sync --extra text                      # adds text perturbers (torch, transformers, nlpaug)
uv sync --extra image                     # adds image perturbers (Pillow, scipy, imagecorruptions)
uv sync --extra text --extra image        # all modalities
```

## Development Setup

For development work, install the dev and test dependency groups alongside the
modality extras:

```bash
uv sync --group dev --group test --extra text --extra image
```

## Quickstart

All perturbers share the same interface: instantiate with configuration, then
call the instance on a `list` of samples.

```python
import numpy as np

# --- Numeric ---
from dist_pert.numeric import AdditiveGaussianPerturber, MultiplicativeNoisePerturber

add = AdditiveGaussianPerturber(sigma=0.5)
perturbed = add([np.array([1.0, 2.0, 3.0])])

mult = MultiplicativeNoisePerturber(scale=0.1, distribution="lognormal")
scaled = mult([np.array([1.0, 2.0, 3.0])])

# --- Image (requires the `image` extra) ---
from dist_pert.image import (
    GaussianNoisePerturber,
    SaltPepperPerturber,
    ImageCorruptionPerturber,
)

noisy = GaussianNoisePerturber(sigma=25)([np.zeros((224, 224, 3), dtype=np.uint8)])
sp = SaltPepperPerturber(amount=0.05)([np.zeros((224, 224, 3), dtype=np.uint8)])
corrupt = ImageCorruptionPerturber(corruption="gaussian_blur", severity=3)(
    [np.zeros((224, 224, 3), dtype=np.uint8)]
)

# --- Text (requires the `text` extra) ---
from dist_pert.text import ContextualWordPerturber

text = ContextualWordPerturber(aug_p=0.2)
result = text(["The storm produced rotating columns of air."])
```

## Command-Line Usage

The `dist-perturb` CLI perturbs a single field of every record in a JSONL file:

```bash
uv run dist-perturb \
  --input data/tasks.jsonl \
  --config perturbation.yaml \
  --field question \
  --output output/aug.jsonl
```

Example `perturbation.yaml`:

```yaml
type: text.contextual_word.ContextualWordPerturber
aug_p: 0.1
model_path: google-bert/bert-base-cased
model_type: bert
```

## Running Tests

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=dist_pert --cov-report=html --cov-report=term-missing
```

Image and text tests skip gracefully if the corresponding extras are not
installed.

## Code Style

We use Ruff for both linting and formatting, and mypy for type checking.

```bash
uv run ruff format .            # format
uv run ruff check .             # lint
uv run mypy src/dist_pert       # type check
```

## Next Steps

- Read the [Contributing Guidelines](../CONTRIBUTING.md)
- Browse the runnable examples in `notebooks/dist_pert_usage.ipynb`

## Support

For questions or issues, please open a GitHub issue or check the [FAQ](./faq.md).
