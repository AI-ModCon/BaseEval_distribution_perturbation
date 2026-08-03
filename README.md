# distribution_perturbation

A library for **first-order distribution perturbation** of text, image, and numeric data, with examples of **second-order perturbation** (perturbing a precursor to produce perturbed input data).

- **First-order**: perturbations applied directly to input data (e.g. word replacement, pixel noise, additive Gaussian)
- **Second-order**: perturbations applied to a data source or generation process, producing perturbed datasets (e.g. augmented benchmark variants for robustness evaluation)

> Note: the first/second-order terminology is still being refined.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/AI-ModCon/distribution_perturbation.git
cd distribution_perturbation
git submodule update --init --recursive

uv sync                                # numeric only (numpy, no heavy deps)
uv sync --extra text                   # adds text perturbers (torch, transformers, nlpaug)
uv sync --extra image                  # adds image perturbers (Pillow, scipy, imagecorruptions)
uv sync --extra text --extra image     # all modalities
```

For the NeMo Skills example only:
```bash
uv pip install -e /path/to/perlmutter_nemo-skills-main
uv pip install -e examples/perlmutter_nemo_skills/nemo-custom-benchmark
```

A pinned `requirements.txt` is also provided for pip-based workflows:
```bash
pip install -r requirements.txt
```

## Notebook

`notebooks/dist_pert_usage.ipynb` contains runnable examples for all three data types. To use it, register the environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name dist_pert_env --display-name "dist_pert_env"
uv run jupyter lab
```

## Python API

All perturbers share the same interface: instantiate with config, then call on a `list`.

```python
from dist_pert.text import ContextualWordPerturber
from dist_pert.image import GaussianNoisePerturber, SaltPepperPerturber, ImageCorruptionPerturber
from dist_pert.numeric import AdditiveGaussianPerturber, MultiplicativeNoisePerturber
import numpy as np

# Text — replaces tokens with contextually similar alternatives (BERT)
t = ContextualWordPerturber(aug_p=0.2)
result = t(["The storm produced rotating columns of air."])

# Image — pixel noise
img = GaussianNoisePerturber(sigma=25)
noisy = img([np.zeros((224, 224, 3), dtype=np.uint8)])

# Image — ImageNet-C style corruptions (gaussian_blur, jpeg_compression, shot_noise, brightness, contrast)
corr = ImageCorruptionPerturber(corruption="gaussian_blur", severity=3)
corrupted = corr([np.zeros((224, 224, 3), dtype=np.uint8)])

# Numeric
num = AdditiveGaussianPerturber(sigma=0.5)
perturbed = num([np.array([1.0, 2.0, 3.0])])
```

## CLI

```bash
uv run dist-perturb \
  --input data/tasks.jsonl \
  --config perturbation.yaml \
  --field question \
  --output output/aug.jsonl
```

Config YAML format:
```yaml
type: text.contextual_word.ContextualWordPerturber
aug_p: 0.1
model_path: google-bert/bert-base-cased
model_type: bert
```

## Second-Order Example: TBD

## Development Setup

Install the development environment (dev tools, test dependencies, and modality extras):

```bash
uv sync --group dev --group test --extra text --extra image
```

Common tasks are wrapped in a `Makefile`:

```bash
make install-dev     # uv sync with dev/test groups and text/image extras
make test            # run the pytest suite
make test-cov        # run tests with coverage
make lint            # ruff check
make format          # ruff format
make type-check      # mypy on src/dist_pert
```

This project uses:

- **[uv](https://docs.astral.sh/uv/)**: fast package manager producing reproducible builds via `uv.lock`.
- **[ruff](https://docs.astral.sh/ruff/)**: unified linter and formatter.
- **[mypy](https://mypy-lang.org/)**: static type checker.

## Testing

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=dist_pert --cov-report=term-missing
```

Image and text tests skip gracefully when the corresponding extras are not installed.

## Documentation

See the [docs](./docs) directory:

- [Getting Started](./docs/getting_started.md)
- [FAQ](./docs/faq.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines, including our policy on AI/LLM-assisted contributions.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](./CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Support

This project acknowledges support from the U.S. Department of Energy's Genesis Mission.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## Questions or Issues?

For questions or to report issues, please open an issue on [GitHub](https://github.com/AI-ModCon/distribution_perturbation/issues).
