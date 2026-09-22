# Frequently Asked Questions (FAQ)

## General Questions

### What is distribution_perturbation?

`distribution_perturbation` (import name `dist_pert`) is a library for
first-order distribution perturbation of text, image, and numeric data, with
examples of second-order perturbation for robustness evaluation.

### What are first-order and second-order perturbations?

- **First-order**: perturbations applied directly to input data (e.g. word
  replacement, pixel noise, additive Gaussian noise).
- **Second-order**: perturbations applied to a data source or generation
  process, producing perturbed datasets (e.g. augmented benchmark variants).

## Installation & Setup

### Python version requirements?

`dist_pert` requires Python 3.12 or higher.

### Why are some perturbers optional?

Text perturbers pull in heavy dependencies (torch, transformers, nlpaug) and
image perturbers require Pillow/scipy/imagecorruptions. To keep the base install
lightweight, these live behind the `text` and `image` extras:

```bash
uv sync --extra text --extra image
```

### The repository uses git submodules — how do I initialise them?

Example dependencies are pulled in as submodules. After cloning, run:

```bash
git submodule update --init --recursive
```

### How do I register the environment as a Jupyter kernel?

To run `notebooks/dist_pert_usage.ipynb`:

```bash
uv run python -m ipykernel install --user --name dist_pert_env --display-name "dist_pert_env"
uv run jupyter lab
```

## Usage

### How do I add a new image corruption type?

`ImageCorruptionPerturber` resolves corruptions through the `CORRUPTION_FNS`
registry in `dist_pert.image.corruption`. Insert a callable with the signature
`(image: np.ndarray, severity: int) -> np.ndarray` to add a new type without
subclassing.

### How does the CLI resolve a perturber type?

The `type` field in the config YAML is either `module.ClassName`
(e.g. `text.contextual_word.ContextualWordPerturber`) or a bare class name that
is searched across the `text`, `image`, and `numeric` subpackages.

## Contributing

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

### What's the code style?

We follow PEP 8 and use Ruff for linting and formatting, with Google-style
docstrings. See [CONTRIBUTING.md](../CONTRIBUTING.md#style-guidelines).

### Do I need to write tests?

Yes, all new features should include tests. Optional-dependency tests should skip
gracefully via `pytest.importorskip` when the relevant extra is absent.

## Licensing

### What license does this project use?

`distribution_perturbation` is licensed under the Apache License 2.0. See
[LICENSE](../LICENSE) for details.

### Can I use this in a commercial project?

Yes, the Apache 2.0 license permits commercial use.

## Support

### Where can I report bugs?

Please open an issue on
[GitHub](https://github.com/AI-ModCon/BaseEval_distribution_perturbation/issues).

### How do I suggest new features?

Open a feature request on
[GitHub](https://github.com/AI-ModCon/BaseEval_distribution_perturbation/issues/new?template=feature_request.md).

## Troubleshooting

### Import errors when running code?

Make sure you've installed the required extras:

```bash
uv sync --extra text --extra image
```

### Tests failing?

Ensure you have development dependencies installed:

```bash
uv sync --group dev --group test --extra text --extra image
```
