# API Reference

This page summarizes the public surface exported by `dist_pert`. For
implementation details and parameter behavior, see the source docstrings in
`src/dist_pert/`.

## Core Interface

### `dist_pert.BasePerturber[T]`

Abstract base class for all perturbers.

- `perturb(data: list[T]) -> list[T]`
- `__call__(data: list[T]) -> list[T]`

All perturbers in this repository follow the same contract: instantiate with
configuration, then call the instance on a list of samples.

## Text

### `dist_pert.text.ContextualWordPerturber`

Perturbs text by replacing words with contextually similar alternatives.

Constructor arguments:

- `model_path`
- `model_type`
- `aug_p`
- `aug_min`
- `aug_max`
- `device`

## Image

### `dist_pert.image.GaussianNoisePerturber`

Adds zero-mean Gaussian noise to each image in a batch.

Constructor arguments:

- `sigma`

### `dist_pert.image.SaltPepperPerturber`

Randomly sets pixels to the minimum or maximum value.

Constructor arguments:

- `amount`

### `dist_pert.image.ImageCorruptionPerturber`

Applies ImageNet-C style corruptions to images.

Constructor arguments:

- `corruption`
- `severity`

Built-in corruption names include:

- `gaussian_blur`
- `jpeg_compression`
- `shot_noise`
- `brightness`
- `contrast`

If the optional `imagecorruptions` dependency is installed, additional
ImageNet-C corruptions supported by that backend can also be used.

## Numeric

### `dist_pert.numeric.AdditiveGaussianPerturber`

Adds zero-mean Gaussian noise to numeric arrays.

Constructor arguments:

- `sigma`

### `dist_pert.numeric.MultiplicativeNoisePerturber`

Scales numeric arrays by a random multiplicative factor per element.

Constructor arguments:

- `scale`
- `distribution`

Supported distributions:

- `lognormal`
- `uniform`

## CLI

### `dist-perturb`

Command-line entry point for perturbing a single JSONL field.

Options:

- `--input` / `-i`
- `--config` / `-c`
- `--output` / `-o`
- `--field` / `-f`

The `type` field in the YAML config can be either `module.ClassName` or a bare
class name resolved within the `text`, `image`, and `numeric` subpackages.
