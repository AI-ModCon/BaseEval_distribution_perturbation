# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apache-2.0 `LICENSE`.
- Governance documents: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`.
- GitHub templates and workflows (`pull_request_template.md`, issue templates,
  `ci.yml`, `publish.yml`).
- Documentation under `docs/` (`index.md`, `getting_started.md`, `faq.md`).
- pytest suite under `tests/` for numeric, image, text, and CLI perturbers.
- Tooling configuration in `pyproject.toml` (ruff, mypy, pytest, dependency groups).
- uv-based `Makefile` and `uv.lock` for reproducible development.

### Changed
- Migrated development workflow to [uv](https://docs.astral.sh/uv/).
- Rewrote `pyproject.toml` with full project metadata and classifiers.

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2026-08-03

### Added
- Initial release of `dist_pert`.
- `BasePerturber` interface with `perturb()`/`__call__` contract.
- Numeric perturbers: `AdditiveGaussianPerturber`, `MultiplicativeNoisePerturber`.
- Image perturbers: `GaussianNoisePerturber`, `SaltPepperPerturber`,
  `ImageCorruptionPerturber`.
- Text perturber: `ContextualWordPerturber`.
- `dist-perturb` command-line interface for perturbing JSONL files.
