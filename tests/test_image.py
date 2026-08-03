"""Tests for image perturbers.

These tests require the `image` extra; they skip gracefully when the optional
dependencies (Pillow, scipy, imagecorruptions) are not installed.
"""

import numpy as np
import pytest

pytest.importorskip("PIL", reason="image extra not installed")

from dist_pert.image import (  # noqa: E402
    GaussianNoisePerturber,
    ImageCorruptionPerturber,
    SaltPepperPerturber,
)


def _sample_image() -> np.ndarray:
    """Return a small mid-gray uint8 RGB image."""
    return np.full((32, 32, 3), 128, dtype=np.uint8)


class TestGaussianNoisePerturber:
    """Tests for GaussianNoisePerturber."""

    def test_preserves_shape_and_dtype(self) -> None:
        """Output preserves the input shape and dtype."""
        perturber = GaussianNoisePerturber(sigma=25)
        result = perturber([_sample_image()])
        assert result[0].shape == (32, 32, 3)
        assert result[0].dtype == np.uint8

    def test_output_in_uint8_range(self) -> None:
        """Output values are clipped to the valid uint8 range."""
        perturber = GaussianNoisePerturber(sigma=100)
        result = perturber([_sample_image()])
        assert result[0].min() >= 0
        assert result[0].max() <= 255

    def test_negative_sigma_raises(self) -> None:
        """A negative sigma is rejected at construction time."""
        with pytest.raises(ValueError):
            GaussianNoisePerturber(sigma=-1)


class TestSaltPepperPerturber:
    """Tests for SaltPepperPerturber."""

    def test_preserves_shape(self) -> None:
        """Output preserves the input shape."""
        perturber = SaltPepperPerturber(amount=0.1)
        result = perturber([_sample_image()])
        assert result[0].shape == (32, 32, 3)

    def test_amount_out_of_range_raises(self) -> None:
        """An amount outside [0, 1] is rejected."""
        with pytest.raises(ValueError):
            SaltPepperPerturber(amount=1.5)


class TestImageCorruptionPerturber:
    """Tests for ImageCorruptionPerturber."""

    @pytest.mark.parametrize("corruption", ["gaussian_blur", "shot_noise"])
    def test_preserves_shape(self, corruption: str) -> None:
        """Corruptions preserve the input shape and produce uint8 output."""
        perturber = ImageCorruptionPerturber(corruption=corruption, severity=3)
        result = perturber([_sample_image()])
        assert result[0].shape == (32, 32, 3)
        assert result[0].dtype == np.uint8

    def test_invalid_severity_raises(self) -> None:
        """A severity outside 1..5 is rejected."""
        with pytest.raises(ValueError):
            ImageCorruptionPerturber(corruption="gaussian_blur", severity=6)
