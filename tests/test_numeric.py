"""Tests for numeric perturbers."""

import numpy as np
import pytest

from dist_pert.numeric import AdditiveGaussianPerturber, MultiplicativeNoisePerturber


class TestAdditiveGaussianPerturber:
    """Tests for AdditiveGaussianPerturber."""

    def test_preserves_shape(self) -> None:
        """Output arrays keep the shape of their inputs."""
        perturber = AdditiveGaussianPerturber(sigma=0.5)
        data = [np.array([1.0, 2.0, 3.0]), np.zeros((2, 2))]
        result = perturber(data)
        assert len(result) == len(data)
        assert result[0].shape == (3,)
        assert result[1].shape == (2, 2)

    def test_zero_sigma_is_identity(self) -> None:
        """With sigma=0 the perturbation adds no noise."""
        perturber = AdditiveGaussianPerturber(sigma=0.0)
        arr = np.array([1.0, 2.0, 3.0])
        result = perturber([arr])
        np.testing.assert_allclose(result[0], arr)

    def test_negative_sigma_raises(self) -> None:
        """A negative sigma is rejected at construction time."""
        with pytest.raises(ValueError):
            AdditiveGaussianPerturber(sigma=-1.0)

    def test_empty_input_raises(self) -> None:
        """An empty batch is rejected."""
        perturber = AdditiveGaussianPerturber(sigma=1.0)
        with pytest.raises(ValueError):
            perturber([])


class TestMultiplicativeNoisePerturber:
    """Tests for MultiplicativeNoisePerturber."""

    def test_preserves_shape(self) -> None:
        """Output arrays keep the shape of their inputs."""
        perturber = MultiplicativeNoisePerturber(scale=0.1)
        data = [np.array([1.0, 2.0, 3.0])]
        result = perturber(data)
        assert result[0].shape == (3,)

    def test_zero_scale_lognormal_is_identity(self) -> None:
        """With scale=0 the lognormal factor is exactly 1."""
        perturber = MultiplicativeNoisePerturber(scale=0.0, distribution="lognormal")
        arr = np.array([1.0, 2.0, 3.0])
        result = perturber([arr])
        np.testing.assert_allclose(result[0], arr)

    def test_zero_scale_uniform_is_identity(self) -> None:
        """With scale=0 the uniform factor is exactly 1."""
        perturber = MultiplicativeNoisePerturber(scale=0.0, distribution="uniform")
        arr = np.array([1.0, 2.0, 3.0])
        result = perturber([arr])
        np.testing.assert_allclose(result[0], arr)

    def test_negative_scale_raises(self) -> None:
        """A negative scale is rejected at construction time."""
        with pytest.raises(ValueError):
            MultiplicativeNoisePerturber(scale=-0.1)

    def test_unknown_distribution_raises(self) -> None:
        """An unsupported distribution name is rejected."""
        with pytest.raises(ValueError):
            MultiplicativeNoisePerturber(scale=0.1, distribution="gamma")
