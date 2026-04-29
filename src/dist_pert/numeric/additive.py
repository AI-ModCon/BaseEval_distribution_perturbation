"""Additive Gaussian noise perturber for numeric data."""

import numpy as np

from dist_pert.base import BasePerturber


class AdditiveGaussianPerturber(BasePerturber[np.ndarray]):
    """Adds zero-mean Gaussian noise to numeric arrays.

    Accepts numpy arrays of any shape or plain Python floats, which are
    converted to 0-D numpy arrays before perturbation.

    Attributes:
        sigma: Standard deviation of the additive Gaussian noise.
    """

    def __init__(self, sigma: float) -> None:
        """Initialise with the desired noise level.

        Args:
            sigma: Standard deviation of the Gaussian noise term added to
                each element.

        Raises:
            ValueError: If sigma is negative.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}.")
        self.sigma = sigma

    def perturb(self, data: list[np.ndarray | float]) -> list[np.ndarray]:
        """Add Gaussian noise to each array in the batch.

        Args:
            data: List of numpy arrays (any shape) or plain floats.

        Returns:
            List of noisy float64 numpy arrays with the same shapes as the input.

        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("data must be non-empty.")

        results: list[np.ndarray] = []
        for item in data:
            arr = np.asarray(item, dtype=np.float64)
            noise = np.random.normal(0.0, self.sigma, arr.shape)
            results.append(arr + noise)
        return results
