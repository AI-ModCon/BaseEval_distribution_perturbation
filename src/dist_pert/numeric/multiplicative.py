"""Multiplicative noise perturber for numeric data."""

import numpy as np

from dist_pert.base import BasePerturber

_SUPPORTED_DISTRIBUTIONS: frozenset[str] = frozenset(["lognormal", "uniform"])


class MultiplicativeNoisePerturber(BasePerturber[np.ndarray]):
    """Scales numeric arrays by a random multiplicative factor per element.

    Two distributions are supported:

    - ``lognormal``: factor = exp(N(0, scale)), giving a mean factor ≈ 1
      for small scale values.
    - ``uniform``: factor = U(1 - scale, 1 + scale).

    Attributes:
        scale: Controls the spread of the multiplicative factors.
        distribution: Name of the sampling distribution.
    """

    def __init__(self, scale: float, distribution: str = "lognormal") -> None:
        """Initialise with noise scale and distribution.

        Args:
            scale: For 'lognormal', the std of the underlying normal
                distribution. For 'uniform', the half-width of the
                interval centred on 1.
            distribution: One of 'lognormal' or 'uniform'.

        Raises:
            ValueError: If scale is negative or distribution is unsupported.
        """
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}.")
        if distribution not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {sorted(_SUPPORTED_DISTRIBUTIONS)}, "
                f"got '{distribution}'."
            )
        self.scale = scale
        self.distribution = distribution

    def perturb(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Multiply each array by per-element random factors.

        Args:
            data: List of numpy arrays (any shape). Plain floats are also
                accepted and coerced to 0-D arrays.

        Returns:
            List of scaled float64 numpy arrays with the same shapes as the input.

        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("data must be non-empty.")

        results: list[np.ndarray] = []
        for item in data:
            arr = np.asarray(item, dtype=np.float64)

            ############### sample factors ###############
            if self.distribution == "lognormal":
                factors = np.exp(np.random.normal(0.0, self.scale, arr.shape))
            else:
                factors = np.random.uniform(1.0 - self.scale, 1.0 + self.scale, arr.shape)

            results.append(arr * factors)
        return results
