"""Pixel-level noise perturbers for image data."""

import numpy as np

from dist_pert.base import BasePerturber


class GaussianNoisePerturber(BasePerturber[np.ndarray]):
    """Adds zero-mean Gaussian noise to each pixel of an image.

    Works with uint8 images (values 0–255) and float32 images (values 0–1).
    Output is clipped to the valid range of the input dtype.

    Attributes:
        sigma: Standard deviation of the Gaussian noise applied to pixel values.
    """

    def __init__(self, sigma: float) -> None:
        """Initialise with the desired noise level.

        Args:
            sigma: Std deviation of the additive Gaussian noise. For uint8
                images a value between 5 and 50 is typical; for float32
                images use values between 0.01 and 0.2.

        Raises:
            ValueError: If sigma is negative.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}.")
        self.sigma = sigma

    def perturb(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Add Gaussian noise to each image in the batch.

        Args:
            data: List of images as H×W×C numpy arrays (uint8 or float32).

        Returns:
            Noisy images clipped to the valid range of the input dtype.

        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("data must be non-empty.")

        results: list[np.ndarray] = []
        for image in data:
            noise = np.random.normal(0, self.sigma, image.shape)
            noisy = image.astype(np.float64) + noise
            if image.dtype == np.uint8:
                results.append(np.clip(noisy, 0, 255).astype(np.uint8))
            else:
                results.append(np.clip(noisy, 0.0, 1.0).astype(image.dtype))
        return results


class SaltPepperPerturber(BasePerturber[np.ndarray]):
    """Randomly sets pixels to the minimum or maximum value (salt-and-pepper noise).

    Attributes:
        amount: Fraction of pixels to corrupt (0.0–1.0).
    """

    def __init__(self, amount: float) -> None:
        """Initialise with the desired corruption fraction.

        Args:
            amount: Proportion of pixels to corrupt. Half become salt (max
                value), half become pepper (min value).

        Raises:
            ValueError: If amount is not in [0, 1].
        """
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"amount must be in [0, 1], got {amount}.")
        self.amount = amount

    def perturb(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Apply salt-and-pepper noise to each image in the batch.

        Args:
            data: List of images as H×W×C numpy arrays (uint8 or float32).

        Returns:
            Images with randomly corrupted pixels.

        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("data must be non-empty.")

        results: list[np.ndarray] = []
        for image in data:
            out = image.copy()
            max_val: int | float = 255 if image.dtype == np.uint8 else 1.0
            num_pixels = image.shape[0] * image.shape[1]
            num_corrupt = int(self.amount * num_pixels)

            ############### salt ###############
            salt_rows = np.random.randint(0, image.shape[0], num_corrupt // 2)
            salt_cols = np.random.randint(0, image.shape[1], num_corrupt // 2)
            out[salt_rows, salt_cols] = max_val

            ############### pepper ###############
            pepper_rows = np.random.randint(0, image.shape[0], num_corrupt // 2)
            pepper_cols = np.random.randint(0, image.shape[1], num_corrupt // 2)
            out[pepper_rows, pepper_cols] = 0

            results.append(out)
        return results
