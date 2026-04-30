"""ImageNet-C style corruption perturber with optional imagecorruptions backend."""

import io
from typing import Callable

import numpy as np

from dist_pert.base import BasePerturber

try:
    from imagecorruptions import corrupt as _ic_corrupt

    _IMAGECORRUPTIONS_AVAILABLE = True
except Exception:
    _IMAGECORRUPTIONS_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# Severity-indexed parameters for PIL/numpy fallback implementations.
# Index 0 = severity 1, index 4 = severity 5.
_BLUR_RADII = [1, 2, 3, 4, 6]
_JPEG_QUALITIES = [25, 18, 15, 10, 7]
_SHOT_SCALES = [60, 25, 12, 5, 3]
_BRIGHTNESS_FACTORS = [1.1, 1.2, 1.5, 2.0, 2.5]
_CONTRAST_FACTORS = [1.1, 1.2, 1.5, 2.0, 2.5]


def _gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """Apply Gaussian blur via PIL."""
    pil_img = Image.fromarray(image)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=_BLUR_RADII[severity - 1]))
    return np.array(blurred)


def _jpeg_compression(image: np.ndarray, severity: int) -> np.ndarray:
    """Simulate JPEG compression artefacts via PIL."""
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=_JPEG_QUALITIES[severity - 1])
    buf.seek(0)
    return np.array(Image.open(buf))


def _shot_noise(image: np.ndarray, severity: int) -> np.ndarray:
    """Add Poisson (shot) noise via numpy."""
    scale = _SHOT_SCALES[severity - 1]
    float_img = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(float_img * scale) / scale
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)


def _brightness(image: np.ndarray, severity: int) -> np.ndarray:
    """Increase brightness via PIL."""
    pil_img = Image.fromarray(image)
    enhanced = ImageEnhance.Brightness(pil_img).enhance(_BRIGHTNESS_FACTORS[severity - 1])
    return np.array(enhanced)


def _contrast(image: np.ndarray, severity: int) -> np.ndarray:
    """Increase contrast via PIL."""
    pil_img = Image.fromarray(image)
    enhanced = ImageEnhance.Contrast(pil_img).enhance(_CONTRAST_FACTORS[severity - 1])
    return np.array(enhanced)


# Registry mapping corruption name → fallback implementation.
# Signature: (image: np.ndarray, severity: int) -> np.ndarray.
# Add entries here to support new corruption types without subclassing.
CORRUPTION_FNS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_blur": _gaussian_blur,
    "jpeg_compression": _jpeg_compression,
    "shot_noise": _shot_noise,
    "brightness": _brightness,
    "contrast": _contrast,
}

SUPPORTED_CORRUPTIONS: frozenset[str] = frozenset(CORRUPTION_FNS)


class ImageCorruptionPerturber(BasePerturber[np.ndarray]):
    """Applies ImageNet-C style corruptions to images.

    Uses the `imagecorruptions` package when available; falls back to
    PIL/numpy implementations otherwise.

    New corruption types can be registered at runtime by inserting a callable
    into CORRUPTION_FNS with the signature (np.ndarray, int) -> np.ndarray.

    Attributes:
        corruption: Name of the corruption to apply.
        severity: Corruption severity level (1–5, where 5 is most severe).
    """

    def __init__(self, corruption: str, severity: int = 1) -> None:
        """Initialise the corruption perturber.

        Args:
            corruption: Name of the corruption. Must be a key in CORRUPTION_FNS
                or a name supported by the imagecorruptions package.
            severity: Severity level between 1 and 5 inclusive.

        Raises:
            ValueError: If corruption is unknown or severity is out of range.
        """
        if not 1 <= severity <= 5:
            raise ValueError(f"severity must be between 1 and 5, got {severity}.")
        if not _IMAGECORRUPTIONS_AVAILABLE and corruption not in CORRUPTION_FNS:
            raise ValueError(
                f"Unknown corruption '{corruption}'. "
                f"Available: {sorted(CORRUPTION_FNS)}."
            )
        self.corruption = corruption
        self.severity = severity

    def perturb(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the configured corruption to each image in the batch.

        Args:
            data: List of H×W×C uint8 numpy arrays.

        Returns:
            Corrupted images as uint8 numpy arrays.

        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("data must be non-empty.")

        results: list[np.ndarray] = []
        for image in data:

            ############### imagecorruptions backend ###############
            if _IMAGECORRUPTIONS_AVAILABLE:
                results.append(
                    _ic_corrupt(image, corruption_name=self.corruption, severity=self.severity)
                )
                continue

            ############### pil/numpy fallback ###############
            fn = CORRUPTION_FNS[self.corruption]
            results.append(fn(image, self.severity))

        return results
