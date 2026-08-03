"""Tests for text perturbers.

These tests require the `text` extra and download a BERT model on first run;
they skip gracefully when nlpaug is not installed.
"""

import pytest

pytest.importorskip("nlpaug", reason="text extra not installed")

from dist_pert.text import ContextualWordPerturber  # noqa: E402


def test_invalid_aug_p_raises() -> None:
    """An aug_p outside [0, 1] is rejected at construction time."""
    with pytest.raises(ValueError):
        ContextualWordPerturber(aug_p=1.5)


@pytest.mark.slow
def test_length_preserving_smoke() -> None:
    """Perturbing a batch returns one output per input string."""
    perturber = ContextualWordPerturber(aug_p=0.2)
    texts = ["The storm produced rotating columns of air."]
    result = perturber(texts)
    assert len(result) == len(texts)
    assert isinstance(result[0], str)
