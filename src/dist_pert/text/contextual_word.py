"""Contextual word embedding augmentation for text perturbation."""

from nlpaug.augmenter.word import ContextualWordEmbsAug
from transformers import BertTokenizer

from dist_pert.base import BasePerturber

# Monkey-patch for transformer version compatibility (missing _convert_token_to_id).
if not hasattr(BertTokenizer, "_convert_token_to_id"):
    BertTokenizer._convert_token_to_id = BertTokenizer.convert_tokens_to_ids  # type: ignore[attr-defined]


class ContextualWordPerturber(BasePerturber[str]):
    """Perturbs text by replacing words with contextually similar alternatives.

    Uses BERT (or another masked-LM) via nlpaug to substitute a random
    fraction of tokens with words that fit the surrounding context.

    Attributes:
        model_path: HuggingFace model identifier used for contextual prediction.
        model_type: Model architecture passed to ContextualWordEmbsAug.
        aug_p: Fraction of eligible tokens to replace (0.0–1.0).
        aug_min: Minimum number of tokens to augment per string, or None.
        aug_max: Maximum number of tokens to augment per string, or None.
        device: Device string passed to the underlying model (e.g. 'cpu', 'cuda').
    """

    def __init__(
        self,
        model_path: str = "google-bert/bert-base-cased",
        model_type: str = "bert",
        aug_p: float = 0.1,
        aug_min: int | None = None,
        aug_max: int | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialise the perturber and load the underlying language model.

        Args:
            model_path: HuggingFace model path or identifier.
            model_type: Model type string passed to ContextualWordEmbsAug
                (e.g. 'bert', 'roberta', 'distilbert').
            aug_p: Proportion of tokens to replace (0.0–1.0).
            aug_min: Minimum number of tokens to augment. Passed to nlpaug
                as-is; None means no minimum is enforced.
            aug_max: Maximum number of tokens to augment. Passed to nlpaug
                as-is; None means no maximum is enforced.
            device: Compute device for the model ('cpu' or 'cuda').

        Raises:
            ValueError: If aug_p is not in [0, 1].
        """
        if not 0.0 <= aug_p <= 1.0:
            raise ValueError(f"aug_p must be in [0, 1], got {aug_p}.")

        self.model_path = model_path
        self.model_type = model_type
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.device = device

        kwargs: dict = dict(
            model_path=model_path,
            model_type=model_type,
            aug_p=aug_p,
            device=device,
        )
        if aug_min is not None:
            kwargs["aug_min"] = aug_min
        if aug_max is not None:
            kwargs["aug_max"] = aug_max

        self._aug = ContextualWordEmbsAug(**kwargs)

    def perturb(self, texts: list[str]) -> list[str]:
        """Replace a random subset of tokens in each string with contextual alternatives.

        Args:
            texts: Input strings to perturb.

        Returns:
            Perturbed strings in the same order as the input.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must be non-empty.")

        result: list[str] = self._aug.augment(texts)
        return result
