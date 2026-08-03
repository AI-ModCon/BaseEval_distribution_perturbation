"""Abstract base class for all dist_pert perturbers."""

from abc import ABC, abstractmethod


class BasePerturber[T](ABC):
    """Abstract base for all perturbation classes.

    Subclasses must implement perturb(). The __call__ method delegates to it
    so instances can be used as callables directly.

    Type parameter T is the element type: str for text perturbers,
    np.ndarray for image and numeric perturbers.
    """

    @abstractmethod
    def perturb(self, data: list[T]) -> list[T]:
        """Apply perturbation to a batch of samples.

        Args:
            data: Batch of samples to perturb. Must be non-empty.

        Returns:
            Perturbed samples in the same order and length as the input.
        """
        ...

    def __call__(self, data: list[T]) -> list[T]:
        """Apply perturbation by delegating to perturb().

        Args:
            data: Batch of samples to perturb.

        Returns:
            Perturbed samples in the same order and length as the input.
        """
        return self.perturb(data)
