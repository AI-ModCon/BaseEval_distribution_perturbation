"""dist_pert — distribution perturbation library.

Provides first-order perturbation classes for text, image, and numeric data.
All perturbers share the BasePerturber interface: instantiate with config,
then call on a list of samples.

Example::

    from dist_pert.text import ContextualWordPerturber
    perturber = ContextualWordPerturber(aug_p=0.1)
    result = perturber(["The storm produced 200 mph winds."])
"""

from dist_pert.base import BasePerturber

__all__ = ["BasePerturber"]
