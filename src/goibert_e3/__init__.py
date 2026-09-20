"""E3 conditional polarity-loss screening.

E3 is deliberately kept in its own package.  The frozen E1/E2 packages are
imported for data, model, optimizer, scheduler, and evaluation contracts, but
their source files are never modified by this experiment.
"""

from .losses import (
    E3LossComponents,
    combined_polarity_loss,
    conditional_polarity_cross_entropy,
)

__all__ = [
    "E3LossComponents",
    "combined_polarity_loss",
    "conditional_polarity_cross_entropy",
]
