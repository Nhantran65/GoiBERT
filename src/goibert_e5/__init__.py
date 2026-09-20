"""E5 mean-pooling screening adapter.

The package is intentionally separate from :mod:`goibert`: E1 remains frozen
and E5 only changes the vector supplied to the existing classification head.
"""

from .modeling import MeanPooledGoiBERTClassifier, build_mean_pool_classifier

__all__ = ["MeanPooledGoiBERTClassifier", "build_mean_pool_classifier"]
