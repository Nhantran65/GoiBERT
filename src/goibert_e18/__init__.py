"""E18 screening 4B: segment-fork-consistency training on the E2/E7 student.

This package only adds a model wrapper (``modeling.py``) and a loss patch
factory (``losses.py``).  Data loading, the training loop, the optimizer
group construction, and the frozen head/backbone all remain the E1/E2/E7
implementations; nothing here reimplements them.
"""

from __future__ import annotations

__all__ = ["modeling", "losses"]
