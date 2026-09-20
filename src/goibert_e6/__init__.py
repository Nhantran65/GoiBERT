"""E6 student MLM continuation utilities.

This package is intentionally isolated from the frozen E1--E5 runners.  E6
only owns corpus construction, masked-language-model continuation, and the
encoder-only export consumed by the downstream fine-tuning pilot.
"""

from .mlm import E6RobertaMLM, fixed_masked_batch, mask_batch, masked_loss

__all__ = ["E6RobertaMLM", "fixed_masked_batch", "mask_batch", "masked_loss"]
