"""E6 model factory.

E6 keeps the E1 encoder and CLS classification head unchanged.  The only
model-input change is that the encoder is loaded from the exported MLM
backbone.  The factory intentionally constructs the same ``GoiBERTClassifier``
as E1 so that the fresh head uses the same RNG sequence and parameter count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from transformers import RobertaModel

from goibert.modeling import GoiBERTClassifier, ModelSpec


E6_BACKBONE_NAME = "goiturn-s-gd-mlm-e6"


def build_e6_classifier(
    _ignored_backbone_dir: Path,
    *,
    e6_backbone_dir: Path,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    local_files_only: bool = True,
    attention_implementation: str | None = None,
) -> tuple[GoiBERTClassifier, ModelSpec]:
    """Build the E1 classifier from the final exported E6 encoder.

    ``_ignored_backbone_dir`` is retained to match the E1 factory signature;
    the caller passes the original path from the frozen E1 trainer, while this
    adapter resolves the approved E6 path explicitly.
    """

    if not e6_backbone_dir.is_dir():
        raise FileNotFoundError(f"E6 backbone directory does not exist: {e6_backbone_dir}")
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "add_pooling_layer": False,
    }
    if attention_implementation is not None:
        kwargs["attn_implementation"] = attention_implementation
    encoder = RobertaModel.from_pretrained(e6_backbone_dir, **kwargs)
    if aspect_class_counts is not None and num_classes != sum(aspect_class_counts):
        raise ValueError(
            f"num_classes={num_classes} does not match aspect_class_counts sum="
            f"{sum(aspect_class_counts)}"
        )
    model = GoiBERTClassifier(
        encoder,
        num_labels=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        backbone_name=E6_BACKBONE_NAME,
    )
    return model, model.spec()

