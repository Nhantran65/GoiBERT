"""Losses for the E3 conditional polarity supervision experiment.

The E1 loss remains the primary objective.  E3 adds a second cross-entropy
term over only the polarity classes of gold-present aspect slots:

    L_E3 = L_E1 + 0.5 * CE(logits[gold > 0, 1:4], gold - 1)

The absent logit is removed from the auxiliary term.  A ViSFD OTHERS slot has
two classes and no polarity, so it is excluded explicitly by the slot list
passed by the runner.  Empty active sets return a graph-connected zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from goibert.train import compute_loss as e1_compute_loss


POLARITY_WEIGHT = 0.5
DEFAULT_POLARITY_SLOTS = None


@dataclass(frozen=True)
class E3LossComponents:
    """Differentiable E3 objective and its two observable components."""

    total: Tensor
    base_ce: Tensor
    polarity_ce: Tensor
    active_slot_count: int


def _normalise_counts(
    logits: Tensor,
    labels: Tensor,
    *,
    num_aspects: int,
    aspect_class_counts: Sequence[int],
) -> tuple[int, ...]:
    counts = tuple(int(value) for value in aspect_class_counts)
    if num_aspects <= 0:
        raise ValueError("E3 conditional polarity loss requires an ABSA task")
    if len(counts) != num_aspects:
        raise ValueError("aspect_class_counts must contain one entry per aspect")
    if any(value <= 0 for value in counts):
        raise ValueError("aspect_class_counts entries must be positive")
    if labels.ndim != 2 or labels.shape[1] != num_aspects:
        raise ValueError("ABSA labels must have shape [batch, aspects]")
    if logits.ndim == 3:
        if len(set(counts)) != 1 or logits.shape != (labels.shape[0], num_aspects, counts[0]):
            raise ValueError("3D logits must have uniform slot widths matching labels")
    elif logits.ndim == 2:
        if logits.shape != (labels.shape[0], sum(counts)):
            raise ValueError("Flat logits must have shape [batch, sum(slot widths)]")
    else:
        raise ValueError("ABSA logits must have shape [batch, aspects, classes] or flat form")
    return counts


def _split_logits(logits: Tensor, counts: Sequence[int]) -> tuple[Tensor, ...]:
    if logits.ndim == 3:
        return tuple(logits[:, index, :] for index in range(len(counts)))
    offsets: list[Tensor] = []
    start = 0
    for width in counts:
        offsets.append(logits[:, start : start + width])
        start += width
    return tuple(offsets)


def _resolve_polarity_slots(
    counts: Sequence[int], polarity_slots: Sequence[int] | None
) -> tuple[int, ...]:
    if polarity_slots is None:
        # Only four-class slots carry the source's absent + three polarities.
        # This conservative default naturally excludes ViSFD's binary OTHERS
        # slot, while the runner records the resolved indices explicitly.
        slots = tuple(index for index, width in enumerate(counts) if int(width) == 4)
    else:
        slots = tuple(int(index) for index in polarity_slots)
    if len(set(slots)) != len(slots):
        raise ValueError("polarity_slots must not contain duplicates")
    if any(index < 0 or index >= len(counts) for index in slots):
        raise ValueError("polarity_slots contains an out-of-range slot")
    if any(int(counts[index]) != 4 for index in slots):
        raise ValueError("Every polarity slot must have exactly four classes")
    return slots


def conditional_polarity_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    num_aspects: int,
    aspect_class_counts: Sequence[int],
    polarity_slots: Sequence[int] | None = DEFAULT_POLARITY_SLOTS,
) -> tuple[Tensor, int]:
    """Compute pooled CE on gold-present polarity slots.

    For every selected four-class slot, class 0 is absent and classes 1..3
    are the source polarity IDs.  Active labels therefore use ``gold - 1``
    against logits ``[..., 1:4]``.  The mean is pooled over all active cells,
    rather than averaging slot means.  When a batch contains no active cells,
    the returned zero remains connected to ``logits`` and has zero gradient.
    """

    counts = _normalise_counts(
        logits,
        labels,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
    )
    slots = _resolve_polarity_slots(counts, polarity_slots)
    split = _split_logits(logits, counts)
    active_logits: list[Tensor] = []
    active_targets: list[Tensor] = []
    active_count = 0
    for index in slots:
        slot_labels = labels[:, index].long()
        active = slot_labels > 0
        if active.any():
            present = slot_labels[active]
            if ((present < 1) | (present > 3)).any():
                raise ValueError(
                    f"Polarity slot {index} contains a label outside source IDs 1..3"
                )
            active_logits.append(split[index][active, 1:4])
            active_targets.append(present - 1)
            active_count += int(active.sum().item())
    if not active_logits:
        return logits.sum() * 0.0, 0
    # Compute in fp32 for a stable auxiliary objective under BF16 autocast.
    pooled_logits = torch.cat(active_logits, dim=0).float()
    pooled_targets = torch.cat(active_targets, dim=0)
    return F.cross_entropy(pooled_logits, pooled_targets, reduction="mean"), active_count


def combined_polarity_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    num_aspects: int,
    aspect_class_counts: Sequence[int],
    polarity_weight: float = POLARITY_WEIGHT,
    polarity_slots: Sequence[int] | None = DEFAULT_POLARITY_SLOTS,
) -> E3LossComponents:
    """Return E1 CE plus the declared E3 conditional polarity term."""

    weight = float(polarity_weight)
    if weight < 0.0:
        raise ValueError("polarity_weight must be non-negative")
    counts = tuple(int(value) for value in aspect_class_counts)
    base = e1_compute_loss(
        logits,
        labels,
        num_aspects=num_aspects,
        aspect_class_counts=counts,
    )
    polarity, active_count = conditional_polarity_cross_entropy(
        logits,
        labels,
        num_aspects=num_aspects,
        aspect_class_counts=counts,
        polarity_slots=polarity_slots,
    )
    return E3LossComponents(
        total=base + weight * polarity,
        base_ce=base,
        polarity_ce=polarity,
        active_slot_count=active_count,
    )
