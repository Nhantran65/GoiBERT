"""E2 loss functions.

The E2 runner deliberately keeps its loss contract in a small, dependency-free
module.  There are three pieces:

* :func:`logits_kd_loss` is logits-only knowledge distillation.  The teacher
  distribution is the target of ``KL(teacher || student)`` and the returned
  value includes the usual ``T**2`` factor.
* :func:`noise_consistency_loss` is a symmetric KL between a clean and a noisy
  student view.  Neither view is detached, so both forward passes receive a
  gradient.
* :func:`combined_kd_noise_loss` combines clean hard-label CE, task KD, and
  noise consistency for the E2c condition.

The helpers understand both ordinary categorical logits ``[batch, classes]``
and aspect-slot logits.  Slot logits can be padded ``[batch, slots, width]``
or packed ``[batch, sum(widths)]``.  Passing ``(4,)*10 + (2,)`` is therefore
the explicit ViSFD policy: ten four-class slots plus a binary OTHERS slot.
All probability arithmetic is forced to FP32, even when the caller is inside a
BF16/FP16 autocast region.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


DEFAULT_KD_TEMPERATURE = 2.0
DEFAULT_KD_ALPHA = 0.5
DEFAULT_NC_TEMPERATURE = 1.0
DEFAULT_NC_WEIGHT = 1.0
DEFAULT_IGNORE_INDEX = -100


def _fp32_context(tensor: Tensor):
    """Disable an outer autocast context for numerically sensitive losses."""

    # ``torch.autocast`` supports both CUDA and CPU.  Keeping this helper
    # device-local avoids changing the global autocast state.
    if tensor.device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0 or not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return value


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1; got {alpha!r}")
    return alpha


def _normalise_slot_counts(
    logits: Tensor,
    slot_class_counts: Sequence[int] | None,
) -> tuple[int, ...] | None:
    if slot_class_counts is None:
        if logits.ndim == 3:
            if logits.shape[1] <= 0 or logits.shape[2] <= 0:
                raise ValueError("slot logits must have positive slot and class dimensions")
            return (int(logits.shape[2]),) * int(logits.shape[1])
        return None
    counts = tuple(int(width) for width in slot_class_counts)
    # E1's task helper naturally returns ``()`` for categorical tasks.  Treat
    # that explicit empty schema exactly like ``None`` so a shared runner can
    # pass the task schema without branching on task type.
    if not counts:
        return None
    if any(width <= 0 for width in counts):
        raise ValueError("slot_class_counts must contain positive widths")
    return counts


def _split_slot_logits(
    logits: Tensor,
    slot_class_counts: Sequence[int],
) -> list[Tensor]:
    """Split padded or packed slot logits while dropping invalid classes."""

    counts = tuple(int(width) for width in slot_class_counts)
    if logits.ndim == 3:
        if logits.shape[1] != len(counts):
            raise ValueError(
                "slot logits have a different number of slots than slot_class_counts: "
                f"{logits.shape[1]} != {len(counts)}"
            )
        if logits.shape[2] < max(counts):
            raise ValueError(
                "padded slot logits do not have enough class columns for the declared widths"
            )
        return [logits[:, index, :width] for index, width in enumerate(counts)]
    if logits.ndim == 2:
        expected = sum(counts)
        if logits.shape[1] != expected:
            raise ValueError(
                "packed slot logits must have sum(slot_class_counts) columns: "
                f"{logits.shape[1]} != {expected}"
            )
        output: list[Tensor] = []
        offset = 0
        for width in counts:
            output.append(logits[:, offset : offset + width])
            offset += width
        return output
    raise ValueError("slot logits must have shape [batch, slots, classes] or [batch, packed_classes]")


def _slot_masks(
    valid_slot_mask: Tensor | None,
    *,
    batch_size: int,
    slot_count: int,
    device: torch.device,
) -> list[Tensor]:
    if valid_slot_mask is None:
        return [torch.ones(batch_size, dtype=torch.bool, device=device)] * slot_count
    if valid_slot_mask.shape != (batch_size, slot_count):
        raise ValueError(
            "valid_slot_mask must have shape [batch, slots]: "
            f"{tuple(valid_slot_mask.shape)} != {(batch_size, slot_count)}"
        )
    return [valid_slot_mask[:, index].to(device=device, dtype=torch.bool) for index in range(slot_count)]


def _check_same_shape(left: Tensor, right: Tensor, name: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{name} tensors must have identical shape: {left.shape} != {right.shape}")
    if left.device != right.device:
        raise ValueError(f"{name} tensors must be on the same device")


def _check_slot_labels(
    labels: Tensor,
    counts: Sequence[int],
    *,
    ignore_index: int,
    valid_slot_mask: Tensor | None,
) -> None:
    if labels.ndim != 2 or labels.shape[1] != len(counts):
        raise ValueError(
            "slot labels must have shape [batch, slots] matching slot_class_counts"
        )
    if not labels.dtype.is_floating_point and labels.dtype != torch.bool:
        integer_labels = labels
    else:
        raise ValueError("labels must be an integer tensor")
    for index, width in enumerate(counts):
        values = integer_labels[:, index]
        active = values != ignore_index
        if valid_slot_mask is not None:
            active = active & valid_slot_mask[:, index].to(device=values.device, dtype=torch.bool)
        if active.any() and ((values[active] < 0) | (values[active] >= width)).any():
            raise ValueError(
                f"slot {index} contains a class outside [0, {width})"
            )


def _directional_kl(
    target_logits: Tensor,
    reference_logits: Tensor,
    *,
    temperature: float,
    valid_rows: Tensor | None = None,
    detach_target: bool = False,
) -> Tensor:
    """Return ``KL(target || reference)`` for one categorical tensor."""

    target = target_logits.float()
    reference = reference_logits.float()
    if detach_target:
        target = target.detach()
    if valid_rows is not None:
        valid_rows = valid_rows.to(device=target.device, dtype=torch.bool)
        target = target[valid_rows]
        reference = reference[valid_rows]
    if target.shape[0] == 0:
        # Keep the result connected to the graph while contributing no loss.
        return (target_logits.sum() + reference_logits.sum()) * 0.0
    target_log_probs = F.log_softmax(target / temperature, dim=-1)
    reference_log_probs = F.log_softmax(reference / temperature, dim=-1)
    target_probs = target_log_probs.exp()
    return (target_probs * (target_log_probs - reference_log_probs)).sum(dim=-1).mean()


def supervised_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    slot_class_counts: Sequence[int] | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    valid_slot_mask: Tensor | None = None,
) -> Tensor:
    """Hard-label CE for categorical or variable-width slot outputs.

    Absent labels (class ``0``) are ordinary targets and are included in the
    mean.  A label is ignored only when it is explicitly ``ignore_index`` or
    when its slot is masked by ``valid_slot_mask``.
    """

    counts = _normalise_slot_counts(logits, slot_class_counts)
    with _fp32_context(logits):
        if counts is None:
            if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
                raise ValueError("categorical logits/labels must have shapes [batch, classes]/[batch]")
            active = labels != ignore_index
            if active.any() and ((labels[active] < 0) | (labels[active] >= logits.shape[1])).any():
                raise ValueError("categorical labels contain a class outside the logits")
            if valid_slot_mask is not None:
                if valid_slot_mask.ndim != 1 or valid_slot_mask.shape[0] != labels.shape[0]:
                    raise ValueError("categorical valid_slot_mask must have shape [batch]")
                active = active & valid_slot_mask.to(device=labels.device, dtype=torch.bool)
            if not active.any():
                return logits.sum() * 0.0
            return F.cross_entropy(
                logits.float()[active], labels.long()[active], reduction="mean"
            )

        if labels.ndim != 2 or labels.shape[0] != logits.shape[0]:
            raise ValueError("slot labels must have shape [batch, slots]")
        _check_slot_labels(
            labels,
            counts,
            ignore_index=ignore_index,
            valid_slot_mask=valid_slot_mask,
        )
        slots = _split_slot_logits(logits, counts)
        masks = _slot_masks(
            valid_slot_mask,
            batch_size=logits.shape[0],
            slot_count=len(counts),
            device=logits.device,
        )
        losses: list[Tensor] = []
        for index, (slot_logits, slot_mask) in enumerate(zip(slots, masks)):
            slot_labels = labels[:, index].long()
            active = (slot_labels != ignore_index) & slot_mask
            if active.any():
                losses.append(
                    F.cross_entropy(
                        slot_logits.float()[active], slot_labels[active], reduction="mean"
                    )
                )
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


# A descriptive alias for callers that use the E1 terminology.
masked_slot_cross_entropy = supervised_cross_entropy


def logits_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = DEFAULT_KD_TEMPERATURE,
    slot_class_counts: Sequence[int] | None = None,
    valid_slot_mask: Tensor | None = None,
) -> Tensor:
    """Logits-only ``KL(teacher || student) * temperature**2``.

    The teacher is detached by design; callers should also run the teacher
    forward under ``torch.inference_mode``.  No hidden states or attention
    tensors are inspected by this function.
    """

    temperature = _positive_float(temperature, "temperature")
    _check_same_shape(student_logits, teacher_logits, "student/teacher logits")
    counts = _normalise_slot_counts(student_logits, slot_class_counts)
    with _fp32_context(student_logits):
        if counts is None:
            if student_logits.ndim != 2:
                raise ValueError("categorical KD logits must have shape [batch, classes]")
            masks = None
            if valid_slot_mask is not None:
                if valid_slot_mask.ndim != 1 or valid_slot_mask.shape[0] != student_logits.shape[0]:
                    raise ValueError("categorical valid_slot_mask must have shape [batch]")
                masks = valid_slot_mask
            value = _directional_kl(
                teacher_logits,
                student_logits,
                temperature=temperature,
                valid_rows=masks,
                detach_target=True,
            )
        else:
            student_slots = _split_slot_logits(student_logits, counts)
            teacher_slots = _split_slot_logits(teacher_logits, counts)
            masks = _slot_masks(
                valid_slot_mask,
                batch_size=student_logits.shape[0],
                slot_count=len(counts),
                device=student_logits.device,
            )
            values = [
                _directional_kl(
                    teacher_slot,
                    student_slot,
                    temperature=temperature,
                    valid_rows=slot_mask,
                    detach_target=True,
                )
                for teacher_slot, student_slot, slot_mask in zip(
                    teacher_slots, student_slots, masks
                )
            ]
            value = torch.stack(values).mean()
        return value * (temperature**2)


# Short aliases make the contract convenient for a runner without weakening
# the explicit name above.
task_kd_logits_loss = logits_kd_loss
kd_loss = logits_kd_loss
# Names used by the first E2 runner draft.  These aliases retain the complete
# contract, including the T**2 multiplier; the runner must not multiply the
# returned value by T**2 a second time.
logit_distillation_loss = logits_kd_loss
logit_kd_loss = logits_kd_loss


def task_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    alpha: float = DEFAULT_KD_ALPHA,
    temperature: float = DEFAULT_KD_TEMPERATURE,
    slot_class_counts: Sequence[int] | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    valid_slot_mask: Tensor | None = None,
) -> Tensor:
    """Return ``(1-alpha) * CE + alpha * logits_kd_loss``."""

    alpha = _validate_alpha(alpha)
    ce = supervised_cross_entropy(
        student_logits,
        labels,
        slot_class_counts=slot_class_counts,
        ignore_index=ignore_index,
        valid_slot_mask=valid_slot_mask,
    )
    kd = logits_kd_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        slot_class_counts=slot_class_counts,
        valid_slot_mask=valid_slot_mask,
    )
    return (1.0 - alpha) * ce + alpha * kd


def task_kd_components(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    alpha: float = DEFAULT_KD_ALPHA,
    temperature: float = DEFAULT_KD_TEMPERATURE,
    slot_class_counts: Sequence[int] | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    valid_slot_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return CE, KD and total tensors for loss logging."""

    alpha = _validate_alpha(alpha)
    ce = supervised_cross_entropy(
        student_logits,
        labels,
        slot_class_counts=slot_class_counts,
        ignore_index=ignore_index,
        valid_slot_mask=valid_slot_mask,
    )
    kd = logits_kd_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        slot_class_counts=slot_class_counts,
        valid_slot_mask=valid_slot_mask,
    )
    return {"ce": ce, "kd": kd, "total": (1.0 - alpha) * ce + alpha * kd}


def symmetric_kl_consistency_loss(
    clean_logits: Tensor,
    noisy_logits: Tensor,
    *,
    temperature: float = DEFAULT_NC_TEMPERATURE,
    slot_class_counts: Sequence[int] | None = None,
    valid_slot_mask: Tensor | None = None,
) -> Tensor:
    """Return ``0.5 KL(clean || noisy) + 0.5 KL(noisy || clean)``.

    This is deliberately separate from KD.  Both ``clean_logits`` and
    ``noisy_logits`` remain attached to the graph, and there is no ``T**2``
    multiplier.  The default temperature is one as required by the E2 NC
    contract.
    """

    temperature = _positive_float(temperature, "temperature")
    _check_same_shape(clean_logits, noisy_logits, "clean/noisy logits")
    counts = _normalise_slot_counts(clean_logits, slot_class_counts)
    with _fp32_context(clean_logits):
        if counts is None:
            if clean_logits.ndim != 2:
                raise ValueError("categorical NC logits must have shape [batch, classes]")
            if valid_slot_mask is not None:
                if valid_slot_mask.ndim != 1 or valid_slot_mask.shape[0] != clean_logits.shape[0]:
                    raise ValueError("categorical valid_slot_mask must have shape [batch]")
            clean_to_noisy = _directional_kl(
                clean_logits,
                noisy_logits,
                temperature=temperature,
                valid_rows=valid_slot_mask,
                detach_target=False,
            )
            noisy_to_clean = _directional_kl(
                noisy_logits,
                clean_logits,
                temperature=temperature,
                valid_rows=valid_slot_mask,
                detach_target=False,
            )
        else:
            clean_slots = _split_slot_logits(clean_logits, counts)
            noisy_slots = _split_slot_logits(noisy_logits, counts)
            masks = _slot_masks(
                valid_slot_mask,
                batch_size=clean_logits.shape[0],
                slot_count=len(counts),
                device=clean_logits.device,
            )
            clean_to_noisy = torch.stack(
                [
                    _directional_kl(
                        clean_slot,
                        noisy_slot,
                        temperature=temperature,
                        valid_rows=slot_mask,
                        detach_target=False,
                    )
                    for clean_slot, noisy_slot, slot_mask in zip(
                        clean_slots, noisy_slots, masks
                    )
                ]
            ).mean()
            noisy_to_clean = torch.stack(
                [
                    _directional_kl(
                        noisy_slot,
                        clean_slot,
                        temperature=temperature,
                        valid_rows=slot_mask,
                        detach_target=False,
                    )
                    for noisy_slot, clean_slot, slot_mask in zip(
                        noisy_slots, clean_slots, masks
                    )
                ]
            ).mean()
        return 0.5 * (clean_to_noisy + noisy_to_clean)


noise_consistency_loss = symmetric_kl_consistency_loss
symmetric_kl = symmetric_kl_consistency_loss


def combined_noise_loss(
    clean_logits: Tensor,
    noisy_logits: Tensor,
    labels: Tensor,
    *,
    noise_weight: float = DEFAULT_NC_WEIGHT,
    temperature: float = DEFAULT_NC_TEMPERATURE,
    slot_class_counts: Sequence[int] | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    valid_slot_mask: Tensor | None = None,
) -> Tensor:
    """Return ``clean CE + noise_weight * symmetric_KL(clean, noisy)``."""

    noise_weight = float(noise_weight)
    if noise_weight < 0.0 or not torch.isfinite(torch.tensor(noise_weight)):
        raise ValueError("noise_weight must be finite and non-negative")
    ce = supervised_cross_entropy(
        clean_logits,
        labels,
        slot_class_counts=slot_class_counts,
        ignore_index=ignore_index,
        valid_slot_mask=valid_slot_mask,
    )
    consistency = symmetric_kl_consistency_loss(
        clean_logits,
        noisy_logits,
        temperature=temperature,
        slot_class_counts=slot_class_counts,
        valid_slot_mask=valid_slot_mask,
    )
    return ce + noise_weight * consistency


def combined_kd_noise_loss(
    clean_logits: Tensor,
    noisy_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    alpha: float = DEFAULT_KD_ALPHA,
    noise_weight: float = DEFAULT_NC_WEIGHT,
    kd_temperature: float = DEFAULT_KD_TEMPERATURE,
    nc_temperature: float = DEFAULT_NC_TEMPERATURE,
    slot_class_counts: Sequence[int] | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    valid_slot_mask: Tensor | None = None,
) -> Tensor:
    """Return E2c ``(1-alpha)CE + alpha KD + lambda NC``."""

    alpha = _validate_alpha(alpha)
    noise_weight = float(noise_weight)
    if noise_weight < 0.0 or not torch.isfinite(torch.tensor(noise_weight)):
        raise ValueError("noise_weight must be finite and non-negative")
    ce = supervised_cross_entropy(
        clean_logits,
        labels,
        slot_class_counts=slot_class_counts,
        ignore_index=ignore_index,
        valid_slot_mask=valid_slot_mask,
    )
    kd = logits_kd_loss(
        clean_logits,
        teacher_logits,
        temperature=kd_temperature,
        slot_class_counts=slot_class_counts,
        valid_slot_mask=valid_slot_mask,
    )
    consistency = symmetric_kl_consistency_loss(
        clean_logits,
        noisy_logits,
        temperature=nc_temperature,
        slot_class_counts=slot_class_counts,
        valid_slot_mask=valid_slot_mask,
    )
    return (1.0 - alpha) * ce + alpha * kd + noise_weight * consistency


def combined_kd_noise_components(
    clean_logits: Tensor,
    noisy_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    alpha: float = DEFAULT_KD_ALPHA,
    noise_weight: float = DEFAULT_NC_WEIGHT,
    kd_temperature: float = DEFAULT_KD_TEMPERATURE,
    nc_temperature: float = DEFAULT_NC_TEMPERATURE,
    slot_class_counts: Sequence[int] | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    valid_slot_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return CE, KD, NC and total tensors for E2c logging."""

    alpha = _validate_alpha(alpha)
    noise_weight = float(noise_weight)
    if noise_weight < 0.0 or not torch.isfinite(torch.tensor(noise_weight)):
        raise ValueError("noise_weight must be finite and non-negative")
    ce = supervised_cross_entropy(
        clean_logits,
        labels,
        slot_class_counts=slot_class_counts,
        ignore_index=ignore_index,
        valid_slot_mask=valid_slot_mask,
    )
    kd = logits_kd_loss(
        clean_logits,
        teacher_logits,
        temperature=kd_temperature,
        slot_class_counts=slot_class_counts,
        valid_slot_mask=valid_slot_mask,
    )
    consistency = symmetric_kl_consistency_loss(
        clean_logits,
        noisy_logits,
        temperature=nc_temperature,
        slot_class_counts=slot_class_counts,
        valid_slot_mask=valid_slot_mask,
    )
    total = (1.0 - alpha) * ce + alpha * kd + noise_weight * consistency
    return {"ce": ce, "kd": kd, "nc": consistency, "total": total}


__all__ = [
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_KD_ALPHA",
    "DEFAULT_KD_TEMPERATURE",
    "DEFAULT_NC_TEMPERATURE",
    "DEFAULT_NC_WEIGHT",
    "combined_kd_noise_components",
    "combined_kd_noise_loss",
    "combined_noise_loss",
    "kd_loss",
    "logit_distillation_loss",
    "logit_kd_loss",
    "logits_kd_loss",
    "masked_slot_cross_entropy",
    "noise_consistency_loss",
    "symmetric_kl",
    "symmetric_kl_consistency_loss",
    "supervised_cross_entropy",
    "task_kd_components",
    "task_kd_logits_loss",
    "task_kd_loss",
]
