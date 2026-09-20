"""E18 loss patch: fork-consistency CE layered on top of the frozen compute_loss.

Only used by ``scripts/run_e18.py``, which monkeypatches
``goibert_e2.train.compute_loss`` with the callable built by
:func:`make_patched_compute_loss` for the duration of one worker process.
The ordinary CE is delegated to the frozen ``goibert.train.compute_loss``
(the caller must pass in a reference captured *before* any monkeypatching);
NC consistency on CLS logits is untouched -- it is computed by the frozen
E2 ``_train_loop`` itself, outside this function, on the CLS logits this
module returns unchanged.

Split scoring/acceptance/assignment (score(s), argmax over splits, gold ->
fork assignment) is done entirely on detached, CPU-resident Python floats so
no gradient flows through the selection.  The fork cross-entropy terms that
*do* carry gradient are built from the original (device-resident,
autograd-connected) fork logits tensor stashed on the model by
``E18Classifier.forward``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def lambda_fork_ramp(lambda_fork: float, epoch: int, ramp_epochs: int) -> float:
    """Linear warmup of ``lambda_fork`` over the first ``ramp_epochs`` epochs.

    ``lambda_eff(epoch) = lambda_fork * min(1, epoch / ramp_epochs)``, with
    ``epoch`` the 1-based training epoch (epoch 1 -> ``lambda_fork/ramp_epochs``,
    epoch >= ramp_epochs -> ``lambda_fork``).  Rationale: with a fresh random
    head the split selection is noise early in training, so the fork loss is
    phased in rather than applied at full weight from epoch 1.
    """

    if ramp_epochs <= 0:
        return float(lambda_fork)
    return float(lambda_fork) * min(1.0, float(epoch) / float(ramp_epochs))


class ForkTelemetry:
    """Accumulates per-batch fork-loss stats for one epoch; reset on flush."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.considered = 0
        self.accepted = 0
        self.connector_hits = 0
        self.fork_ce_sum = 0.0
        self.fork_ce_batches = 0
        self.batches = 0

    def record(
        self,
        *,
        considered: int,
        accepted: int,
        connector_hits: int,
        fork_ce_value: float | None,
    ) -> None:
        self.batches += 1
        self.considered += considered
        self.accepted += accepted
        self.connector_hits += connector_hits
        if fork_ce_value is not None:
            self.fork_ce_sum += fork_ce_value
            self.fork_ce_batches += 1

    def summary(self, *, lambda_fork: float) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "considered": self.considered,
            "accepted": self.accepted,
            "acceptance_rate": (self.accepted / self.considered) if self.considered else None,
            "connector_hits": self.connector_hits,
            "connector_fraction": (self.connector_hits / self.accepted) if self.accepted else None,
            "mean_fork_ce": (self.fork_ce_sum / self.fork_ce_batches) if self.fork_ce_batches else None,
            "lambda_fork": lambda_fork,
        }


def compute_fork_ce(
    forks: dict[str, Any] | None,
    labels: Tensor,
    *,
    num_aspects: int,
    aspect_class_counts: Sequence[int] | None,
) -> tuple[Tensor | None, dict[str, int]]:
    """Core fork-consistency CE, isolated from ``compute_loss`` for testing.

    Returns ``(fork_ce, stats)``.  ``fork_ce`` is ``None`` when there is
    nothing to accept (no forks stashed, no row with >=2 gold aspects, or no
    accepted split); otherwise it is the raw (un-multiplied-by-lambda) mean
    over accepted rows of ``0.5 * (CE(fork_1, target_1) + CE(fork_2,
    target_2))``, gradient-connected to the fork logits tensor.  Selection
    (score(s), argmax, gold->fork assignment) runs on detached, CPU-resident
    plain floats, so no gradient flows through it.
    """

    stats = {"considered": 0, "accepted": 0, "connector_hits": 0}
    if forks is None or not num_aspects:
        return None, stats

    counts = tuple(int(value) for value in (aspect_class_counts or ()))
    if len(set(counts)) != 1:
        raise ValueError("E18 fork loss assumes a uniform ABSA slot width")
    num_classes = int(counts[0])
    num_aspects = int(num_aspects)

    fork_logits = forks["logits"]  # [B, 1+2*S_max, A*C], gradient-carrying
    splits_per_row: list[list[int]] = forks["splits"]
    connector_flags: list[list[bool]] = forks["connector_flags"]
    batch_size = int(labels.shape[0])

    # Selection/scoring on detached, CPU-resident plain floats: no gradient
    # through argmax/assignment, and no per-element GPU sync.
    with torch.autocast(device_type=fork_logits.device.type, enabled=False):
        log_probs = F.log_softmax(
            fork_logits.detach().float().view(batch_size, -1, num_aspects, num_classes),
            dim=-1,
        )
    log_probs_list = log_probs.to("cpu").tolist()
    labels_list = labels.detach().to("cpu").tolist()

    row_losses: list[Tensor] = []
    for row in range(batch_size):
        splits = splits_per_row[row]
        if not splits:
            continue
        gold_row = labels_list[row]
        gold = [(a, int(gold_row[a])) for a in range(num_aspects) if int(gold_row[a]) > 0]
        if len(gold) < 2:
            continue
        stats["considered"] += 1

        row_lp = log_probs_list[row]
        base_score = sum(row_lp[0][a][y] for a, y in gold)
        best_score = base_score
        best_split = None
        for s in range(len(splits)):
            f1, f2 = 1 + 2 * s, 2 + 2 * s
            score = sum(max(row_lp[f1][a][y], row_lp[f2][a][y]) for a, y in gold)
            if score > best_score:
                best_score = score
                best_split = s
        if best_split is None:
            continue
        stats["accepted"] += 1
        if connector_flags[row][best_split]:
            stats["connector_hits"] += 1

        f1, f2 = 1 + 2 * best_split, 2 + 2 * best_split
        assign: dict[int, list[int]] = {f1: [], f2: []}
        for a, y in gold:
            chosen = f1 if row_lp[f1][a][y] >= row_lp[f2][a][y] else f2
            assign[chosen].append(a)

        per_fork_ce: list[Tensor] = []
        for fork_idx in (f1, f2):
            target = torch.zeros(num_aspects, dtype=torch.long, device=labels.device)
            for a in assign[fork_idx]:
                target[a] = int(gold_row[a])
            slot_logits = fork_logits[row, fork_idx].reshape(num_aspects, num_classes)
            per_fork_ce.append(F.cross_entropy(slot_logits, target))
        row_losses.append(0.5 * (per_fork_ce[0] + per_fork_ce[1]))

    if not row_losses:
        return None, stats
    return torch.stack(row_losses).mean(), stats


def make_patched_compute_loss(
    model: Any,
    *,
    original_compute_loss: Callable[..., Tensor],
    lambda_fork: float,
    telemetry: ForkTelemetry,
) -> Callable[..., Tensor]:
    """Return a drop-in replacement for ``goibert_e2.train.compute_loss``.

    ``model`` must expose ``last_forks``: set to a dict with keys
    ``logits``/``splits``/``sizes``/``connector_flags`` by
    ``E18Classifier.forward``'s fork-armed pass, and ``None`` on every other
    forward (b0_baseline, the noisy forward, or eval/validation).
    """

    def patched_compute_loss(
        logits: Tensor,
        labels: Tensor,
        *,
        num_aspects: int,
        aspect_class_counts: Sequence[int] | None = None,
    ) -> Tensor:
        ce = original_compute_loss(
            logits, labels, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts
        )
        forks = model.last_forks
        model.last_forks = None
        fork_ce, stats = compute_fork_ce(
            forks, labels, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts
        )
        if fork_ce is None:
            telemetry.record(
                considered=stats["considered"],
                accepted=stats["accepted"],
                connector_hits=stats["connector_hits"],
                fork_ce_value=None,
            )
            return ce
        telemetry.record(
            considered=stats["considered"],
            accepted=stats["accepted"],
            connector_hits=stats["connector_hits"],
            fork_ce_value=float(fork_ce.detach().cpu().item()),
        )
        return ce + lambda_fork * fork_ce

    return patched_compute_loss
