"""E18 model: segment-fork rows added to the frozen E2 student, K=1, no new parameters.

The construction follows ``SlotEncoder.forward`` in
``scripts/research_checks/competitive_slots_probe_v2.py`` (fork-row attention
math) and ``candidate_splits`` / ``collate_latent`` in
``competitive_slots_probe_v3_latent.py`` (split enumeration and mask layout),
restricted to K=1 (no extra CLS slots, only fork rows, no delta/qbias).

Row layout per forward pass, when forks are armed: ``[CLS] + [1 + 2*S_max
fork rows] + [tokens]``.  Token rows attend only to ``{CLS} u {tokens}``
(never forks); the CLS row attends only to ``{itself} u {tokens}`` (never
forks).  Because attention-key/value computation is row-independent
(pointwise linear projections) and both the CLS and token rows never read a
fork row as a key/value, the CLS/token computation subgraph is, by
construction, mathematically identical to the plain (no-fork) forward at
every layer -- the two forward passes only differ by floating point
summation order over a differently shaped, wider matmul.  Fork rows read
{own} u masked-tokens u {</s>} and never feed back into CLS/token rows.

No parameters are added: the fork rows reuse the exact same per-layer
Q/K/V/output/intermediate/output submodules and the exact same
classification head as the plain E2 student, so a checkpoint saved from this
module round-trips through ``build_e2_student(...).load_state_dict(...,
strict=True)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from goibert_e2.modeling import build_e2_student


ARMS = ("b0_baseline", "b1_fork", "b2_random_mask", "b3_fork_detach")
FORK_ARMS = ("b1_fork", "b2_random_mask", "b3_fork_detach")
S_MAX = 6
LAMBDA_FORK = 0.3
HIDDEN_SIZE = 384

# Connector/punctuation token ids for goiturn-s-gd, locked in docs/E18_PLAN.md:
# {",", ".", ";", "!", "?", " và", " nhưng", " còn", " mà", " tuy", " nên", " thì"}
CONNECTOR_IDS: frozenset[int] = frozenset(
    {15, 17, 30, 4, 34, 258, 643, 617, 572, 2238, 659, 656}
)


@dataclass(frozen=True)
class E18ModelSpec:
    """Serializable model identity used in E18 metadata/preflight."""

    role: str
    backbone: str
    hidden_size: int
    head_hidden_size: int
    head_dropout: float
    num_labels: int
    num_aspects: int
    aspect_class_counts: tuple[int, ...]
    total_parameters: int
    trainable_parameters: int
    arm: str
    s_max: int
    lambda_fork: float
    num_fork_rows: int
    added_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_splits(ids: Sequence[int], s_max: int) -> list[int]:
    """Enumerate candidate split positions ``j`` with segment1=[1,j), segment2=[j,n-1).

    Verbatim port of ``competitive_slots_probe_v3_latent.candidate_splits``:
    split points immediately after a connector/punctuation token, each
    segment keeping >=1 content token; falls back to evenly spaced positions
    when no connector-based split exists; capped at ``s_max``.
    """

    n = len(ids)
    cands = [
        j
        for j in range(2, n - 2)
        if ids[j - 1] in CONNECTOR_IDS and j >= 2 and n - 1 - j >= 1
    ]
    if not cands:
        cands = list(range(2, n - 2))
    if len(cands) > s_max:
        # Evenly spaced sub-selection; verbatim port of the reference
        # implementation's ``np.linspace(...).round().astype(int)`` index
        # selection, including its (harmless) potential for a duplicate
        # index at small s_max.
        idx = np.linspace(0, len(cands) - 1, s_max).round().astype(int)
        cands = [cands[i] for i in idx]
    return cands


def build_fork_masks(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    s_max: int = S_MAX,
    connector_ids: frozenset[int] = CONNECTOR_IDS,
) -> tuple[Tensor, list[list[int]], list[list[bool]]]:
    """Build the [B, 1+2*s_max, N-1] fork content-column mask (CPU, bool).

    Row 0 (full mask) and any unused split slot stay all-True (ignored
    downstream).  Row ``1+2*s``/``2+2*s`` are the two contiguous segments of
    split ``s``: ``{own} u segment tokens u {</s>}``; the final AND against
    token validity happens inside the model (so padding never needs to be
    handled here).  Mirrors ``collate_latent`` exactly, batched from an
    already-padded ``input_ids``/``attention_mask`` pair.
    """

    batch_size, seq_len = input_ids.shape
    m_max = 1 + 2 * s_max
    fork_mask = torch.ones((batch_size, m_max, seq_len - 1), dtype=torch.bool)
    ids_cpu = input_ids.detach().to("cpu")
    lengths = attention_mask.detach().to("cpu").sum(dim=1).tolist()
    splits_per_row: list[list[int]] = []
    connector_flags_per_row: list[list[bool]] = []
    for row in range(batch_size):
        n = int(lengths[row])
        row_ids = ids_cpu[row, :n].tolist()
        splits = candidate_splits(row_ids, s_max)
        splits_per_row.append(splits)
        connector_flags_per_row.append(
            [row_ids[j - 1] in connector_ids for j in splits]
        )
        eos = n - 1
        for s, j in enumerate(splits):
            fork_mask[row, 1 + 2 * s] = False
            fork_mask[row, 1 + 2 * s, 0 : j - 1] = True
            fork_mask[row, 1 + 2 * s, eos - 1] = True
            fork_mask[row, 2 + 2 * s] = False
            fork_mask[row, 2 + 2 * s, j - 1 : eos - 1] = True
            fork_mask[row, 2 + 2 * s, eos - 1] = True
    return fork_mask, splits_per_row, connector_flags_per_row


def randomize_fork_masks(
    fork_mask: Tensor,
    splits_per_row: list[list[int]],
    lengths: Sequence[int],
    *,
    seed: int,
) -> Tensor:
    """b2_random_mask control: complementary random token subsets, same sizes.

    Verbatim port of ``segment_read_ceiling.randomize_masks``: destroys
    contiguity, preserves per-fork size and total content coverage.
    """

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    fork_mask = fork_mask.clone().to("cpu")
    for row, splits in enumerate(splits_per_row):
        if not splits:
            continue
        n = int(lengths[row])
        eos = n - 1
        content = torch.arange(0, eos - 1)
        for s, j in enumerate(splits):
            n1 = j - 1
            perm = content[torch.randperm(len(content), generator=generator)]
            m1 = torch.zeros(fork_mask.shape[2], dtype=torch.bool)
            m1[perm[:n1]] = True
            m1[eos - 1] = True
            m2 = torch.zeros(fork_mask.shape[2], dtype=torch.bool)
            m2[perm[n1:]] = True
            m2[eos - 1] = True
            fork_mask[row, 1 + 2 * s] = m1
            fork_mask[row, 2 + 2 * s] = m2
    return fork_mask


def _apply_head(head: nn.Module, hidden: Tensor) -> Tensor:
    """Row-wise ``ClassificationHead`` application (no ``[:, 0]`` pooling).

    Identical arithmetic to ``ClassificationHead.forward`` once the caller
    has already selected the row(s) to pool; ``nn.Linear``/``nn.Dropout``
    broadcast over any number of leading dimensions.
    """

    hidden = head.input_dropout(hidden)
    hidden = head.dense(hidden)
    hidden = head.activation(hidden)
    hidden = head.dropout(hidden)
    return head.output(hidden)


def _fp32_masked_softmax(scores: Tensor, valid: Tensor) -> Tensor:
    """Softmax in FP32 regardless of the ambient autocast state."""

    with torch.autocast(device_type=scores.device.type, enabled=False):
        scores = scores.float().masked_fill(~valid, float("-inf"))
        return torch.softmax(scores, dim=-1)


def _attention_dropout_p(self_attn: nn.Module) -> float:
    """Read the self-attention module's attention-probability dropout rate.

    RoBERTa's SDPA self-attention stores this as a plain float
    ``dropout_prob`` (passed straight to ``scaled_dot_product_attention``,
    never wrapped in an ``nn.Dropout`` submodule); the eager implementation
    instead has an ``nn.Dropout`` submodule named ``dropout``.  Falls back to
    0.0 if neither is found.
    """

    prob = getattr(self_attn, "dropout_prob", None)
    if isinstance(prob, float):
        return prob
    dropout_module = getattr(self_attn, "dropout", None)
    if isinstance(dropout_module, nn.Dropout):
        return float(dropout_module.p)
    return 0.0


class E18Classifier(nn.Module):
    """Wraps a fresh ``build_e2_student`` model with an optional fork-forward.

    ``self.encoder``/``self.head`` are the *same* submodule objects the base
    E2 student was built with (re-parented, not copied), so
    ``state_dict()`` keys are identical to a plain E2 student's and no new
    parameters are registered.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        arm: str,
        s_max: int = S_MAX,
        lambda_fork: float = LAMBDA_FORK,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown E18 arm {arm!r}; expected one of {ARMS}")
        # Re-parent the exact same encoder/head modules so state_dict() is
        # byte-for-byte the same shape/keys as a plain E2 student.
        self.encoder = base.encoder
        self.head = base.head
        self.num_labels = int(base.num_labels)
        self.num_aspects = int(base.num_aspects)
        self.aspect_class_counts: tuple[int, ...] = tuple(base.aspect_class_counts)
        self.arm = arm
        self.s_max = int(s_max)
        self.lambda_fork = float(lambda_fork)
        self.num_heads = int(self.encoder.config.num_attention_heads)
        self.head_dim = int(self.encoder.config.hidden_size) // self.num_heads
        # Control state; never registered as a buffer/parameter, never
        # touches state_dict()/checkpoints.
        self.forks_armed = True
        self.last_forks: Optional[dict[str, Any]] = None
        self._fork_call_counter = 0
        self._base_seed = int(seed)

    # -- plain (no-fork) path -------------------------------------------------
    def _plain_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        kwargs: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs, return_dict=True)
        logits = self.head(outputs.last_hidden_state)
        return self._reshape_cls_logits(logits)

    def _reshape_cls_logits(self, flat_logits: Tensor) -> Tensor:
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            return flat_logits.reshape(
                flat_logits.shape[0], self.num_aspects, self.aspect_class_counts[0]
            )
        return flat_logits

    # -- fork-augmented path ---------------------------------------------------
    def _layer_with_forks(
        self,
        layer: nn.Module,
        x: Tensor,
        *,
        cls_valid: Tensor,
        tok_valid: Tensor,
        fork_valid: Tensor,
    ) -> Tensor:
        batch_size, total_rows, _ = x.shape
        self_attn = layer.attention.self
        heads, head_dim = self.num_heads, self.head_dim
        m = fork_valid.shape[1]
        scale = 1.0 / math.sqrt(head_dim)

        q = self_attn.query(x).view(batch_size, total_rows, heads, head_dim).transpose(1, 2)
        k = self_attn.key(x).view(batch_size, total_rows, heads, head_dim).transpose(1, 2)
        v = self_attn.value(x).view(batch_size, total_rows, heads, head_dim).transpose(1, 2)

        # Attention-probability dropout: the plain SDPA path applies this
        # (RobertaSdpaSelfAttention passes attention_probs_dropout_prob into
        # scaled_dot_product_attention whenever self.training); the manual
        # fork-path recompute below must match it exactly, on all three row
        # groups, or b1/b2's clean forward trains with strictly less
        # regularization than b0's -- a confound, not a real arm difference.
        dropout_p = _attention_dropout_p(self_attn)

        k_cls = k[:, :, 0:1, :]
        v_cls = v[:, :, 0:1, :]
        k_tok = k[:, :, 1 + m :, :]
        v_tok = v[:, :, 1 + m :, :]

        # ---- CLS row: {own} u tokens (never forks); identical in form to plain attention.
        q_cls = q[:, :, 0:1, :]
        s_own = (q_cls * k_cls).sum(-1, keepdim=True) * scale
        s_tok = torch.matmul(q_cls, k_tok.transpose(-1, -2)) * scale
        scores_cls = torch.cat([s_own, s_tok], dim=-1)
        probs_cls = _fp32_masked_softmax(scores_cls, cls_valid[:, None, None, :])
        probs_cls = F.dropout(probs_cls, p=dropout_p, training=self.training).to(v.dtype)
        ctx_cls = probs_cls[..., :1] * v_cls + torch.matmul(probs_cls[..., 1:], v_tok)

        # ---- Fork rows: {own} u masked tokens (per-row/per-split mask).
        q_fork = q[:, :, 1 : 1 + m, :]
        k_fork = k[:, :, 1 : 1 + m, :]
        v_fork = v[:, :, 1 : 1 + m, :]
        s_own_f = (q_fork * k_fork).sum(-1, keepdim=True) * scale
        s_tok_f = torch.matmul(q_fork, k_tok.transpose(-1, -2)) * scale
        scores_fork = torch.cat([s_own_f, s_tok_f], dim=-1)
        own_valid_f = torch.ones(batch_size, m, 1, dtype=torch.bool, device=x.device)
        valid_fork = torch.cat([own_valid_f, fork_valid], dim=-1)
        probs_fork = _fp32_masked_softmax(scores_fork, valid_fork[:, None, :, :])
        probs_fork = F.dropout(probs_fork, p=dropout_p, training=self.training).to(v.dtype)
        ctx_fork = probs_fork[..., :1] * v_fork + torch.matmul(probs_fork[..., 1:], v_tok)

        # ---- Token rows: {CLS} u tokens (never forks); identical to plain attention.
        q_tok = q[:, :, 1 + m :, :]
        s_cls_t = torch.matmul(q_tok, k_cls.transpose(-1, -2)) * scale
        s_tok_t = torch.matmul(q_tok, k_tok.transpose(-1, -2)) * scale
        scores_tok = torch.cat([s_cls_t, s_tok_t], dim=-1)
        probs_tok = _fp32_masked_softmax(scores_tok, cls_valid[:, None, None, :])
        probs_tok = F.dropout(probs_tok, p=dropout_p, training=self.training).to(v.dtype)
        ctx_tok = probs_tok[..., :1] * v_cls + torch.matmul(probs_tok[..., 1:], v_tok)

        ctx = torch.cat([ctx_cls, ctx_fork, ctx_tok], dim=2)
        ctx = ctx.transpose(1, 2).reshape(batch_size, total_rows, heads * head_dim)
        attn_out = layer.attention.output(ctx, x)
        inter = layer.intermediate(attn_out)
        return layer.output(inter, attn_out)

    def _forward_with_forks(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        self._fork_call_counter += 1
        fork_mask, splits_per_row, connector_flags = build_fork_masks(
            input_ids, attention_mask, s_max=self.s_max
        )
        lengths = attention_mask.detach().to("cpu").sum(dim=1).tolist()
        if self.arm == "b2_random_mask":
            call_seed = self._base_seed * 1_000_003 + self._fork_call_counter
            fork_mask = randomize_fork_masks(fork_mask, splits_per_row, lengths, seed=call_seed)
        fork_mask = fork_mask.to(input_ids.device)

        cls_valid = attention_mask.bool()
        tok_valid = attention_mask[:, 1:].bool()
        fork_valid = fork_mask & tok_valid[:, None, :]

        embeddings = self.encoder.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        batch_size, seq_len, hidden = embeddings.shape
        m = fork_mask.shape[1]
        cls_row = embeddings[:, 0:1, :]
        fork_rows = embeddings[:, 0:1, :].expand(batch_size, m, hidden)
        tok_rows = embeddings[:, 1:, :]
        x = torch.cat([cls_row, fork_rows, tok_rows], dim=1)

        for layer in self.encoder.encoder.layer:
            x = self._layer_with_forks(
                layer, x, cls_valid=cls_valid, tok_valid=tok_valid, fork_valid=fork_valid
            )

        cls_hidden = x[:, 0:1, :]
        fork_hidden = x[:, 1 : 1 + m, :]
        if self.arm == "b3_fork_detach":
            fork_hidden = fork_hidden.detach()

        cls_logits_flat = _apply_head(self.head, cls_hidden).squeeze(1)
        fork_logits_flat = _apply_head(self.head, fork_hidden)

        self.last_forks = {
            "logits": fork_logits_flat,
            "splits": splits_per_row,
            "sizes": [len(row) for row in splits_per_row],
            "connector_flags": connector_flags,
        }
        return self._reshape_cls_logits(cls_logits_flat)

    # -- public forward ---------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        compute_forks = (
            self.training and self.arm != "b0_baseline" and self.forks_armed
        )
        if compute_forks:
            logits = self._forward_with_forks(
                input_ids, attention_mask, token_type_ids=token_type_ids
            )
            self.forks_armed = False
            return logits
        # Plain path.  When training on a fork-capable arm this is exactly
        # the *second* (noisy) forward of the micro-batch: re-arm here so
        # the *next* micro-batch's clean forward computes forks again.  This
        # realises the "clean forward computes forks, noisy forward does
        # not, validation never does" invariant using only information
        # available inside forward() (see docs/E18_EXECUTION_STATUS.md for
        # why the literal "compute_loss re-arms" reading of the plan is not
        # used: compute_loss runs *before* the noisy forward, so re-arming
        # there would arm the noisy forward instead).
        if self.training and self.arm != "b0_baseline":
            self.forks_armed = True
        self.last_forks = None
        return self._plain_forward(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )

    def spec(self) -> E18ModelSpec:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return E18ModelSpec(
            role="e18_student",
            backbone="goiturn-s-gd",
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=int(self.head.dense.out_features),
            head_dropout=float(self.head.input_dropout.p),
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
            arm=self.arm,
            s_max=self.s_max,
            lambda_fork=self.lambda_fork,
            num_fork_rows=1 + 2 * self.s_max,
            added_parameter_count=0,
        )


def build_e18_student(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    arm: str,
    attention_implementation: str = "sdpa",
    s_max: int = S_MAX,
    lambda_fork: float = LAMBDA_FORK,
    seed: int = 42,
) -> tuple[E18Classifier, E18ModelSpec]:
    """Build a fresh E2-identical student, then wrap it for arm ``arm``."""

    base, _ = build_e2_student(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        attention_implementation=attention_implementation,
    )
    model = E18Classifier(base, arm=arm, s_max=s_max, lambda_fork=lambda_fork, seed=seed)
    return model, model.spec()
